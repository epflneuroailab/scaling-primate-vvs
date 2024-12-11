from pathlib import Path
from typing import Union, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import shutil

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .fitting_functions import optimize_with_grid_search, LOSS_FUNCTIONS
from .utils import get_args, apply_filters, load_yaml, get_bootstrapped_samples, prepare_data_for_fitting, get_md5_hash


if __name__ == '__main__':
    ### Parse arguments
    argparser = get_args()
    
    # argparser.add_argument(
    # )
    
    args = argparser.parse_args()
    args = vars(args)
    
    
    ### Load the experiment configuration for curve fitting
    config = load_yaml(args['experiment_config'])
    
    
    ### Load the results
    results_csv = args['results_csv']
    df_results = pd.read_csv(results_csv)
    
    ### Create output directory
    experiment_name = args.get('experiment_name', None) or config['experiment_name']
    output_dir = Path(args['output_dir']) / experiment_name
    artifact_dir = Path(args['artifact_dir']) / experiment_name
    overwrite = args.get('overwrite', False)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if artifact_dir.exists() and overwrite:
        shutil.rmtree(artifact_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    if not artifact_dir.exists():
        artifact_dir.mkdir(parents=True, exist_ok=False)
        
    ### Additional computations
    
    #### Apply filters to the dataframe
    df = apply_filters(df_results, config['data_filters'])
    
    
    ### Create bootstrapped samples
    fitting_parameters = config.get('fitting_parameters', {})
    num_bootstraps = args.get("num_bootstraps", None) or int(fitting_parameters.get('num_bootstraps', -1))
    assert num_bootstraps > 0, "Number of bootstraps must be greater than 0."
    random_state = int(fitting_parameters.get('random_state', 42))
    data_fraction = float(fitting_parameters.get('data_fraction', 1))
    bootstrap_samples = get_bootstrapped_samples(df, num_bootstraps, data_fraction, random_state)
    
    ### Prepare data for curve fitting
    delta = float(fitting_parameters.get('delta', None))
    use_log = fitting_parameters.get('use_log', True)
    method = fitting_parameters.get('method', 'BFGS')
    loss_func = LOSS_FUNCTIONS[fitting_parameters['loss_function']]
    
    ########### Fits ###########
    
    tasks = []
    all_results = []
    all_hashes = []
    num_workers = int(args.get('num_workers', 8))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # with ThreadPoolExecutor(max_workers=8) as executor:
    
        ### Fit to original data
        X_orig, Y_orig, initial_parameters = prepare_data_for_fitting(df, fitting_parameters)
        output_path = output_dir / f'original_fit.csv'
        optimize_with_grid_search(
            X=X_orig, 
            Y=Y_orig,
            delta=delta,
            initial_parameters=initial_parameters,
            loss_func=loss_func,
            use_log=use_log,
            method=method,
            verbose=False,
            output_path=output_path,
            skip_existing=True
        )
        ### Perform bootstrapped curve fitting
        for idx, df_bootstrap in bootstrap_samples.items():
            X_bootstrap, Y_bootstrap, initial_parameters = prepare_data_for_fitting(df_bootstrap, fitting_parameters)
            # rng = np.random.default_rng(random_state)
            # sigma = 1e-1
            # noise = rng.normal(0, sigma, size=Y_bootstrap.shape)
            # Y_bootstrap += noise
            output_path = artifact_dir / f'bootstrap_{idx}.csv'
            all_hashes.append(get_md5_hash(df_bootstrap))
            tasks.append(executor.submit(
                optimize_with_grid_search, 
                X=X_bootstrap, 
                Y=Y_bootstrap,
                delta=delta,
                initial_parameters=initial_parameters,
                loss_func=loss_func,
                use_log=use_log,
                method=method,
                verbose=False,
                output_path=output_path,
                skip_existing=True
            ))
            
        for res in tqdm(tasks):
            # all_results.extend(res.result())
            res.result()
    
    ### Collect the best fits from the bootstrapped samples
    best_fits = []
    for csv_file in artifact_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        bootstrap_id = int(csv_file.stem.split('_')[-1])
        df['bootstrap_id'] = bootstrap_id
        df.loss = df.loss.astype(float)
        df = df.sort_values('loss')
        best_params = df.iloc[0]
        best_fits.append(best_params)
    assert len(best_fits) == num_bootstraps, f"Expected {num_bootstraps} fits, got {len(best_fits)}"
        
    ### Save the best fits
    df_best_fits = pd.DataFrame(best_fits)
    df_best_fits = df_best_fits.sort_values('loss')
    df_best_fits.to_csv(output_dir / 'best_fits.csv', index=False)
    
    ### Convert string of floats to list of floats
    df_best_fits.optimized_params = df_best_fits.optimized_params.apply(lambda x: [float(y) for y in x[1:-1].split()])
    df_best_fits.init_params = df_best_fits.init_params.apply(lambda x: [float(y) for y in x[1:-1].split()])
    
    ### Load original fit
    df_original_fit = pd.read_csv(output_dir / 'original_fit.csv')
    df_original_fit.loss = df_original_fit.loss.astype(float)
    df_original_fit = df_original_fit.sort_values('loss')
    loss, optimized_params, init_params = df_original_fit.iloc[0][['loss', 'optimized_params', 'init_params']]
    optimized_params = [float(y) for y in optimized_params[1:-1].split()]
    init_params = [float(y) for y in init_params[1:-1].split()]    
    
    
    results_dict = {
        'args': args,
        'config': config,
        'loss': loss,
        'optimized_parameters': optimized_params,
        'initial_parameters': init_params,
        'optimized_parameters_bootstrapped': df_best_fits.optimized_params.tolist(),
        'losses_bootstrapped': df_best_fits.loss.tolist(),
        'initial_parameters_bootstrapped':  df_best_fits.init_params.tolist(),
        'config_hash': get_md5_hash(config),
        'data_hash': get_md5_hash(df),
        'results_hash': get_md5_hash(df_best_fits),
        'bootstrap_hashes': all_hashes,
    }
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)