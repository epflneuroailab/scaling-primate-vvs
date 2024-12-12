from typing import List, Tuple, Union
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats as stats


def get_bootstrapped_samples(df: pd.DataFrame, num_bootstraps: int, data_fraction:float = 1, random_state:int = 42) -> List[pd.DataFrame]:
    """
    Generates bootstrap samples from the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to be bootstrapped.
        num_bootstraps (int): The number of bootstrap samples to generate.
        data_fraction (float): The fraction of data to use for each bootstrap sample.
        random_state (int): The random seed to use for reproducibility.
        
    Returns:
        Dict[int, pd.DataFrame]: A dictionary containing the bootstrap samples.
    """
    bootstrap_samples = {}
    rng = np.random.default_rng(random_state)
    num_points = int(len(df)*data_fraction)
    random_indices = rng.integers(0, len(df), size=(num_bootstraps, num_points))
    for idx, indices in enumerate(random_indices):
        bootstrap_samples[idx] = df.iloc[indices].copy()
    return bootstrap_samples


def prepare_data_for_fitting(df: pd.DataFrame, fitting_params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares data for curve fitting based on the provided fitting parameters.
    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    fitting_params (dict): A dictionary containing the fitting parameters. 
        Expected keys:
        - 'curve_type' (str): Type of curve fitting ('one_variable' or 'two_variables').
        - 'X' (str): Column name for the independent variable (used if 'curve_type' is 'one_variable').
        - 'X1' (str): Column name for the first independent variable (used if 'curve_type' is 'two_variables').
        - 'X2' (str): Column name for the second independent variable (used if 'curve_type' is 'two_variables').
        - 'Y' (str): Column name for the dependent variable.
        - 'X_scaler' (float, optional): Scaling factor for the independent variable (default is 1).
        - 'X1_scaler' (float, optional): Scaling factor for the first independent variable (default is 1).
        - 'X2_scaler' (float, optional): Scaling factor for the second independent variable (default is 1).
        - 'initial_parameters' (list of lists): Initial parameters for the fitting process.
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - X (np.ndarray): Scaled independent variable(s).
        - Y (np.ndarray): Transformed dependent variable.
        - initial_params (np.ndarray): Initial parameters for the fitting process.
    """

    fitting_type = fitting_params.get('curve_type', 'one_variable')
    if fitting_type == 'one_variable':
        X_scaler = float(fitting_params.get('X_scaler', 1))
        X = df[fitting_params['X']].values
        X = X / X_scaler
    elif fitting_type == 'two_variables':
        X1_scaler = float(fitting_params.get('X1_scaler', 1))
        X2_scaler = float(fitting_params.get('X2_scaler', 1))
        X1 = df[fitting_params['X1']].values
        X2 = df[fitting_params['X2']].values
        X1 = X1 / X1_scaler
        X2 = X2 / X2_scaler
        X = np.column_stack([X1, X2])
    else:
        raise ValueError(f"Unknown fitting type: {fitting_type}")

    Y = df[fitting_params['Y']].values
    invert_Y = fitting_params.get('invert_Y', True)
    if invert_Y:
        Y = 1 - Y

    initial_params = fitting_params.get('initial_parameters', {})
    initial_params = list(product(*initial_params.values()))
    initial_params = np.array(initial_params).astype(float)
    
    return X, Y, initial_params



def compute_scaling_law_coeffs(params: Tuple[float], data:Tuple[np.ndarray], overwrite_n:bool=False):

    """
    Compute the scaling law coefficients for a given set of parameters and data.
    Parameters:
        params (tuple): A tuple containing the parameters (E, A, alpha, B, beta).
        data (tuple): A tuple containing the data (N, D, C).
        overwrite_n (bool, optional): If True, overwrite the computed slope 'n' with 1. Default is False.
    Returns:
        tuple: A tuple containing the computed coefficients (a, b, G, m, n).
    """
    
    E, A, alpha, B, beta = params # optimized_params_Chinchilla
    a = beta / (alpha + beta) # Allocation of FLOPs to parameters
    b = alpha / (alpha + beta) # Allocation of FLOPs to samples
    G = (alpha*A /(B*beta))**(1/(alpha+beta))
        
    # x_data, x_params, x_flops = data[['n_samples_seen', 'n_params', 'total_flops']].to_numpy().T
    N, D, C = data


    # Fit a power law to better characterize the relationship between 
    # the number of parameters, samples, and FLOPs
    # In Chinchilla and Kaplan's paper, they estimate C=6ND
    # for transformer models but it does not hold for all model architectures
    res = stats.linregress(np.log10(N*D), np.log10(C))

    n, m = res.slope, 10**res.intercept
    
    if overwrite_n:
        print(f"Overwriting n={n:.4f} to 1")
        n = 1

    return a, b, G, m, n

def convert_loss_parameters(params: Union[List[float], Tuple[float]], src_loss: str, dst_loss: str) -> Tuple[float]:
    """
    Converts a set of parameters from one loss function parameterization to another.
    Parameters:
    ----------
    params : Union[List[float], Tuple[float]]
        A list or tuple containing the parameters of the source loss function. The exact 
        structure depends on the type of `source_loss` provided.
    src_loss : str
        The name of the source loss function from which parameters are being converted. 
        Supported values include 'power_law', 'power_law_LSE', 'power_law_shift', 
        'power_law_shift_LSE', 'chinchilla', and 'chinchilla_LSE'.
    dst_loss : str
        The name of the destination loss function to which parameters are being converted. 
        Supported values are the same as `source_loss`.

    Returns:
    -------
    Tuple[float]
        A tuple containing the converted parameters for the destination loss function.

    Raises:
    ------
    ValueError
        If the combination of `src_loss` and `dst_loss` is invalid or unsupported.

    """

    if src_loss == dst_loss:
        return params
    
    elif (src_loss == 'power_law_LSE') and (dst_loss == 'power_law'):
        e, a, alpha = params
        E, A = np.exp(e), np.exp(a)
        alpha = -alpha
        return E, A, alpha
    
    elif (src_loss == 'power_law') and (dst_loss == 'power_law_LSE'):
        E, A, alpha = params
        e, a = np.log(E), np.log(A)
        alpha = -alpha
        return e, a, alpha
    
    elif (src_loss == 'power_law_shift_LSE') and (dst_loss == 'power_law_shift'):
        e, a, lambda_, alpha = params
        E, A = np.exp(e), np.exp(a)
        alpha = -alpha
        return E, A, lambda_, alpha
    
    elif (src_loss == 'power_law_shift') and (dst_loss == 'power_law_shift_LSE'):
        E, A, lambda_, alpha = params
        e, a = np.log(E), np.log(A)
        alpha = -alpha
        return e, a, lambda_, alpha
    
    elif (src_loss == 'chinchilla_LSE') and (dst_loss == 'chinchilla'):
        e, a, alpha, b, beta = params
        E, A, b = np.exp(e), np.exp(a), np.exp(b)
        alpha, beta = -alpha, -beta
        return E, A, alpha, b, beta
    
    elif (src_loss == 'chinchilla') and (dst_loss == 'chinchilla_LSE'):
        E, A, alpha, b, beta = params
        e, a, b = np.log(E), np.log(A), np.log(b)
        alpha, beta = -alpha, -beta
        return e, a, alpha, b, beta
    
    else:
        raise ValueError(f'Invalid source and destination loss functions: {src_loss}, {dst_loss}')

def convert_loss_parameters_batch(params: List[Union[List[float], Tuple[float]]], src_loss: str, dst_loss: str) -> List[Tuple[float]]:
    """
    Converts a batch of parameters from one loss function parameterization to another.
    Parameters:
    ----------
    params : List[Union[List[float], Tuple[float]]]
        A list of lists or tuples containing the parameters of the source loss function. The exact 
        structure depends on the type of `source_loss` provided.
    src_loss : str
        The name of the source loss function from which parameters are being converted. 
    dst_loss : str
        The name of the destination loss function to which parameters are being converted. 

    Returns:
    -------
    List[Tuple[float]]
        A list containing the converted parameters for the destination loss function.

    Raises:
    ------
    ValueError
        If the combination of `src_loss` and `dst_loss` is invalid or unsupported.

    """
    new_params = [convert_loss_parameters(p, src_loss, dst_loss) for p in params]
    new_params = np.array(new_params)
    if np.any(np.isinf(new_params)):
        print("Detected infinite values in the parameters")
        new_params = new_params[~np.isinf(new_params).any(axis=1),:]
        print(f"Removed {len(params) - len(new_params)} rows with infinite values")
    return new_params