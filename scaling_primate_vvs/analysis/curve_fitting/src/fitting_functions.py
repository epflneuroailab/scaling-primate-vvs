import os
import sys
from pathlib import Path
from typing import Union, List, Tuple
from itertools import product
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import numba

from scipy.optimize import minimize

@numba.njit
def huber(delta, r):
    """
    """
    huber_mse = 0.5*r**2
    huber_mae = delta * (np.abs(r) - 0.5 * delta)
    condition = np.abs(r) <= delta
    return np.where(condition, huber_mse, huber_mae)

@numba.njit
def huber_loss_single_var(params, X, Y, delta, L, use_log=True):
    """
    Computes the Huber loss.
    """
    # Single variable scaling curve
    e, a, alpha = params
    Y_hat = L(X, e, a, alpha)
    if use_log:
        Y = np.log(Y)
    results = np.sum(huber(delta, Y_hat - Y).astype(np.float64))
    return results

@numba.njit
def huber_loss_single_var_shift(params, X, Y, delta, L, use_log=True):
    """
    Computes the Huber loss with shift parameter.
    """
    # Single variable scaling curve
    e, a, lambda_, alpha = params
    Y_hat = L(X, e, a, lambda_, alpha)
    if use_log:
        Y = np.log(Y)
    results = np.sum(huber(delta, Y_hat - Y).astype(np.float64))
    return results

@numba.njit
def huber_loss_double_var(params, X, Y, delta, L, use_log=True):
    """
    Computes the Huber loss. Unpacks X into N and D.
    """
    # Double variable scaling curve
    e, a, alpha, b, beta = params
    N, D = X[:, 0], X[:, 1]
    Y_hat = L(N, D, e, a, alpha, b, beta)
        
    if use_log:
        Y = np.log(Y)

    results = np.sum(huber(delta, Y_hat - Y).astype(np.float64))
    return results

@numba.njit
def L_power_law(x, E, A, alpha):
    """
    Computes the power law curve.
    """
    return E + A * np.power(x, alpha)

@numba.njit
def L_power_law_LSE(X, e, a, alpha):
    """
    Computes LSE term compatible with the power law curve.
    E = exp(e)
    A = exp(a)
    """
    return np.log(np.exp(a-alpha*np.log(X)) + np.exp(e))

@numba.njit
def L_power_law_shift(x, E, A, lambda_, alpha):
    """
    Computes the power law curve with shift.
    """
    return E + A * np.power(x + 10**lambda_, alpha)

@numba.njit
def L_power_law_shift_LSE(X, e, a, lambda_, alpha):
    """
    Computes LSE term compatible with the power law curve.
    lambda_ is the shift parameter. It is the base 10 logarithm of the shift value.
    E = exp(e)
    A = exp(a)
    """
    return np.log(np.exp(a-alpha*np.log(X+10**lambda_)) + np.exp(e))

@numba.njit
def L_Chinchilla(N, D, E, A, alpha, B, beta):
    """
    Computes the Chinchilla loss.
    """
    return E + A * np.power(N, alpha) + B * np.power(D, beta)

@numba.njit
def L_Chinchilla_LSE(N, D, e, a, alpha, b, beta):
    """
    Computes LSE term compatible with the Chinchilla model.
    E = exp(e)
    A = exp(a)
    B = exp(b)
    """
    return np.log(np.exp(a-alpha*np.log(N)) + np.exp(b-beta*np.log(D)) + np.exp(e))

def optimize_L(X, Y, delta, initial_params, L, use_log=True, method='BFGS'):
    """
    Optimizes the function L to minimize the Huber loss using BFGS algorithm.
    """
    n_params = len(initial_params)
    assert n_params in [3, 4, 5]
    if n_params == 3:
        huber_loss = huber_loss_single_var
    if n_params == 4:
        huber_loss = huber_loss_single_var_shift
    elif n_params == 5:
        huber_loss = huber_loss_double_var
    result = minimize(huber_loss, initial_params, args=(X, Y, delta, L, use_log), method=method)
    return result


def optimize_with_grid_search(X, Y, delta, initial_parameters, loss_func, use_log=True, method='BFGS', verbose=False, output_path=None, skip_existing=True):
    """
    Perform optimization using grid search over a set of initial parameters.
    
    Parameters:
        X : array-like
            The input data.
        Y : array-like
            The target data.
        delta : float
            A parameter used in the optimization process.
        initial_parameters : iterable
            An iterable of initial parameter sets to try.
        loss_func : callable
            The loss function to minimize.
        use_log : bool, optional
            Whether to use logarithmic scaling (default is True).
        method : str, optional
            The optimization method to use (default is 'BFGS').
        verbose : bool, optional
            Whether to display a progress bar (default is False).
        output_path : str, optional
            The output path to save the results (default is None).
        skip_existing : bool, optional
            Whether to skip existing output files (default is True).
            
    Returns:
        pd.DataFrame
            A DataFrame containing the losses and corresponding parameters, sorted by loss in ascending order.
    """
    
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and skip_existing:
            print(f"Output path {output_path} already exists. Skipping...")
            df = pd.read_csv(output_path)
            return df

    losses = []
    min_loss = np.inf
    
    if verbose:
        initial_parameters = list(initial_parameters)
        iterator = tqdm(initial_parameters, total=len(initial_parameters))
    else:
        iterator = initial_parameters
    
   
    for init_params in iterator:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = optimize_L(X, Y, delta, init_params, loss_func, use_log, method=method)
        optimized_params = results.x
        loss = results.fun
        if loss < min_loss:
            min_loss = loss
            if verbose:
                print(init_params, optimized_params, loss)
        losses.append(
            {
                'loss': loss,
                'init_params': init_params,
                'optimized_params': optimized_params
            }
        )

    df_losses = pd.DataFrame(losses)
    df_losses = df_losses.sort_values('loss', ascending=True)
    if output_path is not None:
        df_losses.to_csv(output_path, index=False)
    
    return df_losses

LOSS_FUNCTIONS = {
    'power_law': L_power_law,
    'power_law_shift': L_power_law_shift,
    'chinchilla': L_Chinchilla,
    'power_law_LSE': L_power_law_LSE,
    'power_law_shift_LSE': L_power_law_shift_LSE,
    'chinchilla_LSE': L_Chinchilla_LSE
}