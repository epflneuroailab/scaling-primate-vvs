from typing import List, Union, Callable, Tuple
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


def get_formula_text_univariate(a, b, d, X='D',):
    return f"$S = {{{1-a:.2f}}} - {{{b:.2f}}}{X}^{{ {d:.2f} }}$"

def get_formula_text_univariate_double_saturation(a, b, c, d, X='C', x_scale_multiplier=1, show_x_scale_multiplier=True):
    base = f"${X}+{10}^{{ {c:.2f} }}$"
    m = np.log10(x_scale_multiplier).astype(int)
    if x_scale_multiplier != 1 and show_x_scale_multiplier:
        # return rf"$S = {{{1-a:.2f}}} - {{{b:.2f}}}({X}\div {10}^{{ {m} }}+{10}^{{ {c:.2f} }})^{{ {d:.2f} }}$"
        return rf"$S = {{{1-a:.2f}}} - {10}^{{{np.log10(b):.2f}}}({X}\div {10}^{{ {m} }}+{10}^{{ {c:.2f} }})^{{ {d:.2f} }}$"
    else:
        # return f"$S = {{{1-a:.2f}}} - {{{b:.2f}}}({X}+{10}^{{ {c:.2f} }})^{{ {d:.2f} }}$"
        return f"$S = {{{1-a:.2f}}} - {10}^{{{np.log10(b):.2f}}}({X}+{10}^{{ {c:.2f} }})^{{ {d:.2f} }}$"

def plot_reg(X, params, L, ax, color, invert_y:bool=True, X_str='D', x_extend=1, x_scaler=1, linestyle='--', show_x_scaler=True, linewidth=4, legend='auto', alpha=0.5, marker=None, marker_size=0.5):
    """
    Plots a regression line on the given axes.
    
    Parameters:
        X (array-like): The input data for the x-axis.
        params (list or array-like): Parameters for the regression function.
        L (callable): The regression function.
        ax (matplotlib.axes.Axes): The axes on which to plot the regression line.
        color (str): Color of the regression line.
        X_str (str, optional): Label for the x-axis variable in the formula. Default is 'D'.
        x_extend (float, optional): Multiplier for the maximum value of X for prediction. Default is 1.
        x_scaler (float, optional): Multiplier for scaling the x-axis values. Default is 1.
        linestyle (str, optional): Line style for the regression line. Default is '--'.
        show_x_scaler (bool, optional): Whether to show the x-axis scale multiplier in the formula. Default is True.
        linewidth (float, optional): Width of the regression line. Default is 4.
        legend (str, optional): Legend option for the plot. Default is 'auto'.
        alpha (float, optional): Transparency level of the regression line. Default is 0.5.
    
    Raises:
        ValueError: If the number of parameters is not 3 or 4.
    
    Returns:
        None
    """


    x_pred = np.geomspace(X.min(), X.max()*x_extend, 100)
    y_pred = L(x_pred, *params)
    if len(params) == 3:
        formula = get_formula_text_univariate(*params, X=X_str)
    elif len(params) == 4:
        formula = get_formula_text_univariate_double_saturation(*params, X=X_str, x_scale_multiplier=x_scaler, show_x_scale_multiplier=show_x_scaler)
    else:
        raise ValueError("Invalid number of parameters")
    # print(formula)
    if invert_y:
        y_pred = 1-y_pred
    sns.lineplot(x=x_pred*x_scaler, y=y_pred, linestyle=linestyle, label=formula, ax=ax, color=color, alpha=alpha, linewidth=linewidth , dashes='-.', legend=legend, marker=marker)
    
def get_formula_text_bivariate(E, A, alpha, B, beta, X_str='N', Y_str='D'):
    # print(E, A, alpha, B, beta)
    # return f"$S = {{{1-E:.2f}}}-\\frac{{{A:.2f}}}{{{X_str}^{{ {alpha:.2f} }}}} - \\frac{{{B:.2f}}}{{{Y_str}^{{ {beta:.2f} }}}}$"
    return rf"$S = {{{1-E:.2f}}}-\frac{{{A:.2f}}}{{{X_str}^{{ {alpha:.2f} }}}} - \frac{{{B:.2f}}}{{{Y_str}^{{ {beta:.2f} }}}}$"

def plot_reg_bivariate(scaling_coeffs, opt_params, L, ax, color, X_str='N', Y_str='D', x1_scale_multiplier=1, x2_scale_multiplier=1, linewidth=1, alpha=0.7, linestyle='-'):
    """
    Plots a bivariate regression line on the given axes.
    
    Parameters:
        scaling_coeffs (tuple): A tuple containing the scaling coefficients (a, b, G, m, n).
        opt_params (tuple): A tuple containing the optimization parameters (E, A, alpha, B, beta).
        L (function): A function that computes the predicted values.
        ax (matplotlib.axes.Axes): The axes on which to plot the regression line.
        color (str): The color of the regression line.
        X_str (str, optional): The label for the X-axis variable. Default is 'N'.
        Y_str (str, optional): The label for the Y-axis variable. Default is 'D'.
        x1_scale_multiplier (float, optional): The scale multiplier for the first X-axis variable. Default is 1.
        x2_scale_multiplier (float, optional): The scale multiplier for the second X-axis variable. Default is 1.
    
    Returns:
        None
    """
    
    compute = np.geomspace(1e14, 3e20, 1000)
    compute = compute / (x1_scale_multiplier*x2_scale_multiplier)
    # compute = np.geomspace(5e13, 1e20, 1000) / (x1_scale_multiplier*x2_scale_multiplier)
    a, b, G, m, n = scaling_coeffs
    X = np.power(compute/m, a/n) * G # N, params
    Y = np.power(compute/m, b/n) / G # D, samples
    
    y_pred = np.array([L(N, D, *opt_params) for N, D in zip(X, Y)])
    formula = get_formula_text_bivariate(*opt_params, X_str, Y_str)
    sns.lineplot(x=compute*(x1_scale_multiplier*x2_scale_multiplier), y=1-y_pred, linestyle=linestyle, label=formula, ax=ax, color=color, alpha=alpha, linewidth=linewidth)



def plot_confidence_intervals(
    X: np.ndarray, 
    optimized_parameters: Union[List[List[float]], np.ndarray], 
    L:Callable, 
    ax: plt.Axes, 
    color: str, 
    x_extend: float = 1.0, 
    x_scaler: float = 1.0,
    is_chinchilla: bool = False,
    scaling_coeffs: Tuple[float, float, float, float, float] = None,
    alpha=0.1,
    percentile: float = 95.0,
    invert_y: bool = True
) -> None:
    """
    Plots confidence intervals for a given set of optimized parameters.
    
    Parameters:
        X (np.ndarray): Array of x-values.
        optimized_parameters (Union[List[List[float]], np.ndarray]): List or array of optimized parameters.
        L (Callable): Function to compute y-values given x and parameters.
        ax (plt.Axes): Matplotlib Axes object to plot on.
        color (str): Color for the confidence interval fill.
        x_extend (float, optional): Factor to extend the x-axis range. Default is 10.0.
        x_scaler (float, optional): Multiplier to scale the x-axis values. Default is 1.0.
        is_chinchilla (bool, optional): Whether to use chinchilla scaling. Default is False.
        scaling_coeffs (Tuple[float, float, float, float, float], optional): Scaling coefficients for chinchilla scaling. Default is None.
        alpha (float, optional): Alpha value for the fill color transparency. Default is 0.1.
        percentile (float, optional): Percentile for the confidence interval. Default is 95.0.
        invert_y (bool, optional): Whether to invert the y-values. Default is True.
    
    Returns:
        None
    """
    
    if is_chinchilla:
        compute = X / x_scaler
        x_pred = compute
        y_preds = np.empty((len(optimized_parameters), len(compute)))

        for idx, (opt_params, scaling_coeff) in enumerate(zip(optimized_parameters, scaling_coeffs)):
            aa, bb, G, m, n = scaling_coeff
        
            X = np.power(compute/m, aa/n) * G # N, params
            Y = np.power(compute/m, bb/n) / G # D, samples
        
            y = np.array([L(N, D, *opt_params) for N, D in zip(X, Y)])
            y = np.array(y)
            if invert_y:
                y = 1-y
            y_preds[idx] = y
            
            
    else:
        x_pred = np.geomspace(X.min(), X.max()*x_extend, 100)
        y_preds = np.empty((len(optimized_parameters), len(x_pred)))
        
        for idx, opt_params in enumerate(optimized_parameters):
            y = [L(x, *opt_params) for x in x_pred]
            y = np.array(y)
            if invert_y:
                y = 1-y
            y_preds[idx] = y

    gamma = (100 - percentile)/2
    y_min = np.percentile(y_preds, gamma, axis=0)
    y_max = np.percentile(y_preds, 100-gamma, axis=0)

    ax.fill_between(x_pred*x_scaler, y_min, y_max, color=color, alpha=alpha)