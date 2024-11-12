from typing import Literal, List
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mticker


def set_ticks(
    ax: Axes,
    xticks_mode: Literal['linear', 'log', None] = 'log',
    yticks_mode: Literal['linear', 'log', None] = 'log',
    yticks: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    yticklabels: list = None,
):
    # Set auto yticks
    if xticks_mode:
        if xticks_mode == 'linear':
            ax.xaxis.set_minor_locator(mticker.LinearLocator(numticks=5))
        elif xticks_mode == 'log':
            ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    
    # Set auto yticks
    if yticks_mode:
        if yticks_mode == 'linear':
            ax.yaxis.set_minor_locator(mticker.LinearLocator(numticks=5))
        elif yticks_mode == 'log':
            ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
            
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    
    # Set yticks and yticklabels
    if yticks:
        if not yticklabels:
            yticklabels = [f'{t:.2f}' for t in yticks]
        ax.set_yticks(ticks=yticks, labels=yticklabels)
        
    return ax


def save_figs(save_dir:str, fig_name:str, formats:List[str]):
    for fmt in formats:
        file_path = Path(save_dir) / fmt / fig_name
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            
        file_path = file_path.with_suffix(f'.{fmt}')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')