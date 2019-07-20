"""
Main file for running experiments.
"""

import matplotlib.pyplot as plt
import numpy as np

from functions import get_function_info
from gp.gp_creation import create_gp, create_tuned_gp
from util.visuals import plot_oned_info, plot_mean_surface

def random_oned_visual(kernel_name, num_rand_pts, maxfs=None):
    """Create visual for one D functions where data is chosen randomly.
    Args:
        kernel_name: The name of the kernel to use.
        num_rand_pts: Number of random points to use.
        maxfs: Maximum number of GPs to build in tuning.
    """
    rand_x = [[x] for x in np.linspace(0, 1, num_rand_pts)]
    rand_y = [[np.random.uniform(-1, 1)] for _ in xrange(num_rand_pts)]
    gp = create_tuned_gp([[0, 1]], kernel_name, rand_x, rand_y, maxfs=maxfs)
    gp.add_observations(rand_x, rand_y)
    plot_oned_info(gp)
    plt.show()

def incremental_oned_viz(kernel_name, max_pts, maxfs=None):
    """Create visual for one D functions where data is chosen randomly.
    Args:
        kernel_name: The name of the kernel to use.
        num_rand_pts: Number of random points to use.
        maxfs: Maximum number of GPs to build in tuning.
    """
    rand_x = [[x] for x in np.linspace(0, 1, max_pts)]
    rand_y = [[np.random.uniform(-1, 1)] for _ in xrange(max_pts)]
    gp = create_tuned_gp([[0, 1]], kernel_name, rand_x, rand_y, maxfs=maxfs)
    for data_idx in range(max_pts):
        gp.add_observations([rand_x[data_idx]], [rand_y[data_idx]])
        plot_oned_info(gp)
        plt.show()

def plot_fitted_2d_func(kernel_name, func_name, num_evals, maxfs=50):
    """Create 2D surface plot for a specific function.
    Args:
        kernel_name: Name of the kernel to use.
        func_name: Name of the function to use.
        num_evals: Number of points to draw along each axis to eval func at.
        maxfs: Maximum number of GPs to build in tuning.
    """
    # Get function.
    f_info = get_function_info(func_name)
    # Get function evaluations.
    x1_pts = np.linspace(f_info.domain[0][0], f_info.domain[0][1], num_evals)
    x2_pts = np.linspace(f_info.domain[1][0], f_info.domain[1][1], num_evals)
    x1_pts, x2_pts = np.meshgrid(x1_pts, x2_pts)
    grid_pts = np.vstack([x1_pts.ravel(), x2_pts.ravel()]).T.tolist()
    f_evals = [f_info.function(x) for x in grid_pts]
    # Fit GP and plot.
    gp = create_tuned_gp(f_info.domain, kernel_name, grid_pts, f_evals, maxfs)
    gp.add_observations(grid_pts, f_evals)
    plot_mean_surface(gp)
    plt.show()
    print gp.get_log_marginal_likelihood()

def plot_pre_tuned_jeff(num_evals):
    """Create 2D surface plot for a specific function.
    Args:
        kernel_name: Name of the kernel to use.
        func_name: Name of the function to use.
        num_evals: Number of points to draw along each axis to eval func at.
    """
    # Get function.
    f_info = get_function_info('jeff')
    # Get function evaluations.
    x1_pts = np.linspace(f_info.domain[0][0], f_info.domain[0][1], num_evals)
    x2_pts = np.linspace(f_info.domain[1][0], f_info.domain[1][1], num_evals)
    x1_pts, x2_pts = np.meshgrid(x1_pts, x2_pts)
    grid_pts = np.vstack([x1_pts.ravel(), x2_pts.ravel()]).T.tolist()
    f_evals = [f_info.function(x) for x in grid_pts]
    # Fit GP and plot.
    gp = create_gp(f_info.domain, 'sqexp',
                   hps={'bandwidth': 0.5005, 'scale': 0.9445})
    gp.add_observations(grid_pts, f_evals)
    plot_mean_surface(gp)
    plt.show()
    print gp.get_log_marginal_likelihood()

if __name__ == '__main__':
    # import pudb; pudb.set_trace()
    incremental_oned_viz('sqexp', 3)
