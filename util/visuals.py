"""
Visual utils.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_oned_info(gp, fidelity=50, num_samps=3, ax=None):
    """Plot info for the one D function.
    Args:
        gp: The GP object.
        fidelity: Number of points to draw for curves.
        num_samps: Number of samples to plot.
        ax: pyplot ax object. if None create one.
    """
    if gp.dim != 1:
        raise ValueError('GP needs to be 1-Dimenion, got dimension %d' % gp.dim)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # Plot the observations.
    x_pts = [dat[0] for dat in gp.x_data]
    ax.scatter(gp.x_data, gp.y_data, marker='X', color='k')
    # Plot the mean curve.
    samp_pts = np.linspace(0, 1, fidelity)
    samp_data = [[pt] for pt in samp_pts]
    mu, cov = gp.get_posterior(samp_data)
    mu = mu.flatten()
    ax.plot(samp_pts, mu, color='k')
    # Plot the high confidence region.
    stds = np.sqrt(cov.diagonal())
    upper, lower = mu + 2 * stds, mu - 2 * stds
    ax.fill_between(samp_pts, lower, upper, where=lower <= upper,
                    facecolor='grey', interpolate=True, alpha=0.5)
    # Plot samples.
    for _ in xrange(num_samps):
        samp_ys = gp.draw_samples(1, samp_pts, mean=mu, cov=cov).flatten()
        ax.plot(samp_pts, samp_ys)
    return ax

def plot_mean_surface(gp, fidelity=50, ax=None):
    """Plot surface of posterior mean.
    Args:
        gp: GP object
        fidelity: Number of points to draw for each axis.
        ax: pyplot ax object. If None create one.
    Returns: ax object.
    """
    pass
