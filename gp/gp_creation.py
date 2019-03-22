"""
Functions for GP creation and tuning.
"""

from scipydirect import minimize as direct_min

from gp import GP
from kernel import all_kernels

def create_gp(domain, kernel_name, noise=None, hps=None, **kwargs):
    """Create GP with the specified kernel.
    Args:
        domain: List of lists [[dim1_low, dim1_high], ...]
        kernel_name: Name of kernel to use.
        noise: The amount of noise in the system. If None tune or make default.
        pre_tune_pts: The amount of points to use to learn the GP.
        kwargs: Other arguments to be passed to kernel.
    Returns: GP object.
    """
    kernel = None
    for k_info in all_kernels:
        if k_info.name.lower() == kernel_name.lower():
            kernel = k_info.obj(hps=hps, **kwargs)
    if kernel is None:
        raise ValueError('Kernel %s not found.' % kernel_name)
    # Make default noise small but positive. Helps with SPD conditions.
    default_noise = 0.01
    gp = GP(domain, kernel, default_noise)
    return gp

def create_tuned_gp(domain, kernel_name, x_data, y_data, maxfs=50, **kwargs):
    """Tune the gp.
    Args:
        gp: The GP object.
        pts: List of lists representing the points.
        num_pts: Number of random points to be used if pts not specified.
        maxfs: Maximum number of GPs to build in tuning.
    Returns: Tuned GP (note does not have data added to it).
    """
    kernel_creator = None
    for k_info in all_kernels:
        if k_info.name.lower() == kernel_name.lower():
            kernel_creator = k_info.obj
    if kernel_creator is None:
        raise ValueError('Kernel %s not found.' % kernel_name)
    hp_specs = kernel_creator.get_hp_specs()
    def objective(x):
        noise = x[0]
        hps = {}
        for idx in xrange(len(hp_specs)):
            hps[hp_specs[idx].name] = x[idx + 1]
        kernel = kernel_creator(hps=hps, **kwargs)
        gp = GP(domain, kernel, noise)
        gp.add_observations(x_data, y_data)
        return_val = -1 * gp.get_log_marginal_likelihood()
        return return_val
    bounds = [[0.0001, 1]] + [[hp_info.lower, hp_info.upper]
                               for hp_info in hp_specs]
    if maxfs is not None:
        best_specs = direct_min(objective, bounds, maxf=maxfs).x
    else:
        best_specs = direct_min(objective, bounds).x
    noise = best_specs[0]
    hps = {}
    for idx in xrange(len(hp_specs)):
        hps[hp_specs[idx].name] = best_specs[idx + 1]
    print hps
    kernel = kernel_creator(hps=hps, **kwargs)
    return GP(domain, kernel, noise)
