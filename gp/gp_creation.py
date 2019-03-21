"""
Functions for GP creation and tuning.
"""

from gp import GP
from kernel import all_kernels

def create_specified_gp(domain, kernel_name, noise=None, pre_tune_pts=0,
                        **kwargs):
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
            kernel = k_info.obj(**kwargs)
    if kernel is None:
        raise ValueError('Kernel %s not found.' % kernel_name)
    # Make default noise small but positive. Helps with SPD conditions.
    default_noise = 0.001
    gp = GP(domain, kernel, default_noise)
    if pre_tune_pts > 0:
        gp = tune_gp(gp, num_pts=pre_tune_pts)
    return gp

def tune_gp(gp, pts=None, num_pts=None):
    """Tune the gp.
    Args:
        gp: The GP object.
        pts: List of lists representing the points.
        num_pts: Number of random points to be used if pts not specified.
    Returns: Tune GP.
    """
    raise NotImplementedError('TODO')
