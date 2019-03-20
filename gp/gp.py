"""
Implementation for standard Euclidean GP.
"""

import numpy as np
import scipy.linalg as la

class GP(object):
    """Basic GP implementation."""

    def __init__(self, domain, kernel, noise):
        """Constructor.
        Args:
            domain: List of lists [[dim1_low, dim1_high],...]
            kernel: Kernel object.
            noise: The noise in the system.
        """
        pass

    def add_observations(self, x_data, y_data):
        """ Add observations to the GP.
        Args:
            x_data: ndarray of x points with dim (num obs x X dimension)
            y_data: 1 dimensional ndarray of y obvservations.
        """
        pass

    def get_posterior(self, pts, only_mean=False):
        """Get the posterior mean and (maybe) cov matrix.
        Args:
            pts: ndarray of points to evaluate at (num pts x dimension) or
                (dimension) if there is only one point.
            only_mean: Only calculate the mean without returning cov mat.
        """
        pass

    def draw_sample(self, pts):
        """Draw a sample from the posterior.
        Args:
            pts: ndarray of points to draw at (num pts x dimension) or
                (dimension) if there is only one point.
        """
        pass

    def _get_single_pt_posterior(self, pt, only_mean):
        """Get the posterior mean and (maybe) covariance
        Args:
            pt: 1 dimensional ndarray of the single point.
            only_mean: Only calculate the mean without returning cov mat.
        """
        pass

    def _update_data_cov_mat(self):
        """Update covariance matrix of seen observations."""
        pass

    def _compute_cholesky(self):
        """Get the cholesky decomposition of the data covariance matrix."""
        pass
