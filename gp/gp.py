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
        self.domain = domain
        self.dim = len(domain)
        self.kernel = kernel
        self.noise = noise
        self.x_data = []
        self.y_data = []
        self._data_cov = None
        self._data_cov_inv = None
        self._cached_cho = None

    def add_observations(self, x_data, y_data):
        """ Add observations to the GP.
        Args:
            x_data: List of list of points.
            y_data: List of observations.
        """
        self.x_data += x_data
        self.y_data += y_data
        self._data_cov_inv = None
        self._cached_cho = None
        self._update_data_cov_mat(x_data)

    def get_posterior(self, pts, only_mean=False):
        """Get the posterior mean and (maybe) cov matrix.
        Args:
            pts: Points represented as list of lists.
            only_mean: Only calculate the mean without returning cov mat.
        """
        num_pts, num_data = len(pts), len(self.x_data)
        if num_pts == 1:
            return _get_single_pt_posterior(pts[0], only_mean)
        # Get matrix K(X_*, X) (i.e. covariances between seen points)
        k_star = self._get_data_interaction_mat(pts)
        # Get (K(X, X) + sigma I)^-1
        data_inv = self._get_cached_inverse()
        # Get posterior mean info.
        intermediate = np.dot(k_star, data_inv)
        mu = np.dot(intermediate, np.asarray(self.y_data))
        if only_mean:
            return mu
        # Get covariance matrix for new points.
        pt_cov = self._get_pt_cov_mat(pts)
        # Get posterior covariance info.
        cov = pt_cov - np.dot(intermediate, k_star.T)
        return mu, cov

    def draw_samples(self, num_samples, pts, mean=None, cov=None):
        """Draw a sample from the posterior.
        Args:
            num_samples: Number of samples to draw.
            pts: Points represented as list of lists.
            mean: The means of the points, if available.
            cov: The covariance matrix of the points, if available.
        """
        if mean is None or cov is None:
            mean, cov = self.get_posterior(pts)
        return np.random.multivariate_normal(mean, cov, size=num_samples)

    def _get_single_pt_posterior(self, pt, only_mean):
        """Get the posterior mean and (maybe) covariance
        Args:
            pt: Point as a list.
            only_mean: Only calculate the mean without returning cov mat.
        """
        k_star = self._get_data_interaction_mat([pt]).flatten()
        L = self._get_cholesky()
        alpha = la.cho_solve((L, True), np.asarray(self.y_data))
        v = la.solve_triangular(L, k_star, lower=True)
        mu = np.dot(k_star, alpha)
        if only_mean:
            return mu
        cov = self.kernel(pt, pt) - np.dot(v, v)
        return mu, cov

    def _update_data_cov_mat(self, x_data):
        """Update covariance matrix of seen observations."""
        pt_cov = self._get_pt_cov_mat(x_data)
        if self._data_cov is None:
            self._data_cov = pt_cov
            return self._data_cov
        k_star = self._get_data_interaction_mat(x_data)
        self._data_cov = np.block([[self._data_cov, k_star.T],
                                   [k_star, pt_cov]])
        return self._data_cov

    def _get_pt_cov_mat(self, pts):
        """Get covariance matrix between points."""
        num_pts = len(pts)
        pt_cov = np.ndarray((num_pts, num_pts))
        for i in xrange(num_pts):
            for j in xrange(i, num_pts):
                val = self.kernel(pts[i], pts[j])
                if i == j:
                    val /= 2
                pt_cov[i, j] = val
        pt_cov += pt_cov.T
        return pt_cov

    def _get_data_interaction_mat(self, pts):
        """Get covariance matrix between new points and old points."""
        num_pts, num_data = len(pts), len(self.x_data)
        k_star = np.ndarray((num_pts, num_data))
        for i in xrange(num_pts):
            for j in xrange(num_data):
                k_star[i, j] = self.kernel(pts[i], self.x_data[j])
        return k_star

    def _get_cholesky(self):
        """Get the cholesky decomposition of the data covariance matrix."""
        if self._cached_cho is None:
            self._cached_cho = la.cho_factor(self._kern_mat, lower=True)[0]
        return self._cached_cho

    def _get_cached_inverse(self):
        """Get or compute the return the inverse of data covariance."""
        if self._data_cov_inv is None:
            noise_offset = self.noise * np.eye(self._data_cov.shape[0])
            self._data_cov_inv = np.linalg.inv(self._data_cov + noise_offset)
        return self._data_cov_inv
