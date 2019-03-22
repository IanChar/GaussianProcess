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
        self._cho = None
        self._alpha = None

    def add_observations(self, x_data, y_data):
        """ Add observations to the GP.
        Args:
            x_data: List of list of points.
            y_data: List of observations.
        """
        self.x_data += x_data
        self.y_data += y_data
        self._cho = None
        self._alpha = None
        self._update_data_cov_mat(x_data)

    def get_posterior(self, pts, only_mean=False):
        """Get the posterior mean and (maybe) cov matrix.
        Args:
            pts: Points represented as list of lists.
            only_mean: Only calculate the mean without returning cov mat.
        """
        k_star = self._get_data_interaction_mat(pts)
        if len(pts) == 1:
            k_star = k_star.flatten()
        L, alpha = self._get_posterior_help()
        mu = np.dot(k_star, alpha)
        if only_mean:
            return mu
        v = la.solve_triangular(L, k_star.T, lower=True)
        pt_cov = self._get_pt_cov_mat(pts)
        cov = pt_cov - np.dot(v.T, v)
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

    def get_log_marginal_likelihood(self):
        """Get the log marginal likelihood for observations."""
        L, alpha = self._get_posterior_help()
        y = np.asarray(self.y_data).flatten()
        fit_term = np.dot(y, alpha)
        penalty_term = np.sum(np.log(L.diagonal()))
        regularizer = len(self.x_data) * np.log(2 * np.pi)
        return -0.5 * (fit_term + penalty_term + regularizer)

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
        pt_cov = np.zeros((num_pts, num_pts))
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

    def _get_posterior_help(self):
        """Get the cholesky decomposition of the data covariance matrix."""
        if self._cho is None or self._alpha is None:
            kern_mat = self._data_cov + self.noise * np.eye(len(self.x_data))
            self._cho = la.cho_factor(kern_mat, lower=True)[0]
            self._alpha = la.cho_solve((self._cho, True),
                                        np.asarray(self.y_data))
        return self._cho, self._alpha.flatten()
