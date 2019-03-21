"""
Kernel parent class and its implementations.
"""

from argparse import Namespace

import numpy as np

class Kernel(object):

    def __init__(self, hps=None, **kwargs):
        """Constructor.
        Args:
            hps: Dictionary of hyperparameters for the kernel.
                If None, get default hps.
        """
        if hps is None:
            self._hps = self._get_default_hps()
        else:
            self._hps = hps
        self._set_up(kwargs)

    def update_hps(self, hps):
        self._hps = hps

    def get_hps(self):
        return self._hps

    def __call__(self, x1, x2):
        """Evaluate kernel function."""
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()
        return self._make_comparison(x1, x2)

    def _make_comparison(self, x1, x2):
        """Make comparison between two points, x1 and x2 (both ndarrays)."""
        raise NotImplementedError('To be implemented in child.')

    def _get_default_hps(self):
        """Get default HPs if none are specified."""
        raise NotImplementedError('To be implemented in child.')

    def _set_up(self, kwargs):
        """Any additional set up should go here."""
        pass

"""
--------------------------- IMPLEMENTATIONS ------------------------------
"""

class SqExpKernel(Kernel):
    """Squared exponential kernel.
    Hyperparameters: [scale, bandwidth]
    """

    def _make_comparison(self, x1, x2):
        """Make comparison between two points, x1 and x2 (both ndarrays)."""
        b, s = self._hps['bandwidth'], self._hps['scale']
        scale = s ** 2
        return scale * np.exp(-1 * np.linalg.norm(x1 - x2) ** 2 / (2 * b ** 2))

    def _get_default_hps(self):
        """Get default HPs if none are specified."""
        return {'bandwidth': 0.1, 'scale': 1}


basic_kernels = [Namespace(name='sqexp', obj=SqExpKernel)]

all_kernels = basic_kernels
