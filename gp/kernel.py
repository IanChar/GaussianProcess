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
        self._hps = None
        if hps is None:
            self._set_default_hps()
        else:
            self.set_hps(hps)
        self._set_up(kwargs)

    def set_hps(self, hps):
        """Set the kernel hyperparameters."""
        hp_specs = self.get_hp_specs()
        if len(hps) != len(hp_specs):
            raise ValueError('HPs do not match specification.')
        for hp_info in hp_specs:
            if hp_info.name not in hps:
                raise ValueError('Missing hp: %s' % hp_info.name)
            hp_val = hps[hp_info.name]
            if hp_val > hp_info.upper and hp_val < hp_info.lower:
                raise ValueError('HP out of bound with val %f' % hp_val)
        self._hps = hps

    def get_hps(self):
        return self._hps

    def _set_default_hps(self):
        """Get default HPs if none are specified."""
        specs = self.get_hp_specs()
        self._hps = {}
        for hp_info in specs:
            self._hps[hp_info.name] = hp_info.default

    def __call__(self, x1, x2):
        """Evaluate kernel function."""
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()
        return self._make_comparison(x1, x2)

    @staticmethod
    def get_hp_specs():
        """Get HP specs which is a list of Namespace objects with each having:
            * name: Name of HP.
            * default: Default value.
            * lower: Lower bound for HP.
            * upper: Upper bound for HP.
        """
        raise NotImplementedError('To be implemented in child.')

    def _make_comparison(self, x1, x2):
        """Make comparison between two points, x1 and x2 (both ndarrays)."""
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

    @staticmethod
    def get_hp_specs():
        """Get HP specs which is a list of Namespace objects with each having:
            * name: Name of HP.
            * default: Default value.
            * lower: Lower bound for HP.
            * upper: Upper bound for HP.
        """
        return [\
            Namespace(name='bandwidth', default=0.1, lower=0.001, upper=1),
            Namespace(name='scale', default=1, lower=0.001, upper=1)
        ]

basic_kernels = [Namespace(name='sqexp', obj=SqExpKernel)]

all_kernels = basic_kernels
