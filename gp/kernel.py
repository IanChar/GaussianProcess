"""
Kernel parent class and its implementations.
"""

class Kernel(object):

    def __init__(self, hps=None, **kwargs):
        """Constructor.
        Args:
            hps: Dictionary of hyperparameters for the kernel.
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

    def __call__(self, a, b):
        """Evaluate kernel function."""
        return self._make_comparison(a, b)

    def _make_comparison(self, a, b):
        """Make comparison between two points, a and b (both ndarrays)."""
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

    def _make_comparison(self, a, b):
        """Make comparison between two points, a and b (both ndarrays)."""
        pass

    def _get_default_hps(self):
        """Get default HPs if none are specified."""
        pass


