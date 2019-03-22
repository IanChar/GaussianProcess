"""
Two dimensional functions to use.
"""
from __future__ import division

from argparse import Namespace

import numpy as np

def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return -1 * (a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 \
                 + s * (1 - t) * np.cos(x[0]) + s)

def jeff(x):
    return x[0] * (np.sin(x[0] * x[1]) + x[1] / 4)

twod_functions = [\
    Namespace(function=branin, domain=[[-5, 10], [0, 15]], max_val=-0.397887,
              name='branin'),
    Namespace(function=jeff, domain=[[-1, 3], [-1, 3]],
              max_val=4.973906864710533, name='jeff'),
]
