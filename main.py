"""
Main file for running experiments.
"""

import matplotlib.pyplot as plt
import numpy as np

from gp.gp_creation import create_gp, create_tuned_gp
from util.visuals import plot_oned_info

def random_oned_visual(kernel_name, num_rand_pts):
    rand_x = [[x] for x in np.linspace(0, 1, num_rand_pts)]
    rand_y = [[np.random.uniform(-1, 1)] for _ in xrange(num_rand_pts)]
    gp = create_tuned_gp([[0, 1]], 'sqexp', rand_x, rand_y)
    gp.add_observations(rand_x, rand_y)
    plot_oned_info(gp)
    plt.show()

if __name__ == '__main__':
    import pudb; pudb.set_trace()
    random_oned_visual('sqexp', 5)
