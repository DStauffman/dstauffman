r"""
Script to demonstrate how to plot on custom subplots using the fig_ax_iter argument.

Notes
-----
#.  Written by David C. Stauffer in January 2022.
"""

#%% Imports
from dstauffman.plotting import make_connected_sets, figmenu
import numpy as np
import matplotlib.pyplot as plt

#%% Script
if __name__ == '__main__':
    # create data
    description = 'Focal Plane Sightings'
    points = 2 * np.random.rand(2, 1000) - 1.
    innovs = 0.1 * np.random.randn(*points.shape)
    innovs[:, points[0, :] < 0] -= 0.1
    innovs[:, points[1, :] > 0] += 0.2

    # build the figures and axes combinations to use
    (fig, axes) = plt.subplots(2, 2, sharex=True)

    # create an iterable to return (fig, ax) pairs in the order you want things plotted by each call
    fig_ax_iter = iter(((fig, axes[0, 0]), (fig, axes[0, 1]), (fig, axes[1, 0]), (fig, axes[1, 1])))

    fig1 = make_connected_sets(description, points, innovs=None,   use_datashader=False, fig_ax_iter=fig_ax_iter, hide_innovs=True, color_by='none')
    fig2 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax_iter=fig_ax_iter, hide_innovs=False, color_by='none')
    fig3 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax_iter=fig_ax_iter, hide_innovs=True, color_by='direction')
    fig4 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax_iter=fig_ax_iter, hide_innovs=True, color_by='magnitude')

    figmenu(fig)

    plt.show(block=False)

    assert fig1 is fig2 and fig1 is fig3 and fig1 is fig4 and fig1 is fig, 'All figures should be identical.'
