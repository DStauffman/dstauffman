r"""
Script to demonstrate how to plot on custom subplots using the fig_ax_iter argument.

Notes
-----
#.  Written by David C. Stauffer in January 2022.
"""

#%% Imports
from dstauffman import unit
from dstauffman.plotting import figmenu, make_connected_sets, Opts, plot_correlation_matrix, plot_histogram, setup_plots
import numpy as np
import matplotlib.pyplot as plt

#%% Script
if __name__ == '__main__':
    #%% Example 1
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

    #%% Example 2
    (fig2, axes2) = plt.subplots(1, 2, sharex=False)
    fig2.suptitle('Combined Plots')
    fig_ax_iter2 = iter(((fig2, axes2[0]), (fig2, axes2[1])))
    # histogram
    description = 'Histogram'
    data = np.array([0.5, 3.3, 1., 1.5, 1.5, 1.75, 2.5, 2.5])
    bins = np.array([0., 1., 2., 3., 5., 7.])
    plot_histogram(description, data, bins, fig_ax_iter=fig_ax_iter2, skip_setup_plots=True)
    # correlation matrix
    data = unit(np.random.rand(10, 10), axis=0)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    units = 'm'
    opts = Opts(case_name='Test')
    matrix_name = 'Correlation Matrix'
    cmin = 0
    cmax = 1
    xlabel = ''
    ylabel = ''
    plot_lower_only = True
    label_values = True
    x_lab_rot = 90
    colormap = None
    plot_border=None
    leg_scale = 'centi'
    plot_correlation_matrix(data, labels, units=units, opts=opts, matrix_name=matrix_name, \
        cmin=cmin, cmax=cmax, xlabel=xlabel, ylabel=ylabel, plot_lower_only=plot_lower_only, \
        label_values=label_values, x_lab_rot=x_lab_rot, colormap=colormap, plot_border=plot_border, \
        leg_scale=leg_scale, fig_ax_iter=fig_ax_iter2, skip_setup_plots=True)
    setup_plots(fig2, opts)
