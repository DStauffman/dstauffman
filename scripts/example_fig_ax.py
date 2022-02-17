r"""
Script to demonstrate how to plot on custom subplots using the fig_ax argument.

Notes
-----
#.  Written by David C. Stauffer in January 2022.
"""

#%% Imports
from dstauffman import unit
from dstauffman.plotting import fig_ax_factory, figmenu, make_connected_sets, Opts, plot_correlation_matrix, plot_histogram, plot_time_history, setup_plots
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
    fig_ax = fig_ax_factory(num_axes=[2, 2], layout='colwise', sharex=True)
    fig = fig_ax[0][0]

    # populate the plots
    fig1 = make_connected_sets(description, points, innovs=None,   use_datashader=False, fig_ax=fig_ax[0], hide_innovs=True, color_by='none')
    fig2 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax=fig_ax[1], hide_innovs=False, color_by='none')
    fig3 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax=fig_ax[2], hide_innovs=True, color_by='direction')
    fig4 = make_connected_sets(description, points, innovs=innovs, use_datashader=False, fig_ax=fig_ax[3], hide_innovs=True, color_by='magnitude')

    figmenu(fig)

    plt.show(block=False)

    assert fig1 is fig2 and fig1 is fig3 and fig1 is fig4 and fig1 is fig, 'All figures should be identical.'

    #%% Example 2
    fig_ax2 = fig_ax_factory(num_axes=2, layout='cols', sharex=False, return_figs=False, suptitle='Combined Plots')
    # histogram
    description = 'Histogram'
    data = np.array([0.5, 3.3, 1., 1.5, 1.5, 1.75, 2.5, 2.5])
    bins = np.array([0., 1., 2., 3., 5., 7.])
    plot_histogram(description, data, bins, fig_ax=fig_ax2[0], skip_setup_plots=True)
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
    fig_ex2 = plot_correlation_matrix(data, labels, units=units, opts=opts, matrix_name=matrix_name, \
        cmin=cmin, cmax=cmax, xlabel=xlabel, ylabel=ylabel, plot_lower_only=plot_lower_only, \
        label_values=label_values, x_lab_rot=x_lab_rot, colormap=colormap, plot_border=plot_border, \
        leg_scale=leg_scale, fig_ax=fig_ax2[1], skip_setup_plots=True)
    setup_plots(fig_ex2, opts)

    #%% Example 3
    fig_ax3 = fig_ax_factory(num_figs=None, num_axes=[2, 2], layout='rowwise', sharex=True, suptitle='Vector Plots')
    fig_ax3[0][0].canvas.manager.set_window_title('Vector Plots')
    time = np.arange(30)
    plot_time_history('1st', time, np.ones(30), units='one', fig_ax=([fig_ax3[0][0]], [fig_ax3[0][1]]), skip_setup_plots=True)
    plot_time_history('2nd', time, np.array([[10], [11]]) + np.ones((2, 30)), units='two', fig_ax=([fig_ax3[1][0]], [fig_ax3[1][1]]), skip_setup_plots=True)
    plot_time_history('3rd', time, np.array([[100], [110], [120]]) + np.ones((3, 30)), units='three', fig_ax=([fig_ax3[2][0]], [fig_ax3[2][1]]), skip_setup_plots=True)
    fig_ex3 = plot_time_history('4th', time, np.array([[1000], [1100], [1200], [1300]]) + np.ones((4, 30)), units='four', fig_ax=([fig_ax3[3][0]], [fig_ax3[3][1]]), skip_setup_plots=False)
