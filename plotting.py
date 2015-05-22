# -*- coding: utf-8 -*-
r"""
Plotting module file for the "dstauffman" library.  It defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

# pylint: disable=E1101

#%% Imports
# normal imports
from __future__ import print_function
from __future__ import division
import doctest
import numpy as np
import os
import unittest
# plotting imports
import matplotlib
# set plotting backend (note, must happen before importing pyplot)
matplotlib.use('QT4Agg') # TODO: works well for me in Python 3.4 on Windows, make configurable?
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle
from PyQt4 import QtGui
# model imports
from dstauffman.classes import Frozen

#%% Classes - Opts
class Opts(Frozen):
    r"""
    Opts is the class that holds all the optional plotting configurations.
    """
    def __init__(self):
        # TODO: expand functionality
        self.case_name = 'baseline'
        self.save_path = os.getcwd()
        self.save_plot = False
        self.plot_type = 'png'
        self.sub_plots = True
        self.show_plot = True
        self.disp_xmin = -np.inf
        self.disp_xmax =  np.inf
        self.rms_xmin  = -np.inf
        self.rms_xmax  =  np.inf
        self.names     = list()

    def get_names(self, ix):
        r"""Gets the specified name from the list."""
        if ix > len(self.names):
            name = ''
        else:
            name = self.names[ix]
        return name

#%% Classes - MyCustomToolbar
class MyCustomToolbar():
    def __init__(self, fig):
        # Store the figure number for use later (Note this works better than relying on plt.gcf()
        # to determine which figure actually triggered the button events.)
        self.fig_number = fig.number
        # create buttons - Prev Plot
        self.btn_prev_plot = QtGui.QPushButton(' << ')
        self.btn_prev_plot.setToolTip('Show the previous plot')
        fig.canvas.toolbar.addWidget(self.btn_prev_plot)
        self.btn_prev_plot.clicked.connect(self.prev_plot)
        # create buttons - Next Plot
        self.btn_next_plot = QtGui.QPushButton(' >> ')
        self.btn_next_plot.setToolTip('Show the next plot')
        fig.canvas.toolbar.addWidget(self.btn_next_plot)
        self.btn_next_plot.clicked.connect(self.next_plot)
        # create buttons - Close all
        self.btn_close_all = QtGui.QPushButton('Close All')
        self.btn_close_all.setToolTip('Close all the open plots')
        fig.canvas.toolbar.addWidget(self.btn_close_all)
        self.btn_close_all.clicked.connect(self.close_all)

    def close_all(self, *args):
        plt.close('all')

    def next_plot(self, *args):
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i < len(all_figs)-1:
                    next_fig = all_figs[i+1]
                else:
                    next_fig = all_figs[0]
        # set the appropriate active figure
        fig = plt.figure(next_fig)
        # make it the active window
        fig.canvas.manager.window.raise_()

    def prev_plot(self, *args):
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i > 0:
                    prev_fig = all_figs[i-1]
                else:
                    prev_fig = all_figs[-1]
        # set the appropriate active figure
        fig = plt.figure(prev_fig)
        # make it the active window
        fig.canvas.manager.window.raise_()

#%% Functions - plot_correlation_matrix
def plot_correlation_matrix(data, labels=None, opts=Opts(), matrix_name='Correlation Matrix'):
    r"""
    Visually plots a correlation matrix.

    Parameters
    ----------


    Examples
    --------

    >>> from dstauffman import plot_correlation_matrix, unit
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.rand(10, 10)
    >>> labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    >>> data = unit(data, axis=0)
    >>> fig = plot_correlation_matrix(data, labels)

    Close plots
    >>> plt.close(fig)

    """
    # Hard-coded values
    box_size        = 1
    cmin            = 0
    cmax            = 1
    precision       = 1e-12
    # plot_lower_only is a stub for future development of optional outputs
    plot_lower_only = True
    # the color map could also potentially be overriden in the future
    color_map       = 'cool'

    # get sizes
    (n, m) = data.shape

    # check labels
    if labels is None:
        xlab = [str(i) for i in range(m)]
        ylab = [str(i) for i in range(n)]
    else:
        if isinstance(labels[0], list):
            xlab = labels[0]
            ylab = labels[1]
        else:
            xlab = labels
            ylab = labels
    # check lengths
    if len(xlab) != m or len(ylab) != n:
        raise ValueError('Incorrectly sized labels.')

    # Determine if symmetric
    if m == n and np.all(np.abs(data - np.transpose(data)) < precision):
        is_symmetric = True
    else:
        is_symmetric = False
    plot_lower_only  = plot_lower_only and is_symmetric

    # Override color ranges based on data
    # test if in -1 to 1 range instead of 0 to 1
    if np.all(data >= -1 + precision) and np.any(data <= -precision):
        cmin = -1
    # test if outside the cmin to cmax range, and if so, adjust range.
    temp = np.min(data)
    if temp < cmin:
        cmin = temp
    temp = np.max(data)
    if temp > cmax:
        cmax = temp

    # Create plots
    # turn interaction off
    plt.ioff()
    # create figure
    fig = plt.figure()
    # set figure title
    fig.canvas.set_window_title(matrix_name)
    # get handle to axes for use later
    ax = fig.add_subplot(111)
    # set axis color to none
    ax.patch.set_facecolor('none')
    # set title
    plt.title(fig.canvas.get_window_title())
    # get colormap based on high and low limits
    cmap  = plt.get_cmap(color_map)
    cnorm = colors.Normalize(vmin=cmin, vmax=cmax)
    smap  = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    # must initialize the empty scalar mapplable to show the colorbar
    smap.set_array([])
    # loop through and plot each element with a corresponding color
    for i in range(m):
        for j in range(n):
            if not plot_lower_only or (i <= j):
                ax.add_patch(Rectangle((box_size*i,box_size*j),box_size, box_size, \
                    color=smap.to_rgba(np.abs(data[j, i]))))
    # show colorbar
    plt.colorbar(smap)
    # make square
    plt.axis('equal')
    # set limits and tick labels
    plt.xlim(0, m)
    plt.ylim(0, n)
    plt.xticks(np.arange(0, m)+box_size/2, xlab)
    plt.yticks(np.arange(0, n)+box_size/2, ylab)
    # reverse the y axis
    ax.invert_yaxis()

    # Setup plots
    setup_plots(fig, opts, 'dist')

    return fig

#%% Functions - storefig
def storefig(fig, folder=None, plot_type='png'):
    r"""
    Stores the specified figures in the specified folder and with the specified plot type(s)

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    folder : str
        Location to save figures to
    plot_type : str
        Type of figure to save to disk, like 'png' or 'jpg'

    Raises
    ------
    ValueError
        Specified folder to save figures to doesn't exist.

    Notes
    -----
    #. Uses the figure.canvas.get_window_title property to determine the figure name.

    See Also
    --------
    matplotlib.pyplot.savefig, titleprefix

    Examples
    --------
    Create figure and then save to disk

    >>> from dstauffman import storefig
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import os
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False)
    >>> folder = os.getcwd()
    >>> plot_type = 'png'
    >>> storefig(fig, folder, plot_type)

    Close plot
    >>> plt.close()

    Delete file
    >>> os.remove(os.path.join(folder, 'Figure Title.png'))

    """
    # make sure figs is a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # make sure types is a list
    if not isinstance(plot_type, list):
        types = []
        types.append(plot_type)
    else:
        types = plot_type
    # if no folder was specified, then use the current working directory
    if folder is None:
        folder = os.getcwd() #pragma: no cover
    # confirm that the folder exists
    if not os.path.isdir(folder):
        raise ValueError('The specfied folder "{}" does not exist.'.format(folder))
    # loop through the figures
    for this_fig in figs:
        # get the title of the figure canvas
        this_title = this_fig.canvas.get_window_title()
        # loop through the plot types
        for this_type in types:
            # save the figure to the specified plot type
            this_fig.savefig(os.path.join(folder, this_title + '.' + this_type))

#%% Functions - titleprefix
def titleprefix(fig, prefix=''):
    r"""
    Prepends a text string to all the titles on existing figures.

    It also sets the canvas title used by storefig when saving to a file.

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    prefix : str
        Text to be prepended to the title and figure name

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Desired this function to also check for suptitles.

    See Also
    --------
    storefig

    Examples
    --------
    Create figure and then change the title

    >>> from dstauffman import titleprefix
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') #doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False)
    >>> prefix = 'Baseline'
    >>> titleprefix(fig, prefix)

    Close plot
    >>> plt.close()

    """
    # check for non-empty prefix
    if not prefix:
        return
    # force figs to be a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # loop through figures
    for this_fig in figs:
        # get axis list and loop through them
        for this_axis in this_fig.axes:
            # get title for this axis
            this_title = this_axis.get_title()
            # if the title is empty, then don't do anything
            if not this_title:
                continue
            # modify and set new title
            new_title = prefix + ' - ' + this_title
            this_axis.set_title(new_title)
        # update canvas name
        this_canvas_title = this_fig.canvas.get_window_title()
        this_fig.canvas.set_window_title(prefix + ' - ' + this_canvas_title)
    # force updating of all the figures
    plt.draw()

def setup_plots(figs, opts, plot_type='time'):
    r"""
    Combines common plot operations into one easy command.

    Parameters
    ----------
    figs : array_like
        List of figures
    opts : class Opts
        Optional plotting controls
    plot_type : optional, {'time', 'time_no_yscale', 'dist', 'dist_no_yscale'}

    Notes
    -----
    #.  Written by David Stauffer in May 2015.

    Examples
    --------

    >>> from dstauffman import setup_plots, Opts
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') #doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') #doctest: +SKIP
    >>> plt.ylabel('value [radians]') #doctest: +SKIP
    >>> plt.show(block=False)
    >>> opts = Opts()
    >>> opts.case_name = 'Testing'
    >>> opts.show_plot = True
    >>> opts.save_plot = False
    >>> setup_plots(fig, opts)

    Close plots
    >>> plt.close(fig)

    """
    # check for single figure
    if not isinstance(figs, list):
        figs = [figs]

    # prepend a title
    if opts.case_name:
        titleprefix(figs, opts.case_name)

    # add a custom toolbar
    figmenu(figs)

    # show the plot
    if opts.show_plot:
        plt.show(block=False)

    # optionally save the plot
    if opts.save_plot:
        storefig(figs, opts.save_path, opts.plot_type)

#%% Functions - figmenu
def figmenu(figs):
    r"""
    Adds a custom toolbar to the figures.

    Parameters
    ----------
    figs : class matplotlib.pyplot.Figure, or list of such
        List of figures

    Examples
    --------

    >>> from dstauffman import figmenu
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') #doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') #doctest: +SKIP
    >>> plt.ylabel('value [radians]') #doctest: +SKIP
    >>> plt.show(block=False)
    >>> figmenu(fig)

    Close plot
    >>> plt.close(fig)

    """
    if not isinstance(figs, list):
        figs.toolbar_custom_ = MyCustomToolbar(figs)
    else:
        for i in range(len(figs)):
            figs[i].toolbar_custom_ = MyCustomToolbar(figs[i])

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
