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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle
from PyQt4 import QtGui, QtCore
# model imports
from dstauffman.classes   import Frozen
from dstauffman.constants import DEFAULT_COLORMAP
from dstauffman.utils     import get_images_dir, rms

#%% Private Classes - _HoverButton
class _HoverButton(QtGui.QPushButton):
    r"""Custom button that allows hovering and icons."""
    def __init__(self, *args, **kwargs):
        # initialize
        super(_HoverButton, self).__init__(*args, **kwargs)
        # Enable mouse hover event tracking
        self.setMouseTracking(True)
        self.setStyleSheet('border: 0px;')
        # set icon
        for this_arg in args:
            if isinstance(this_arg, QtGui.QIcon):
                self.setIcon(this_arg)
                self.setIconSize(QtCore.QSize(24, 24))

    def enterEvent(self, event):
        # Draw border on hover
        self.setStyleSheet('border: 1px; border-style: solid;') # pragma: no cover

    def leaveEvent(self, event):
        # Delete border after hover
        self.setStyleSheet('border: 0px;') # pragma: no cover

#%% Classes - Opts
class Opts(Frozen):
    r"""
    Contains all the optional plotting configurations.
    """
    def __init__(self):
        self.case_name = 'baseline'
        self.save_path = os.getcwd()
        self.save_plot = False
        self.plot_type = 'png'
        self.sub_plots = True
        self.show_plot = True
        self.show_link = False
        self.disp_xmin = -np.inf
        self.disp_xmax =  np.inf
        self.rms_xmin  = -np.inf
        self.rms_xmax  =  np.inf
        self.names     = list()

    def get_names(self, ix):
        r"""Gets the specified name from the list."""
        if hasattr(self, 'names') and len(self.names) >= ix+1:
            name = self.names[ix]
        else:
            name = ''
        return name

#%% Classes - MyCustomToolbar
class MyCustomToolbar():
    r"""
    Defines a custom toolbar to use in any matplotlib plots.

    Examples
    --------

    >>> from dstauffman import MyCustomToolbar
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> fig.toolbar_custom_ = MyCustomToolbar(fig)

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, fig):
        r"""Initializes the custom toolbar."""
        # Store the figure number for use later (Note this works better than relying on plt.gcf()
        # to determine which figure actually triggered the button events.)
        self.fig_number = fig.number
        # create buttons - Prev Plot
        icon = QtGui.QIcon(os.path.join(get_images_dir(), 'prev_plot.png'))
        self.btn_prev_plot = _HoverButton(icon, '')
        self.btn_prev_plot.setToolTip('Show the previous plot')
        fig.canvas.toolbar.addWidget(self.btn_prev_plot)
        self.btn_prev_plot.clicked.connect(self.prev_plot)
        # create buttons - Next Plot
        icon = QtGui.QIcon(os.path.join(get_images_dir(), 'next_plot.png'))
        self.btn_next_plot = _HoverButton(icon, '')
        self.btn_next_plot.setToolTip('Show the next plot')
        fig.canvas.toolbar.addWidget(self.btn_next_plot)
        self.btn_next_plot.clicked.connect(self.next_plot)
        # create buttons - Close all
        icon = QtGui.QIcon(os.path.join(get_images_dir(), 'close_all.png'))
        self.btn_close_all = _HoverButton(icon, '')
        self.btn_close_all.setToolTip('Close all the open plots')
        fig.canvas.toolbar.addWidget(self.btn_close_all)
        self.btn_close_all.clicked.connect(self.close_all)

    def close_all(self, *args):
        r"""Closes all the currently open plots."""
        # Note that it's better to loop through and close the plots individually than to use
        # plt.close('all'), as that can sometimes cause the iPython kernel to quit #DCS: 2015-06-11
        for this_fig in plt.get_fignums():
            plt.close(this_fig)

    def next_plot(self, *args):
        r"""Brings up the next plot in the series."""
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
        r"""Brings up the previous plot in the series."""
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

#%% Classes - ColorMap
class ColorMap(Frozen):
    r"""
    Colormap class for easier setting of colormaps in matplotlib.

    Parameters
    ----------
    colormap : str, optional
        Name of the colormap to use
    low : int, optional
        Low value to use as an index to a specific color within the map
    high : int, optional
        High value to use as an index to a specific color within the map
    num_colors : int, optional
        If not None, then this replaces the low and high inputs

    Notes
    -----
    #.  Written by David C. Stauffer in July 2015.

    Examples
    --------

    >>> from dstauffman import ColorMap
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> cm = ColorMap('Paired', 1, 2)
    >>> time = np.arange(0, 10, 0.1)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(time, np.sin(time), color=cm.get_color(1)) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    >>> ax.plot(time, np.cos(time), color=cm.get_color(2)) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    >>> plt.legend(['Sin', 'Cos']) # doctest: +ELLIPSIS
    <matplotlib.legend.Legend object at 0x...>

    >>> plt.show()

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, colormap=DEFAULT_COLORMAP, low=0, high=1, num_colors=None):
        self.num_colors = num_colors
        # check for optional inputs
        if self.num_colors is not None:
            low = 0
            high = num_colors-1
        # get colormap based on high and low limits
        cmap  = plt.get_cmap(colormap)
        cnorm = colors.Normalize(vmin=low, vmax=high)
        self.smap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        # must initialize the empty scalar mapplable to show the colorbar correctly
        self.smap.set_array([])

    def get_color(self, value):
        r"""Get the color based on the scalar value."""
        return self.smap.to_rgba(value)

    def get_smap(self):
        r"""Returns the smap being used."""
        return self.smap

    def set_colors(self, ax):
        r"""Set the colors for the given axis based on internal instance information."""
        if self.num_colors is None:
            raise ValueError("You can't call ColorMap.set_colors unless it was given a num_colors input.")
        ax.set_color_cycle([self.get_color(i) for i in range(self.num_colors)])

#%% Functions - plot_time_history
def plot_time_history(time, data, description, type_, opts=None, plot_indiv=True, \
    truth_time=None, truth_data=None, plot_as_diffs=False, colormap=None):
    r"""
    Plots the given data channel versus time, with a generic description argument.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history
    description : str
        generic text to put on the plot title and figure name
    type_ : str {'population', 'percentage', 'cost'}
        description of the type of data that is being plotted
    opts : class Opts, optional
        plotting options
    plot_indiv : bool, optional
        Plot the individual cycles, default is true
    truth_time : array_like, optional
        Truth time to plot
    truth_data : array_like, optional
        Truth data to plot
    plot_as_diffs : bool, optional, default is False
        Plot each entry in results against the other ones

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in Mar 2015.

    Examples
    --------

    >>> from dstauffman import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.arange(0, 10, 0.1)
    >>> data = np.sin(time)
    >>> description = 'Sin'
    >>> type_ = 'population'
    >>> plot_time_history(time, data, description, type_) # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at 0x...>

    Close plot
    >>> plt.close()

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        colormap = DEFAULT_COLORMAP
    # determine which type of data to plot
    if type_ == 'population':
        scale = 1
        units = '#'
    elif type_ == 'percentage':
        scale = 100
        units = '%'
    elif type_ == 'per 100K':
        scale = 100000
        units = 'per 100,000'
    elif type_ == 'cost':
        scale = 1e-3
        units = "$K's"
    else:
        raise ValueError('Unknown data type "' + type_ + '" for plot.')

    if plot_as_diffs:
        # calculate RMS
        rms_data = rms(scale*data, axis=1)
        # build colormap
        cm = ColorMap(colormap, num_colors=data.shape[0])
    else:
        # calculate the mean and std of data
        if data.ndim == 1:
            mean = data
            std  = np.zeros(len(data))
        else:
            mean = np.mean(data, axis=0)
            std  = np.std(data, axis=0)

        # calculate an RMS
        rms_data = rms(scale*mean, axis=0)

    # turn interaction off to make the plots draw all at once on a show() command
    plt.ioff()
    # alias the title
    this_title = description + ' vs. Time'
    # create the figure and set the title
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    # add an axis and plot the data
    ax = fig.add_subplot(111)
    if plot_as_diffs:
        cm.set_colors(ax)
        for ix in range(data.shape[0]):
            this_label = opts.get_names(ix)
            if not this_label:
                this_label = 'Series {}'.format(ix+1)
            this_label = this_label + ' (RMS: {:.2f})'.format(rms_data[ix])
            ax.plot(time, scale*data[ix, :], '.-', linewidth=2, zorder=10, label=this_label)
    else:
        ax.plot(time, scale*mean, 'b.-', linewidth=2, zorder=10, \
            label=opts.get_names(0) + description + ' (RMS: {:.2f})'.format(rms_data))
        ax.errorbar(time, scale*mean, scale*std, linestyle='None', marker='None', ecolor='c', zorder=6)
        # inidividual line plots
        if plot_indiv and data.ndim > 1:
            for ix in range(data.shape[0]):
                ax.plot(time, scale*data[ix, :], color='0.75', zorder=1)
    # optionally plot truth (without changing the axis limits)
    if truth_data is not None:
        limits = plt.axis()
        ax.plot(truth_time, scale*truth_data, 'k.-', linewidth=2, zorder=8, label='Truth')
        plt.axis(limits)
    # add labels and legends
    plt.xlabel('Time [year]')
    plt.ylabel(description + ' [' + units + ']')
    plt.title(this_title)
    plt.legend()
    # show a grid
    plt.grid(True)
    # set the colormap (only applies to lines that don't specify a default color)
    plt.set_cmap(colormap)
    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_correlation_matrix
def plot_correlation_matrix(data, labels=None, opts=Opts(), matrix_name='Correlation Matrix', \
        cmin=0, cmax=1, colormap='cool', plot_lower_only=True, label_values=False):
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
    precision       = 1e-12

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
    if np.all(data >= -1 + precision) and np.any(data <= -precision) and cmin == 0 and cmax == 1:
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
    cm = ColorMap(colormap, low=cmin, high=cmax)
    # loop through and plot each element with a corresponding color
    for i in range(m):
        for j in range(n):
            if not plot_lower_only or (i <= j):
                ax.add_patch(Rectangle((box_size*i,box_size*j),box_size, box_size, \
                    color=cm.get_color(data[j, i])))
                if label_values:
                    #ax.text(box_size*i+box_size/2, box_size*j+box_size/2, '99', \
                    #    verticalalignment='center', horizontalalignment='center', \
                    #    transform=ax.transAxes, color='black', fontsize=15)
                    ax.annotate('{:.2g}'.format(data[j,i]), xy=(box_size*i + box_size/2, box_size*j + box_size/2), \
                        xycoords='data', horizontalalignment='center', \
                        verticalalignment='center', fontsize=15)
    # show colorbar
    plt.colorbar(cm.get_smap())
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
        folder = os.getcwd() # pragma: no cover
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
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
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

#%% Functions - disp_xlimits
def disp_xlimits(figs, xmin=None, xmax=None):
    r"""
    Sets the xlimits to the specified xmin and xmax.

    Parameters
    ----------
    figs : array_like
        List of figures
    xmin : scalar
        Minimum X value
    xmax : scalar
        Maximum X value

    Notes
    -----
    #.  Written by David C. Stauffer in August 2015.

    Examples
    --------

    >>> from dstauffman import disp_xlimits
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False)
    >>> xmin = 2
    >>> xmax = 5
    >>> disp_xlimits(fig, xmin, xmax)

    Close plot
    >>> plt.close()

    """
    # check for single figure
    if not isinstance(figs, list):
        figs = [figs]
    # loop through figures
    for this_fig in figs:
        # get axis list and loop through them
        for this_axis in this_fig.axes:
            # get xlimits for this axis
            (old_xmin, old_xmax) = this_axis.get_xlim()
            # set the new limits
            if xmin is not None:
                new_xmin = np.max([xmin, old_xmin])
            else:
                new_xmin = old_xmin
            if xmax is not None:
                new_xmax = np.min([xmax, old_xmax])
            else:
                new_xmax = old_xmax
            # modify xlimits
            this_axis.set_xlim((new_xmin, new_xmax))
    # force updating of all the figures
    plt.draw()

#%% Functions - setup_plots
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
    #.  Written by David C. Stauffer in May 2015.

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
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') # doctest: +SKIP
    >>> plt.ylabel('value [radians]') # doctest: +SKIP
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

    # change the display range
    if plot_type in {'time', 'time_no_yscale'}:
        disp_xlimits(figs, opts.disp_xmin, opts.disp_xmax)

    # add a custom toolbar
    figmenu(figs)

    # show the plot
    if opts.show_plot:
        plt.show(block=False)

    # optionally save the plot
    if opts.save_plot:
        storefig(figs, opts.save_path, opts.plot_type)
        if opts.show_link & len(figs) > 0:
            print(r'Plots saved to <a href="{}">{}</a>'.format(opts.save_path, opts.save_path))

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
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') # doctest: +SKIP
    >>> plt.ylabel('value [radians]') # doctest: +SKIP
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
