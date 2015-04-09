# -*- coding: utf-8 -*-
r"""
Plotting module file for the "dstauffman" library.  It defines useful plotting utilities.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

# pylint: disable=E1101

#%% Imports
from __future__ import print_function
from __future__ import division
import doctest
import numpy as np
import os
import unittest
from dstauffman.classes import Frozen

import matplotlib
# set plotting backend (note, must happen before importing pyplot)
matplotlib.use('QT4Agg') # TODO: works well for me in Python 3.4 on Windows, make configurable?
import matplotlib.pyplot as plt

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
        self.name_one  = ''
        self.name_two  = ''
        self.plot_true = 'none' # from {'none', 'SA', 'Rwanda', 'Rwanda_DHS', 'Rwanda_poly_fit'}

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
    if not isinstance(fig, list):
        figs = [fig]
    else:
        figs = fig
    # make sure types is a list
    if not isinstance(plot_type, list):
        types = []
        types.append(plot_type)
    else:
        types = plot_type
    # if no folder was specified, then use the current working directory
    if folder is None:
        folder = os.getcwd()
    # confirm that the folder exists
    if not os.path.isdir(folder):
        raise ValueError('The specfied folder "{}" does not exist.'.format(folder))
    # loop through the figures
    for this_fig in figs:
        # get the title of the figure canvas
        this_title = fig.canvas.get_window_title()
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
    #. Written by David C. Stauffer in March 2015.

    See Also
    --------
    storefig

    Examples
    --------
    Create figure and then change the title

    >>> from ghap import titleprefix
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
    if not isinstance(fig, list):
        figs = [fig]
    # loop through figures
    for this_fig in figs:
        # get axis list and loop through them
        for this_axis in this_fig.axes:
            # get title for this axis
            this_title = this_axis.get_title()
            # mofidy and set new title
            new_title = prefix + ' - ' + this_title
            this_axis.set_title(new_title)
        # update canvas name
        this_canvas_title = this_fig.canvas.get_window_title()
        this_fig.canvas.set_window_title(prefix + ' - ' + this_canvas_title)
    # force updating of all the figures
    plt.draw()

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
