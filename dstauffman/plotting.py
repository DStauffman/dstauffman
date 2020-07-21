r"""
Defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import doctest
import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter

from dstauffman.classes      import Frozen
from dstauffman.constants    import DEFAULT_COLORMAP
from dstauffman.plot_generic import make_time_plot
from dstauffman.plot_support import ColorMap, ignore_plot_data, setup_plots
from dstauffman.time         import convert_date, convert_time_units
from dstauffman.units        import get_factors

#%% Globals
logger = logging.getLogger(__name__)

#%% Classes - Opts
class Opts(Frozen):
    r"""Optional plotting configurations."""
    def __init__(self, *args, **kwargs):
        r"""
        Default configuration for plots.
            .case_name : str
                Name of the case to be plotted
            .date_zero : (1x6) datevec (or datetime)
                Date of t = 0 time [year month day hour minute second]
            .save_plot : bool
                Flag for whether to save the plots
            .save_path : str
                Location for the plots to be saved
            .show_plot : bool
                Flag to show the plots or only save to disk
            .show_link : bool
                Flag to show a link to the folder where the plots were saved
            .plot_type : str
                Type of plot to save to disk, from {'png','jpg','fig','emf'}
            .sub_plots : bool
                Flag specifying whether to plot as subplots or separate figures
            .sing_line : bool
                Flag specifying whether to plot only one line per axes, using subplots as necessary
            .disp_xmin : float
                Minimum time to display on plot [sec]
            .disp_xmax : float
                Maximum time to display on plot [sec]
            .rms_xmin  : float
                Minimum time from which to begin RMS calculations [sec]
            .rms_xmax  : float
                Maximum time from which to end RMS calculations [sec]
            .show_rms  : bool
                Flag for whether to show the RMS in the legend
            .use_mean  : bool
                Flag for using mean instead of RMS for legend calculations
            .show_zero : bool
                Flag for whether to show Y=0 on the plot axis
            .quat_comp : bool
                Flag to plot quaternion component differences or just the angle
            .show_xtra : bool
                Flag to show extra points in one vector or the other when plotting differences
            .time_base : str
                Base units of time, typically from {'sec', 'months'}
            .time_unit : str
                Time unit for the x axis, from {'', 'sec', 'min', 'hr', 'day', 'month', 'year'}
            .vert_fact : str
                Vertical factor to apply to the Y axis,
                from: {'yotta','zetta','exa','peta','tera','giga','mega','kilo','hecto','deca',
                'unity','deci','centi','milli', 'micro','nano','pico','femto','atto','zepto','yocto'}
            .colormap  : str
                Name of the colormap to use
            .leg_spot  : str
                Location to place the legend, from {'best', 'upper right', 'upper left',
                'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center',
                'upper center', 'center' or tuple of position}
            .classify  : str
                Classification level to put on plots
            .names     : list of str
                Names of the data structures to be plotted
        """
        self.case_name = ''
        self.date_zero = None
        self.save_plot = False
        self.save_path = os.getcwd()
        self.show_plot = True
        self.show_link = False
        self.plot_type = 'png'
        self.sub_plots = True
        self.sing_line = False
        self.disp_xmin = -np.inf
        self.disp_xmax =  np.inf
        self.rms_xmin  = -np.inf
        self.rms_xmax  =  np.inf
        self.show_rms  = True
        self.use_mean  = False
        self.show_zero = False
        self.quat_comp = True
        self.show_xtra = True
        self.time_base = 'sec'
        self.time_unit = 'sec'
        self.vert_fact = 'unity'
        self.colormap  = None
        self.leg_spot  = 'best'
        self.classify  = ''
        self.names     = list()
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, self.__class__):
                for (key, value) in vars(arg).items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        raise ValueError(f'Unexpected option of "{key}" passed to Opts initializer."')
            else:
                raise ValueError('Unexpected input argument receieved.')
        for (key, value) in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unexpected option of "{key}" passed to Opts initializer."')

    def __copy__(self):
        r"""Allows a new copy to be generated with data from the original."""
        new = type(self)(self)
        return new

    def get_names(self, ix):
        r"""Get the specified name from the list."""
        if hasattr(self, 'names') and len(self.names) >= ix+1:
            name = self.names[ix]
        else:
            name = ''
        return name

    def get_date_zero_str(self):
        r"""
        Gets a string representation of date_zero, typically used to print on an X axis.

        Returns
        -------
        start_date : str
            String representing the date of time zero.

        Examples
        --------
        >>> from dstauffman import Opts
        >>> from datetime import datetime
        >>> opts = Opts()
        >>> opts.date_zero = datetime(2019, 4, 1, 18, 0, 0)
        >>> print(opts.get_date_zero_str())
          t(0) = 01-Apr-2019 18:00:00 Z

        """
        TIMESTR_FORMAT = '%d-%b-%Y %H:%M:%S'
        start_date = '  t(0) = ' + self.date_zero.strftime(TIMESTR_FORMAT) + ' Z' if self.date_zero is not None else ''
        return start_date

    def get_time_limits(self):
        r"""Returns the display and RMS limits in the current time units."""
        def _convert(value):
            if value is not None and np.isfinite(value):
                return convert_time_units(value, self.time_base, self.time_unit)
            return value

        if self.time_base == 'datetime':
            return (self.disp_xmin, self.disp_xmax, self.rms_xmin, self.rms_xmax)

        disp_xmin = _convert(self.disp_xmin)
        disp_xmax = _convert(self.disp_xmax)
        rms_xmin  = _convert(self.rms_xmin)
        rms_xmax  = _convert(self.rms_xmax)
        return (disp_xmin, disp_xmax, rms_xmin, rms_xmax)

    def convert_dates(self, form, numpy_form='datetime64[ns]'):
        r"""Converts between double and datetime representations."""
        assert form in {'datetime', 'sec'}, f'Unexpected form of "{form}".'
        self.time_base = form
        self.time_unit = form
        self.disp_xmin = convert_date(self.disp_xmin, form=form, date_zero=self.date_zero)
        self.disp_xmax = convert_date(self.disp_xmax, form=form, date_zero=self.date_zero)
        self.rms_xmin  = convert_date(self.rms_xmin,  form=form, date_zero=self.date_zero)
        self.rms_xmax  = convert_date(self.rms_xmax,  form=form, date_zero=self.date_zero)
        return self

#%% Functions - plot_time_history
def plot_time_history(description, time, data, opts=None, *, ignore_empties=False, **kwargs):
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    description : str
        Name to label on the plots
    time : 1D ndarray
        time history
    data : 1D, 2D or 3D ndarray
        data for corresponding time history, time is first dimension, last dimension is bin
        middle dimension if 3D is the cycle
    opts : class Opts, optional
        plotting options
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    save_plot : bool, optional
        Ability to overide the option in opts
    kwargs : dict
        Remaining keyword arguments will be passed to make_time_plot

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    See Also
    --------
    make_time_plot

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Updated by David C. Stauffer in October 2017 to do comparsions of multiple runs.

    Examples
    --------
    >>> from dstauffman import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = 'Random Data'
    >>> time = np.arange(0, 5, 1./12) + 2000
    >>> data = np.random.rand(5, len(time)).cumsum(axis=1)
    >>> data = 10 * data / np.expand_dims(data[:, -1], axis=1)
    >>> fig  = plot_time_history(description, time, data)

    Date based version
    >>> time2 = np.datetime64('2020-05-01 00:00:00', 'ns') + 10**9*np.arange(0, 5*60, 5, dtype=np.int64)
    >>> fig2 = plot_time_history(description, time2, data, time_units='datetime')

    Close plots
    >>> plt.close(fig)
    >>> plt.close(fig2)

    """
    # force inputs to be ndarrays
    time = np.atleast_1d(np.asanyarray(time))
    data = np.asanyarray(data)

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        logger.info(f' {description} plot skipped due to missing data.')
        return None
    assert time.ndim == 1, 'Time must be a 1D array.'

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    colormap     = kwargs.pop('colormap', this_opts.colormap)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # call wrapper function for most of the details
    fig = make_time_plot(description, time, data, \
        time_units=time_units, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
        disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, colormap=colormap, \
        use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, **kwargs)

    # setup plots
    setup_plots(fig, this_opts)
    return fig

#%% Functions - plot_correlation_matrix
def plot_correlation_matrix(data, labels=None, units='', opts=None, *, matrix_name='Correlation Matrix', \
        cmin=0, cmax=1, xlabel='', ylabel='', plot_lower_only=True, label_values=False, x_lab_rot=90, \
        colormap=None, plot_border=None):
    r"""
    Visually plot a correlation matrix.

    Parameters
    ----------
    data : array_like
        data for corresponding time history
    labels : list of str, optional
        Names to put on row and column headers
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    matrix_name : str, optional
        Name to put on figure and plot title
    cmin : float, optional
        Minimum value for color range, default is zero
    cmax : float, optional
        Maximum value for color range, default is one
    xlabel : str, optional
        X label to put on plot
    ylabel : str, optional
        Y label to put on plot
    plot_lower_only : bool, optional
        Plots only the lower half of a symmetric matrix, default is True
    label_values : bool, optional
        Annotate the numerical values of each square in addition to the color code, default is False
    x_lab_rot : float, optional
        Amount in degrees to rotate the X labels, default is 90
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap
    plot_border : str, optional
        Color of the border to plot

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in July 2015.

    Examples
    --------
    >>> from dstauffman import plot_correlation_matrix, unit
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.rand(10, 10)
    >>> labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    >>> data = unit(data, axis=0)
    >>> fig = plot_correlation_matrix(data, labels)

    Close plot
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = 'cool'
        else:
            colormap = opts.colormap
    (scale, prefix) = get_factors(opts.vert_fact)

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
    if m == n and np.all((np.abs(data - np.transpose(data)) < precision) | np.isnan(data)):
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

    # determine which type of data to plot
    this_title = matrix_name + (' [' + units + ']' if units else '')

    # Create plots
    # create figure
    fig = plt.figure()
    # set figure title
    fig.canvas.set_window_title(matrix_name)
    # get handle to axes for use later
    ax = fig.add_subplot(111)
    # set axis color to none
    ax.patch.set_facecolor('none')
    # set title
    ax.set_title(this_title)
    # get colormap based on high and low limits
    cm = ColorMap(colormap, low=scale*cmin, high=scale*cmax)
    # loop through and plot each element with a corresponding color
    for i in range(m):
        for j in range(n):
            if not plot_lower_only or (i <= j):
                if not np.isnan(data[j, i]):
                    ax.add_patch(Rectangle((box_size*i, box_size*j), box_size, box_size, \
                        facecolor=cm.get_color(scale*data[j, i]), edgecolor=plot_border))
                if label_values:
                    ax.annotate('{:.2g}'.format(scale*data[j, i]), xy=(box_size*i + box_size/2, box_size*j + box_size/2), \
                        xycoords='data', horizontalalignment='center', \
                        verticalalignment='center', fontsize=15)
    # show colorbar
    fig.colorbar(cm.get_smap())
    # make square
    ax.set_aspect('equal')
    # set limits and tick labels
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(0, m)+box_size/2)
    ax.set_xticklabels(xlab, rotation=x_lab_rot)
    ax.set_yticks(np.arange(0, n)+box_size/2)
    ax.set_yticklabels(ylab)
    # label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # reverse the y axis
    ax.invert_yaxis()

    # Setup plots
    setup_plots(fig, opts)
    return fig

#%% Functions - plot_bar_breakdown
def plot_bar_breakdown(time, data, label, opts=None, *, legend=None, ignore_empties=False, colormap=None):
    r"""
    Plot the pie chart like breakdown by percentage in each category over time.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history, 2D: time by ratio in each category
    label : str
        Name to label on the plots
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in June 2015.

    Examples
    --------
    >>> from dstauffman import plot_bar_breakdown
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5)
    >>> mag   = np.sum(data, axis=1)
    >>> data  = data / np.expand_dims(mag, axis=1)
    >>> label = 'Test'
    >>> fig   = plot_bar_breakdown(time, data, label)

    Close plot
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    legend_loc = opts.leg_spot
    time_units = opts.time_base

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        logger.info(f' {label} plot skipped due to missing data.')
        return

    # hard-coded values
    this_title = label + ' vs. Time'
    scale      = 100
    units      = '%'
    unit_text  = ' [' + units + ']'

    # data checks
    num_bins   = data.shape[1]
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Series {}'.format(i+1) for i in range(num_bins)]

    # get colormap based on high and low limits
    cm = ColorMap(colormap, 0, num_bins-1)

    # figure out where the bottoms should be to stack the data
    bottoms = np.concatenate((np.zeros((len(time), 1)), np.cumsum(data, axis=1)), axis=1)

    # plot breakdown
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    ax = fig.add_subplot(111)
    for i in range(num_bins):
        if not ignore_plot_data(data, ignore_empties, col=i):
            # Note: The performance of ax.bar is really slow with large numbers of bars (>20), so
            # fill_between is a better alternative
            ax.fill_between(time, scale*bottoms[:, i], scale*bottoms[:, i+1], step='mid', \
                label=legend[i], color=cm.get_color(i), edgecolor='none')
    ax.set_xlabel('Time [' + time_units + ']')
    ax.set_ylabel(label + unit_text)
    ax.set_ylim(0, 100)
    ax.grid(True)
    # set years to always be whole numbers on the ticks
    if (time[-1] - time[0]) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.legend(loc=legend_loc)
    ax.set_title(this_title)

    # Setup plots
    setup_plots(fig, opts)
    return fig

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
