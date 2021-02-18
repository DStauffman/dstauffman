r"""
Defines low-level plotting routines meant to be wrapped in higher level ones.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

#%% Imports
import doctest
import logging
import unittest

from dstauffman import get_legend_conversion, HAVE_MPL, HAVE_NUMPY, intersect, is_datetime, LogLevel, \
    rms

from dstauffman.plotting.support import ColorMap, DEFAULT_COLORMAP, disp_xlimits, get_rms_indices, \
    plot_second_units_wrapper, plot_vert_lines, show_zero_ylim, zoom_ylim

if HAVE_MPL:
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np
    inf = np.inf
else:
    from math import inf

#%% Constants
# hard-coded values
_LEG_FORMAT  = '{:1.3f}'
_TRUTH_COLOR = 'k'

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - make_time_plot
def make_time_plot(description, time, data, *, name='', elements=None, units='', time_units='sec', \
        leg_scale='unity', start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, \
        disp_xmax=inf, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, plot_zero=False, \
        show_rms=True, legend_loc='best', second_yscale=None, ylabel=None, data_as_rows=True, \
        extra_plotter=None, use_zoh=False, label_vert_lines=True):
    r"""
    Generic data versus time plotting routine.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title, must be given
    time : (A, ) array_like
        time history [sec] or datetime64
    data : (A, ) or (N, A) ndarray, or (A, N) ndarray if data_as_rows is False
        vector history
    name : str, optional
        name of data
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec', use 'datetime' for datetime histories
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is 'unity'
    start_date : str, optional
        date of t(0), may be an empty string
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    single_lines : bool, optional
        flag meaning to plot subplots by channel instead of together
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best', use 'none' to suppress legend
    second_yscale : dict, optional
        single key and value pair to use for scaling data to a second Y axis
    ylabel : str, optional
        Labels to put on the Y axes, potentially by element
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    extra_plotter : callable, optional
        Extra callable plotting function to add more details to the plot
    use_zoh : bool, optional, default is False
        Whether to plot as a zero-order hold, instead of linear interpolation between data points
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman.plotting import make_time_plot
    >>> import numpy as np
    >>> description      = 'Values vs Time'
    >>> time             = np.arange(-10., 10.1, 0.1)
    >>> data             = time + np.cos(time)
    >>> name             = ''
    >>> elements         = None
    >>> units            = ''
    >>> time_units       = 'sec'
    >>> leg_scale        = 'unity'
    >>> start_date       = ''
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = 'Paired'
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> second_yscale    = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig = make_time_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, leg_scale=leg_scale, start_date=start_date, rms_xmin=rms_xmin, \
    ...     rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     legend_loc=legend_loc, second_yscale=second_yscale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
    ...     label_vert_lines=label_vert_lines)

    """
    # some basic flags and checks
    time_is_list = isinstance(time, list) or isinstance(time, tuple)
    data_is_list = isinstance(data, list) or isinstance(data, tuple)
    if not time_is_list:
        time = np.atleast_1d(time)
    if not data_is_list:
        data = np.atleast_2d(data)
        assert data.ndim < 3, 'Data must be 0d, 1d or 2d.'

    # calculate sizes
    temp1 = len(time) if time_is_list else 1
    if data is None:
        temp2 = 0
    elif data_is_list:
        temp2 = len(data)
    elif data_as_rows:
        temp2 = data.shape[0]
    else:
        temp2 = data.shape[1]
    if elements is None:
        elements = [f'Channel {i+1}' for i in range(temp2)]
    num_channels = len(elements)
    assert temp2 == 0 or temp2 == num_channels, "The data doesn't match the number of elements."
    assert temp1 == 1 or temp2 == 0 or temp1 == temp2, "The time doesn't match the size of the data."

    #% Calculations
    # build RMS indices
    if data_is_list:
        ix = {'one': [], 't_min': None, 't_max': None}
        for j in range(num_channels):
            temp_ix = get_rms_indices(time[j], xmin=rms_xmin, xmax=rms_xmax)
            ix['one'].append(temp_ix['one'])
            if j == 0:
                ix['pts'] = temp_ix['pts']
            else:
                ix['pts'] = [min((ix['pts'][0], temp_ix['pts'][0])), max((ix['pts'][1], temp_ix['pts'][1]))]
    else:
        ix = get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)
    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=num_channels)
    # calculate the rms (or mean) values
    if show_rms:
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x, y: np.nanmean(x, axis=y)
        if data_is_list:
            data_func = [func_lamb(data[j][ix['one'][j]], None) for j in range(num_channels)]
        elif data_as_rows:
            data_func = func_lamb(data[:, ix['one']], 1) if np.any(ix['one']) else np.full(num_channels, np.nan)
        else:
            data_func = func_lamb(data[ix['one'], :], 1) if np.any(ix['one']) else np.full(num_channels, np.nan)
    # unit conversion value
    (leg_conv, new_units) = get_legend_conversion(leg_scale, units)

    #% Create plots
    # create figures
    fig = plt.figure()
    fig.canvas.set_window_title(description)

    # create axes
    if single_lines:
        ax = []
        ax_prim = None
        for i in range(num_channels):
            temp_axes = fig.add_subplot(num_channels, 1, i+1, sharex=ax_prim)
            if ax_prim is None:
                ax_prim = temp_axes
            ax.append(temp_axes)
    else:
        ax = [fig.add_subplot(1, 1, 1)]

    # plot data
    for (i, this_axes) in enumerate(ax):
        if single_lines:
            loop_counter = [i]
        else:
            loop_counter = range(num_channels)
        # standard plot
        for j in loop_counter:
            this_label = f'{name} {elements[j]}' if name else str(elements[j])
            if show_rms:
                value = _LEG_FORMAT.format(leg_conv*data_func[j])
                if units:
                    this_label += f' ({func_name}: {value} {new_units})'
                else:
                    this_label += f' ({func_name}: {value})'
            this_time = time[j] if time_is_list else time
            this_data = data[j] if data_is_list else data[j, :] if data_as_rows else data[:, j]
            if use_zoh:
                this_axes.step(this_time, this_data, '.-', where='post', markersize=4, label=this_label, \
                    color=cm.get_color(j), zorder=9)
            else:
                this_axes.plot(this_time, this_data, '.-', markersize=4, label=this_label, \
                    color=cm.get_color(j), zorder=9)

        # set X display limits
        if i == 0:
            disp_xlimits(this_axes, xmin=disp_xmin, xmax=disp_xmax)
            xlim = this_axes.get_xlim()
        this_axes.set_xlim(xlim)
        zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # format display of plot
        if i == 0:
            this_axes.set_title(description)
        if (time_is_list and is_datetime(time[0])) or is_datetime(time):
            this_axes.set_xlabel('Date')
            assert time_units in {'datetime', 'numpy'}, 'Mismatch in the expected time units.'
        else:
            this_axes.set_xlabel(f'Time [{time_units}]{start_date}')
        if ylabel is None:
            this_axes.set_ylabel(f'{description} [{units}]')
        else:
            this_ylabel = ylabel[i] if isinstance(ylabel, list) else ylabel
            this_axes.set_ylabel(this_ylabel)
        this_axes.grid(True)
        # optionally add second Y axis
        plot_second_units_wrapper(this_axes, second_yscale)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)
    # plot any extra information through a generic callable
    if extra_plotter is not None:
        extra_plotter(fig=fig, ax=ax)
    # add legend at the very end once everything has been done
    if legend_loc.lower() != 'none':
        for this_axes in ax:
            this_axes.legend(loc=legend_loc)

    return fig

#%% Functions - make_error_bar_plot
def make_error_bar_plot(description, time, data, mins, maxs, *, elements=None, units='', time_units='sec', \
        leg_scale='unity', start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, \
        disp_xmax=inf, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, \
        plot_zero=False, show_rms=True, legend_loc='best', second_yscale=None, ylabel=None, \
        data_as_rows=True, label_vert_lines=True):
    r"""
    Generic plotting routine to make error bars.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title
    time : (N, ) array_like
        time history [sec]
    data : (A, N) ndarray
        data history
    mins : (A, N) ndarray
        data minimum bound history
    maxs : (A, N) ndarray
        data maximum bound history
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec'
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is 'unity'
    start_date : str, optional
        date of t(0), may be an empty string
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    single_lines : bool, optional
        flag meaning to plot subplots by channel instead of together
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best'
    second_yscale : dict, optional
        single key and value pair to use for scaling data to a second Y axis
    ylabel : str, optional
        Labels to put on the Y axes, potentially by element
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle
    err : (A,N) ndarray
        Differences

    See Also
    --------
    TBD_wrapper

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Made fully functional by David C. Stauffer in April 2020.

    Examples
    --------
    >>> from dstauffman.plotting import make_error_bar_plot
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from datetime import datetime
    >>> description      = 'Random Data Error Bars'
    >>> time             = np.arange(11)
    >>> data             = np.array([[3.], [-2.], [5]]) + np.random.rand(3, 11)
    >>> mins             = data - 0.5 * np.random.rand(3, 11)
    >>> maxs             = data + 1.5 * np.random.rand(3, 11)
    >>> elements         = ['x', 'y', 'z']
    >>> units            = 'rad'
    >>> time_units       = 'sec'
    >>> leg_scale        = 'milli'
    >>> start_date       = '  t0 = ' + str(datetime.now())
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = 'tab10'
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> second_yscale    = {'mrad': 1e3}
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> label_vert_lines = True
    >>> fig              = make_error_bar_plot(description, time, data, mins, maxs, elements=elements, \
    ...     units=units, time_units=time_units, leg_scale=leg_scale, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, second_yscale=second_yscale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, label_vert_lines=label_vert_lines)

    Close plots
    >>> plt.close(fig)

    """
    # data checks
    assert description, 'You must give the plot a description.'

    # convert rows/cols as necessary
    data = np.atleast_2d(data)
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        data = data.T
        mins = mins.T
        maxs = maxs.T

    # optional inputs
    if elements is None:
        elements = [f'Channel {i+1}' for i in range(data.shape[0])]
    # find number of elements being differenced
    num_channels = len(elements)

    #% Calculations
    # build RMS indices
    ix = get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)
    cm = ColorMap(colormap=colormap, num_colors=num_channels)
    # calculate the rms (or mean) values
    if show_rms:
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x: rms(x, axis=1, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x: np.nanmean(x, axis=1)
        data_func = func_lamb(data[:, ix['one']]) if np.any(ix['one']) else np.full(num_channels, np.nan)
    # unit conversion value
    (leg_conv, new_units) = get_legend_conversion(leg_scale, units)
    # error calculation
    err_neg = data - mins
    err_pos = maxs - data
    # get the number of axes to make
    if single_lines:
        num_axes = num_channels
    else:
        num_axes = 1

    #% Create plots
    # create figures
    fig = plt.figure()
    fig.canvas.set_window_title(description)
    # create axes
    ax = []
    ax_prim = None
    for i in range(num_axes):
        temp_axes = fig.add_subplot(num_axes, 1, i+1, sharex=ax_prim)
        if ax_prim is None:
            ax_prim = temp_axes
        ax.append(temp_axes)
    assert num_axes == len(ax), 'There is a mismatch in the number of axes.'
    # plot data
    for (i, this_axes) in enumerate(ax):
        if single_lines:
            loop_counter = [i]
        else:
            loop_counter = range(num_channels)
        # standard plot
        for j in loop_counter:
            if show_rms:
                value = _LEG_FORMAT.format(leg_conv*data_func[j])
                this_label = '{} ({}: {} {})'.format(elements[j], func_name, value, new_units)
            else:
                this_label = elements[j]
            this_axes.plot(time, data[j, :], '.-', markersize=4, label=this_label, \
                color=cm.get_color(j), zorder=3)
            # plot error bars
            this_axes.errorbar(time, data[j, :], yerr=np.vstack((err_neg[j, :], err_pos[j, :])), \
                color='None', ecolor=cm.get_color(j), zorder=5, capsize=2)

        # set X display limits
        if i == 0:
            disp_xlimits(this_axes, xmin=disp_xmin, xmax=disp_xmax)
            xlim = this_axes.get_xlim()
        this_axes.set_xlim(xlim)
        channel = i if single_lines else None
        zoom_ylim(this_axes, time, data.T, t_start=xlim[0], t_final=xlim[1], channel=channel)
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # format display of plot
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)
        this_axes.set_title(description)
        if is_datetime(time):
            this_axes.set_xlabel('Date')
            assert time_units == 'datetime', 'Mismatch in the expected time units.'
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if isinstance(ylabel, list):
            this_ylabel = ylabel[i]
        else:
            this_ylabel = ylabel
        if this_ylabel is None:
            this_axes.set_ylabel(description + ' [' + units + ']')
        else:
            this_axes.set_ylabel(this_ylabel + ' [' + units + ']')
        this_axes.grid(True)
        # optionally add second Y axis
        plot_second_units_wrapper(this_axes, second_yscale)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)

    return fig

#%% Functions - make_difference_plot
def make_difference_plot(description, time_one, time_two, data_one, data_two, *, \
        name_one='', name_two='', elements=None, units='', time_units='sec', leg_scale='unity', \
        start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, \
        make_subplots=True, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, \
        plot_zero=False, show_rms=True, legend_loc='best', show_extra=True, second_yscale=None, \
        ylabel=None, truth_name='Truth', truth_time=None, truth_data=None, data_as_rows=True, \
        tolerance=0, return_err=False, use_zoh=False, label_vert_lines=True):
    r"""
    Generic difference comparison plot for use in other wrapper functions.
    Plots two vector histories over time, along with a difference from one another.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title
    time_one : (A, ) array_like
        time history one [sec]
    time_two : (B, ) array_like
        time history two [sec]
    data_one : (N, A) ndarray
        vector one history
    data_two : (M, B) ndarray
        vector two history
    name_one : str, optional
        name of data source 1
    name_two : str, optional
        name of data source 2
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec'
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is 'unity'
    start_date : str, optional
        date of t(0), may be an empty string
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    make_subplots : bool, optional
        flag to use subplots for differences
    single_lines : bool, optional
        flag meaning to plot subplots by channel instead of together
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best'
    show_extra : bool, optional
        whether to show missing data on difference plots
    second_yscale : dict, optional
        single key and value pair to use for scaling data to a second Y axis
    ylabel : str, optional
        Labels to put on the Y axes, potentially by element
    truth_name : str, optional
        name to associate with truth data, default is 'Truth'
    truth_time : ndarray, optional
        truth time history
    truth_data : ndarray, optional
        truth quaternion history
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    tolerance : float, optional, default is zero
        Numerical tolerance on what should be considered a match between data_one and data_two
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles
    use_zoh : bool, optional, default is False
        Whether to plot as a zero-order hold, instead of linear interpolation between data points
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)

    Returns
    -------
    fig_hand : list of class matplotlib.Figure
        list of figure handles
    err : (A,N) ndarray
        Differences

    See Also
    --------
    TBD_wrapper

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Made fully functional by David C. Stauffer in April 2020.

    Examples
    --------
    >>> from dstauffman.plotting import make_difference_plot, get_color_lists
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from datetime import datetime
    >>> description      = 'example'
    >>> time_one         = np.arange(11)
    >>> time_two         = np.arange(2, 13)
    >>> data_one         = 50e-6 * np.random.rand(2, 11)
    >>> data_two         = data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6 * np.random.rand(2, 11)
    >>> name_one         = 'test1'
    >>> name_two         = 'test2'
    >>> elements         = ['x', 'y']
    >>> units            = 'rad'
    >>> time_units       = 'sec'
    >>> leg_scale        = 'micro'
    >>> start_date       = str(datetime.now())
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> color_lists      = get_color_lists()
    >>> colormap         = ListedColormap(color_lists['dbl_diff'].colors + color_lists['double'].colors)
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> show_extra       = True
    >>> second_yscale    = {u'µrad': 1e6}
    >>> ylabel           = None
    >>> truth_name       = 'Truth'
    >>> truth_time       = None
    >>> truth_data       = None
    >>> data_as_rows     = True
    >>> tolerance        = 0
    >>> return_err       = False
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig_hand = make_difference_plot(description, time_one, time_two, data_one, data_two, \
    ...     name_one=name_one, name_two=name_two, elements=elements, units=units, time_units=time_units, \
    ...     leg_scale=leg_scale, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, make_subplots=make_subplots, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     show_extra=show_extra, second_yscale=second_yscale, ylabel=ylabel, truth_name=truth_name, \
    ...     truth_time=truth_time, truth_data=truth_data, data_as_rows=data_as_rows, tolerance=tolerance, \
    ...     return_err=return_err, use_zoh=use_zoh, label_vert_lines=label_vert_lines)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # determine if you have the histories
    have_data_one = data_one is not None and np.any(~np.isnan(data_one))
    have_data_two = data_two is not None and np.any(~np.isnan(data_two))
    have_both     = have_data_one and have_data_two
    have_truth    = truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data))
    if not have_data_one and not have_data_two:
        logger.log(LogLevel.L5, f'No difference data was provided, so no plot was generated for "{description}".')
        # TODO: return NaNs instead of None for this case?
        out = ([], {'one': None, 'two': None, 'diff': None}) if return_err else []
        return out

    # data checks
    assert description, 'You must give the plot a description.'
    if have_data_one:
        assert data_one.ndim == 2, f'Data must be 2D, not {data_one.ndim}' # TODO: change this restriction
    if have_data_two:
        assert data_two.ndim == 2, f'Data must be 2D, not {data_two.ndim}' # TODO: change this restriction

    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        if have_data_one:
            data_one = data_one.T
        if have_data_two:
            data_two = data_two.T
        if have_truth:
            truth_data = truth_data.T

    # determine which plotting function to use
    if use_zoh:
        plot_func = lambda ax, *args, **kwargs: ax.step(*args, **kwargs, where='post')
    else:
        plot_func = lambda ax, *args, **kwargs: ax.plot(*args, **kwargs)

    # calculate sizes
    s1 = data_one.shape[0] if data_one is not None else 0
    s2 = data_two.shape[0] if data_two is not None else 0
    assert s1 == 0 or s2 == 0 or s1 == s2, f'Sizes of data channels must be consistent, got {s1} and {s2}.'
    if elements is None:
        elements = [f'Channel {i+1}' for i in range(np.max((s1, s2)))]
    num_channels = len(elements)
    assert num_channels == np.maximum(s1, s2), 'The given elements need to match the data sizes, got ' + \
        '{} and {}.'.format(num_channels, np.maximum(s1, s2))

    #% Calculations
    if have_both:
        # find overlapping times
        (time_overlap, d1_diff_ix, d2_diff_ix) = intersect(time_one, time_two, tolerance=tolerance, \
            return_indices=True)
        # find differences
        d1_miss_ix = np.setxor1d(np.arange(len(time_one)), d1_diff_ix)
        d2_miss_ix = np.setxor1d(np.arange(len(time_two)), d2_diff_ix)
    else:
        time_overlap = None
    # build RMS indices
    ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)
    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=3*num_channels)
    # calculate the differences
    if have_both:
        diffs = data_two[:, d2_diff_ix] - data_one[:, d1_diff_ix]
    # calculate the rms (or mean) values
    if show_rms or return_err:
        nans = np.full(num_channels, np.nan, dtype=float)
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x: rms(x, axis=1, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x: np.nanmean(x, axis=1)
        data1_func    = func_lamb(data_one[:, ix['one']]) if have_data_one and np.any(ix['one']) else nans
        data2_func    = func_lamb(data_two[:, ix['two']]) if have_data_two and np.any(ix['two']) else nans
        nondeg_func   = func_lamb(diffs[:, ix['overlap']]) if have_both and np.any(ix['overlap']) else nans
        # output errors
        err = {'one': data1_func, 'two': data2_func, 'diff': nondeg_func}
    # unit conversion value
    (leg_conv, new_units) = get_legend_conversion(leg_scale, units)
    # determine which symbols to plot with
    if have_both:
        symbol_one = '^-'
        symbol_two = 'v:'
    elif have_data_one:
        symbol_one = '.-'
        symbol_two = '' # not-used
    elif have_data_two:
        symbol_one = '' # not-used
        symbol_two = '.-'
    else:
        symbol_one = '' # invalid case
        symbol_two = '' # invalid case
    # pre-plan plot layout
    if have_both:
        if make_subplots:
            num_figs = 1
            if single_lines:
                num_rows = num_channels
                num_cols = 2
            else:
                num_rows = 2
                num_cols = 1
        else:
            num_figs = 2
            num_cols = 1
            if single_lines:
                num_rows = num_channels
            else:
                num_rows = 1
    else:
        num_figs = 1
        if single_lines:
            num_rows = num_channels
            num_cols = 1
        else:
            num_rows = 1
            num_cols = 1
    num_axes = num_figs * num_rows * num_cols

    #% Create plots
    # create figures
    f1 = plt.figure()
    f1.canvas.set_window_title(description)
    if have_both and not make_subplots:
        f2 = plt.figure()
        f2.canvas.set_window_title(description + ' Difference')
        fig_hand = [f1, f2]
    else:
        fig_hand = [f1]
    # create axes
    ax = []
    ax_prim = None
    for i in range(num_figs):
        for j in range(num_cols):
            for k in range(num_rows):
                temp_axes = fig_hand[i].add_subplot(num_rows, num_cols, k*num_cols + j + 1, sharex=ax_prim)
                if ax_prim is None:
                    ax_prim = temp_axes
                ax.append(temp_axes)
    assert num_axes == len(ax), 'There is a mismatch in the number of axes.'
    # plot data
    for (i, this_axes) in enumerate(ax):
        is_diff_plot = i > num_rows-1 or (not single_lines and make_subplots and i == 1)
        if single_lines:
            if is_diff_plot:
                loop_counter = [i - num_rows]
            else:
                loop_counter = [i]
        else:
            loop_counter = range(num_channels)
        if not is_diff_plot:
            # standard plot
            if have_data_one:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(leg_conv*data1_func[j])
                        this_label = '{} {} ({}: {} {})'.format(name_one, elements[j], func_name, \
                            value, new_units)
                    else:
                        this_label = name_one + ' ' + elements[j]
                    plot_func(this_axes, time_one, data_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=cm.get_color(j), zorder=3)
            if have_data_two:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(leg_conv*data2_func[j])
                        this_label = '{} {} ({}: {} {})'.format(name_two, elements[j], func_name, \
                            value, new_units)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    plot_func(this_axes, time_two, data_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=cm.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            for j in loop_counter:
                if single_lines and i % num_channels != j:
                    continue
                if show_rms:
                    value = _LEG_FORMAT.format(leg_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {})'.format(elements[j], func_name, value, new_units)
                else:
                    this_label = elements[j]
                plot_func(this_axes, time_overlap, diffs[j, :], '.-', markersize=4, label=this_label, \
                    color=cm.get_color(j+2*num_channels))
            if show_extra:
                this_axes.plot(time_one[d1_miss_ix], np.zeros(len(d1_miss_ix)), 'kx', markersize=8, \
                    markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_two[d2_miss_ix], np.zeros(len(d2_miss_ix)), 'go', markersize=8, \
                    markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

        # set X display limits
        if i == 0:
            disp_xlimits(this_axes, xmin=disp_xmin, xmax=disp_xmax)
            xlim = this_axes.get_xlim()
        this_axes.set_xlim(xlim)
        zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # optionally plot truth (after having set axes limits)
        if i < num_rows and have_truth:
            # TODO: apply use_zoh logic to truth?
            if single_lines:
                this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, \
                    markerfacecolor=_TRUTH_COLOR, linewidth=2, label=truth_name + ' ' + elements[i])
            else:
                if i == 0:
                    # TODO: add RMS to Truth data?
                    this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, \
                        markerfacecolor=_TRUTH_COLOR, linewidth=2, label=truth_name)
        # format display of plot
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description)
        elif (single_lines and i == num_rows) or (not single_lines and i == 1):
            this_axes.set_title(description + ' Difference')
        if is_datetime(time_one) or is_datetime(time_two):
            this_axes.set_xlabel('Date')
            assert time_units in {'datetime', 'numpy'}, 'Expected time units of "datetime" on "numpy", ' + \
                'not "{}".'.format(time_units)
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if ylabel is None:
            if is_diff_plot:
                this_axes.set_ylabel(description + ' Difference [' + units + ']')
            else:
                this_axes.set_ylabel(description + ' [' + units + ']')
        else:
            # TODO: handle single_lines case by allowing list for ylabel
            bracket = ylabel.find('[')
            if is_diff_plot:
                if bracket > 0:
                    this_axes.set_ylabel(ylabel[:bracket-1] + ' Difference ' + ylabel[bracket:])
                else:
                    this_axes.set_ylabel(ylabel + ' Difference')
            else:
                this_axes.set_ylabel(ylabel)
        this_axes.grid(True)
        # optionally add second Y axis
        this_second_yscale = second_yscale[i] if isinstance(second_yscale, list) else second_yscale
        plot_second_units_wrapper(this_axes, this_second_yscale)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)

    if return_err:
        return (fig_hand, err)
    return fig_hand

#%% Functions - make_categories_plot
def make_categories_plot(description, time, data, cats, *, cat_names=None, name='', elements=None, \
        units='', time_units='sec', leg_scale='unity', start_date='', rms_xmin=-inf, \
        rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, make_subplots=True, single_lines=False, \
        colormap=DEFAULT_COLORMAP, use_mean=False, plot_zero=False, show_rms=True, \
        legend_loc='best', second_yscale=None, ylabel=None, data_as_rows=True, use_zoh=False, \
        label_vert_lines=True):
    r"""
    Data versus time plotting routine when grouped into categories.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title, must be given
    time : (A, ) array_like
        time history [sec] or datetime64
    data : (N, ) or (N, A) ndarray
        vector history
    name : str, optional
        name of data
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec', use 'datetime' for datetime histories
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is 'unity'
    start_date : str, optional
        date of t(0), may be an empty string
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    make_subplots : bool, optional
        flag to use subplots for differences
    single_lines : bool, optional
        whether to plot each channel on a new figure instead of subplots
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best', use 'none' to suppress legend
    second_yscale : dict, optional
        single key and value pair to use for scaling data to a second Y axis
    ylabel : str, optional
        Labels to put on the Y axes, potentially by element
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    use_zoh : bool, optional, default is False
        Whether to plot as a zero-order hold, instead of linear interpolation between data points
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)

    Returns
    -------
    figs : list of class matplotlib.Figure
        Figure handles

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman.plotting import make_categories_plot
    >>> import numpy as np
    >>> description      = 'Values vs Time'
    >>> time             = np.arange(-10., 10.1, 0.1)
    >>> data             = np.vstack((time + np.cos(time), np.ones(time.shape, dtype=float)))
    >>> data[1, 60:85]   = 2
    >>> MeasStatus       = type('MeasStatus', (object,), {'rejected': 0, 'accepted': 1})
    >>> cats             = np.full(time.shape, MeasStatus.accepted, dtype=int)
    >>> cats[50:100]     = MeasStatus.rejected
    >>> cat_names        = {0: 'rejected', 1: 'accepted'}
    >>> name             = ''
    >>> elements         = None
    >>> units            = ''
    >>> time_units       = 'sec'
    >>> leg_scale        = 'unity'
    >>> start_date       = ''
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> colormap         = 'Paired'
    >>> use_mean         = True
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> second_yscale    = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> figs = make_categories_plot(description, time, data, cats, cat_names=cat_names, name=name, \
    ...     elements=elements, units=units, time_units=time_units, leg_scale=leg_scale, \
    ...     start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, make_subplots=make_subplots, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     legend_loc=legend_loc, second_yscale=second_yscale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, use_zoh=use_zoh, label_vert_lines=label_vert_lines)

    Close plots
    >>> for fig in figs:
    ...     plt.close(fig)

    """
    # some basic flags
    time_is_list = isinstance(time, list)
    data_is_list = isinstance(data, list)

    # data checks
    assert description, 'You must give the plot a description.'

    # convert rows/cols as necessary
    if not data_is_list:
        data = np.atleast_2d(data)
        if not data_as_rows:
            # TODO: is this the best way or make branches lower?
            data = data.T

    # get the categories
    unique_cats = set(cats)
    num_cats = len(unique_cats)
    if cat_names is None:
        cat_names = {}
    # Add any missing dictionary values
    for x in unique_cats:
        if x not in cat_names:
            cat_names[x] = 'Status='+str(x)
    ordered_cats = [x for x in cat_names if x in unique_cats]
    cat_keys = np.array(list(cat_names.keys()), dtype=int)

    # calculate sizes
    temp1 = len(time) if time_is_list else 1
    temp2 = len(data) if data_is_list else data.shape[0] if data is not None else 0
    if elements is None:
        elements = [f'Channel {i+1}' for i in range(temp2)]
    num_channels = len(elements)
    assert temp2 == 0 or temp2 == num_channels, "The data doesn't match the number of elements."
    assert temp1 == 1 or temp2 == 0 or temp1 == temp2, "The time doesn't match the size of the data."

    #% Calculations
    # build RMS indices
    if data_is_list:
        ix = {'one': [], 't_min': None, 't_max': None}
        for j in range(num_channels):
            temp_ix = get_rms_indices(time[j], xmin=rms_xmin, xmax=rms_xmax)
            ix['one'].append(temp_ix['one'])
            if j == 0:
                ix['pts'] = temp_ix['pts']
            else:
                ix['pts'] = [min((ix['pts'][0], temp_ix['pts'][0])), max((ix['pts'][1], temp_ix['pts'][1]))]
    else:
        ix = get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)
    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=len(cat_keys)*num_channels)
    # calculate the rms (or mean) values
    if show_rms:
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x, y: np.nanmean(x, axis=y)
        data_func = {}
        for cat in ordered_cats:
            if data_is_list:
                this_ix = ix['one'][j] & (cats[j] == cat)
                data_func[cat] = [func_lamb(data[j][this_ix], None) for j in range(num_channels)]
            else:
                this_ix = ix['one'] & (cats == cat)
                data_func[cat] = func_lamb(data[:, this_ix], 1) if np.any(this_ix) else np.full(num_channels, np.nan)
    # unit conversion value
    (leg_conv, new_units) = get_legend_conversion(leg_scale, units)
    # pre-plan plot layout
    if make_subplots:
        num_figs = 1
        num_rows = num_channels
        num_cols = num_cats if single_lines else 1
    else:
        num_figs = num_channels * num_cats if single_lines else num_channels
        num_cols = 1
        num_rows = 1
    if single_lines:
        titles = [f'{description} {e} {cat_names[cat]}' for cat in ordered_cats for e in elements]
    else:
        titles = [f'{description} {e}' for e in elements]
    num_axes = num_figs * num_rows * num_cols

    #% Create plots
    # create figure(s) and axes
    figs = []
    ax = []
    ax_prim = None
    for i in range(num_figs):
        fig = plt.figure()
        fig.canvas.set_window_title(titles[i])
        for j in range(num_cols):
            for k in range(num_rows):
                temp_axes = fig.add_subplot(num_rows, num_cols, k*num_cols + j + 1, sharex=ax_prim)
                if ax_prim is None:
                    ax_prim = temp_axes
                ax.append(temp_axes)
        figs.append(fig)
    assert num_axes == len(ax), 'There is a mismatch in the number of axes.'

    # plot data
    for (i, this_axes) in enumerate(ax):
        if single_lines:
            ix_data = i % num_channels
            ix_cat  = [i // num_channels]
        else:
            ix_data = i
            ix_cat  = list(range(num_cats))
        # pull out data for this channel
        this_time = time[ix_data] if time_is_list else time
        this_data = data[ix_data] if data_is_list else data[ix_data, :]
        root_label = name if name else '' + str(elements[ix_data])
        # plot the full underlying line once
        if not single_lines:
            if use_zoh:
                this_axes.step(this_time, this_data, ':', where='post', \
                    label='', color='xkcd:slate', linewidth=1, zorder=2)
            else:
                this_axes.plot(this_time, this_data, ':', \
                    label='', color='xkcd:slate', linewidth=1, zorder=2)
        # plot the data with this category value
        for j in ix_cat:
            cat = ordered_cats[j]
            this_cat_name = cat_names[cat]
            if show_rms:
                value = _LEG_FORMAT.format(leg_conv*data_func[cat][ix_data])
                if units:
                    this_label = f'{root_label} {this_cat_name} ({func_name}: {value} {new_units})'
                else:
                    this_label = f'{root_label} {this_cat_name} ({func_name}: {value})'
            else:
                this_label = f'{root_label} {this_cat_name}'
            this_cats = cats == cat
            this_linestyle = '-' if single_lines else 'none'
            # Note: Use len(cat_keys) here instead of num_cats so that potentially missing categories
            # won't mess up th ecolor scheme by skipping colors
            this_cat_ix = np.argmax(cat == cat_keys)
            this_color = cm.get_color(this_cat_ix + ix_data*len(cat_keys))
            this_axes.plot(this_time[this_cats], this_data[this_cats], linestyle=this_linestyle, marker='.', \
                markersize=6, label=this_label, color=this_color, zorder=3)

        # set title and axes labels
        this_axes.set_title(titles[i])
        if (time_is_list and is_datetime(time[0])) or is_datetime(time):
            this_axes.set_xlabel('Date')
            assert time_units in {'datetime', 'numpy'}, 'Mismatch in the expected time units.'
        else:
            this_axes.set_xlabel(f'Time [{time_units}]{start_date}')
        if ylabel is None:
            this_axes.set_ylabel(f'{titles[i]} [{units}]')
        else:
            this_ylabel = ylabel[ix_data] if isinstance(ylabel, list) else ylabel
            this_axes.set_ylabel(this_ylabel)
        this_axes.grid(True)
        # optionally add second Y axis
        plot_second_units_wrapper(this_axes, second_yscale)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)

    # Manipulate the axes limits at the end, so you have the total of all the data
    # set X display limits
    disp_xlimits(ax, xmin=disp_xmin, xmax=disp_xmax)
    for this_axes in ax:
        xlim = this_axes.get_xlim()
        zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # format display of plot
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)

    return figs

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting_generic', exit=False)
    doctest.testmod(verbose=False)
