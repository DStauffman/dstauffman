# -*- coding: utf-8 -*-
r"""
Defines low-level plotting routines meant to be wrapped in higher level ones.

Notes
-----
#.  Written by David C. Stauffer in May 2020.

"""

#%% Imports
# normal imports
import doctest
import unittest

# plotting/numpy imports
import matplotlib.pyplot as plt
import numpy as np

# model imports
from dstauffman.constants    import DEFAULT_COLORMAP
from dstauffman.plot_support import ColorMap, disp_xlimits, get_color_lists, get_rms_indices, \
                                        is_datetime, plot_second_units_wrapper, plot_vert_lines, \
                                        show_zero_ylim, zoom_ylim
from dstauffman.quat  import quat_angle_diff
from dstauffman.stats import intersect
from dstauffman.units import get_factors
from dstauffman.utils import rms

#%% Constants
# hard-coded values
_LEG_FORMAT  = '{:1.3f}'
_TRUTH_COLOR = 'k'

#%% Functions - make_time_plot
def make_time_plot(description, time, data, name='', elements=None, units='', time_units='sec',
        leg_scale='unity', start_date='', rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf,
        disp_xmax=np.inf, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, plot_zero=False,
        show_rms=True, legend_loc='best', second_yscale=None, ylabel=None, data_as_rows=True):
    r"""
    Generic data versus time plotting routine.

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

    Returns
    -------
    fig : class matplotlib.Figure

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman import make_time_plot
    >>> import numpy as np
    >>> description = 'Values vs Time'
    >>> time = np.arange(-10., 10.1, 0.1)
    >>> data = time + np.cos(time)
    >>> name = ''
    >>> elements = None
    >>> units = ''
    >>> time_units = 'sec'
    >>> leg_scale = 'unity'
    >>> start_date = ''
    >>> rms_xmin = -np.inf
    >>> rms_xmax = np.inf
    >>> disp_xmin = -np.inf
    >>> disp_xmax = np.inf
    >>> single_lines = False
    >>> colormap = 'Paired'
    >>> use_mean = False
    >>> plot_zero = False
    >>> show_rms = True
    >>> legend_loc = 'best'
    >>> second_yscale = None
    >>> ylabel = None
    >>> data_as_rows = True
    >>> fig = make_time_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, leg_scale=leg_scale, start_date=start_date, rms_xmin=rms_xmin, \
    ...     rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     legend_loc=legend_loc, second_yscale=second_yscale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows)

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
    cm = ColorMap(colormap=colormap, num_colors=num_channels)
    # calculate the rms (or mean) values
    if not use_mean:
        func_name = 'RMS'
        func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
    else:
        func_name = 'Mean'
        func_lamb = lambda x, y: np.nanmean(x, axis=y)
    if data_is_list:
        data_func = func_lamb(data[j][ix['one'][j]], None)
    else:
        data_func = func_lamb(data[:, ix['one']], 1)
    # unit conversion value
    (temp, prefix) = get_factors(leg_scale)
    leg_conv = 1/temp
    if prefix:
        assert units, 'You must give units if using a non-unity scale factor.'

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
                    this_label += f' ({func_name}: {value} {prefix}{units})'
                else:
                    this_label += f' ({func_name}: {value})'
            this_time = time[j] if time_is_list else time
            this_data = data[j] if data_is_list else data[j, :]
            this_axes.plot(this_time, this_data, '.-', markersize=4, label=this_label, \
                color=cm.get_color(j), zorder=3)

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
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description)
        if is_datetime(time):
            this_axes.set_xlabel('Date')
            assert time_units == 'datetime', 'Mismatch in the expected time units.'
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
            plot_vert_lines(this_axes, ix['pts'])

    return fig

#%% Functions - make_error_bar_plot
def make_error_bar_plot(description, time, data, mins, maxs, elements=None, units='', time_units='sec', \
        leg_scale='unity', start_date='', rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf, \
        disp_xmax=np.inf, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, \
        plot_zero=False, show_rms=True, legend_loc='best', second_yscale=None, ylabel=None, \
        data_as_rows=True):
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
    #.  Made fully function by David C. Stauffer in April 2020.

    Examples
    --------
    >>> from dstauffman import make_error_bar_plot
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from datetime import datetime
    >>> description     = 'Random Data Error Bars'
    >>> time            = np.arange(11)
    >>> data            = np.array([[3.], [-2.], [5]]) + np.random.rand(3, 11)
    >>> mins            = data - 0.5 * np.random.rand(3, 11)
    >>> maxs            = data + 1.5 * np.random.rand(3, 11)
    >>> elements        = ['x', 'y', 'z']
    >>> units           = 'rad'
    >>> time_units      = 'sec'
    >>> leg_scale       = 'milli'
    >>> start_date      = '  t0 = ' + str(datetime.now())
    >>> rms_xmin        = 1
    >>> rms_xmax        = 10
    >>> disp_xmin       = -2
    >>> disp_xmax       = np.inf
    >>> single_lines    = False
    >>> colormap        = 'tab10'
    >>> use_mean        = False
    >>> plot_zero       = False
    >>> show_rms        = True
    >>> legend_loc      = 'best'
    >>> second_yscale   = {'mrad': 1e3}
    >>> ylabel          = None
    >>> data_as_rows    = True
    >>> fig             = make_error_bar_plot(description, time, data, mins, maxs, elements=elements, \
    ...     units=units, time_units=time_units, leg_scale=leg_scale, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, second_yscale=second_yscale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows)

    Close plots
    >>> plt.close(fig)

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman import make_error_bar_plot

    """
    # data checks
    assert description, 'You must give the plot a description.'

    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        data = data.T
        mins = mins.T
        maxs = maxs.T

    #% Calculations
    # build RMS indices
    ix = get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)
    # find number of elements being differenced
    num_channels = len(elements)
    cm = ColorMap(colormap=colormap, num_colors=num_channels)
    # calculate the rms (or mean) values
    if not use_mean:
        func_name = 'RMS'
        func_lamb = lambda x: rms(x, axis=1, ignore_nans=True)
    else:
        func_name = 'Mean'
        func_lamb = lambda x: np.nanmean(x, axis=1)
    data_func = func_lamb(data[:, ix['one']])
    # unit conversion value
    (temp, prefix) = get_factors(leg_scale)
    leg_conv = 1/temp
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
                this_label = '{} ({}: {} {}{})'.format(elements[j], func_name, value, prefix, units)
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
            plot_vert_lines(this_axes, ix['pts'])

    return fig

#%% Functions - make_quaternion_plot
def make_quaternion_plot(description, time_one, time_two, quat_one, quat_two, *,
        name_one='', name_two='', time_units='sec', start_date='', plot_components=True,
        rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf, disp_xmax=np.inf,
        make_subplots=True, single_lines=False, use_mean=False, plot_zero=False, show_rms=True,
        legend_loc='best', show_extra=True, truth_name='Truth', truth_time=None, truth_data=None,
        data_as_rows=True):
    r"""
    Generic quaternion comparison plot for use in other wrapper functions.
    Plots two quaternion histories over time, along with a difference from one another.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title
    time_one : (N, ) array_like
        time history one [sec]
    time_two : (M, ) array_like
        time history two [sec]
    quat_one : (4, N) ndarray
        quaternion one
    quat_two : (4, M) ndarray
        quaternion two
    name_one : str, optional
        name of data source 1
    name_two : str, optional
        name of data source 2
    time_units : str, optional
        time units, defaults to 'sec'
    start_date : str, optional
        date of t(0), may be an empty string
    plot_components : bool, optional
        Whether to plot the quaternion components, or just the angular difference
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
    truth_name : str, optional
        name to associate with truth data, default is 'Truth'
    truth_time : ndarray, optional
        truth time history
    truth_data : ndarray, optional
        truth quaternion history
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector

    Returns
    -------
    fig_hand : list of class matplotlib.Figure
        list of figure handles
    err : (3,N) ndarray
        Quaternion differences expressed in Q1 frame

    See Also
    --------
    TBD_wrapper

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in December 2018.
    #.  Made fully functional by David C. Stauffer in March 2019.

    Examples
    --------
    >>> from dstauffman import make_quaternion_plot, quat_norm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from datetime import datetime
    >>> description     = 'example'
    >>> time_one        = np.arange(11)
    >>> time_two        = np.arange(2, 13)
    >>> quat_one        = quat_norm(np.random.rand(4, 11))
    >>> quat_two        = quat_norm(quat_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] + 1e-5 * np.random.rand(4, 11))
    >>> name_one        = 'test1'
    >>> name_two        = 'test2'
    >>> time_units      = 'sec'
    >>> start_date      = str(datetime.now())
    >>> plot_components = True
    >>> rms_xmin        = 1
    >>> rms_xmax        = 10
    >>> disp_xmin       = -2
    >>> disp_xmax       = np.inf
    >>> make_subplots   = True
    >>> single_lines    = False
    >>> use_mean        = False
    >>> plot_zero       = False
    >>> show_rms        = True
    >>> legend_loc      = 'best'
    >>> show_extra      = True
    >>> truth_name      = 'Truth'
    >>> truth_time      = None
    >>> truth_data      = None
    >>> data_as_rows    = True
    >>> (fig_hand, err) = make_quaternion_plot(description, time_one, time_two, quat_one, quat_two,
    ...     name_one=name_one, name_two=name_two, time_units=time_units, start_date=start_date, \
    ...     plot_components=plot_components, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, make_subplots=make_subplots, single_lines=single_lines, \
    ...     use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     show_extra=show_extra, truth_name=truth_name, truth_time=truth_time, truth_data=truth_data, \
    ...     data_as_rows=data_as_rows)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # data checks
    assert description, 'You must give the plot a description.'

    # determine if you have the quaternions
    have_quat_one = quat_one is not None and np.any(~np.isnan(quat_one))
    have_quat_two = quat_two is not None and np.any(~np.isnan(quat_two))
    have_both     = have_quat_one and have_quat_two
    have_truth    = truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data))

    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        if have_quat_one:
            quat_one = quat_one.T
        if have_quat_two:
            quat_two = quat_two.T
        if have_truth:
            truth_data = truth_data.T

    #% Calculations
    # find overlapping times
    (time_overlap, q1_diff_ix, q2_diff_ix) = intersect(time_one, time_two) # TODO: add a tolerance?
    # find differences
    q1_miss_ix = np.setxor1d(np.arange(len(time_one)), q1_diff_ix)
    q2_miss_ix = np.setxor1d(np.arange(len(time_two)), q2_diff_ix)
    # build RMS indices
    ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)
    # get default plotting colors
    color_lists = get_color_lists()
    colororder3 = ColorMap(color_lists['vec'], num_colors=3)
    colororder8 = ColorMap(color_lists['quat_diff'], num_colors=8)
    # quaternion component names
    elements = ['X', 'Y', 'Z', 'S']
    num_channels = len(elements)
    # calculate the difference
    if have_both:
        (nondeg_angle, nondeg_error) = quat_angle_diff(quat_one[:, q1_diff_ix], quat_two[:, q2_diff_ix])
    # calculate the rms (or mean) values
    nans = np.full(3, np.nan, dtype=float)
    if not use_mean:
        func_name = 'RMS'
        func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
    else:
        func_name = 'Mean'
        func_lamb = lambda x, y: np.nanmean(x, axis=y)
    q1_func     = func_lamb(quat_one[:, ix['one']], 1) if have_quat_one else nans
    q2_func     = func_lamb(quat_two[:, ix['two']], 1) if have_quat_two else nans
    nondeg_func = func_lamb(nondeg_error[:, ix['overlap']], 1) if have_both else nans
    mag_func    = func_lamb(nondeg_angle[ix['overlap']], 0) if have_both else nans[0:1]
    # output errors
    err = {'one': q1_func, 'two': q2_func, 'diff': nondeg_func, 'mag': mag_func}
    # unit conversion value
    (temp, prefix) = get_factors('micro')
    leg_conv = 1/temp
    # determine which symbols to plot with
    if have_both:
        symbol_one = '^-'
        symbol_two = 'v:'
    elif have_quat_one:
        symbol_one = '.-'
        symbol_two = '' # not-used
    elif have_quat_two:
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
    num_axes = num_figs*num_rows*num_cols

    #% Create plots
    # create figures
    f1 = plt.figure()
    if make_subplots:
        f1.canvas.set_window_title(description)
    else:
        f1.canvas.set_window_title(description + ' Quaternion Components')
    if have_both and not make_subplots:
        f2 = plt.figure()
        f2.canvas.set_window_title(description + 'Difference')
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
    # plot data
    for i in range(num_axes):
        this_axes = ax[i]
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
            if have_quat_one:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(q1_func[j])
                        this_label = '{} {} ({}: {})'.format(name_one, elements[j], func_name, value)
                    else:
                        this_label = name_one + ' ' + elements[j]
                    this_axes.plot(time_one, quat_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+(0 if have_quat_two else num_channels)), zorder=3)
            if have_quat_two:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(q2_func[j])
                        this_label = '{} {} ({}: {})'.format(name_two, elements[j], func_name, value)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    this_axes.plot(time_two, quat_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            zorders = [8, 6, 5]
            for j in range(3):
                if not plot_components or (single_lines and i % num_channels != j):
                    continue
                if show_rms:
                    value = _LEG_FORMAT.format(leg_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {}rad)'.format(elements[j], func_name, value, prefix)
                else:
                    this_label = elements[j]
                this_axes.plot(time_overlap, nondeg_error[j, :], '.-', markersize=4, label=this_label, zorder=zorders[j], \
                    color=colororder3.get_color(j))
            if not plot_components or (single_lines and (i + 1) % num_channels == 0):
                if show_rms:
                    value = _LEG_FORMAT.format(leg_conv*mag_func)
                    this_label = 'Angle ({}: {} {}rad)'.format(func_name, value, prefix)
                else:
                    this_label = 'Angle'
                this_axes.plot(time_overlap, nondeg_angle, '.-', markersize=4, label=this_label, color=colororder3.get_color(0))
            if show_extra:
                this_axes.plot(time_one[q1_miss_ix], np.zeros(len(q1_miss_ix)), 'kx', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_one[q2_miss_ix], np.zeros(len(q2_miss_ix)), 'go', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

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
            if single_lines:
                this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, markerfacecolor=_TRUTH_COLOR, \
                    linewidth=2, label=truth_name + ' ' + elements[i])
            else:
                if i == 0:
                    # TODO: add RMS to Truth data?
                    this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, markerfacecolor=_TRUTH_COLOR, \
                        linewidth=2, label=truth_name)
        # format display of plot
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description + ' Quaternion Components')
        elif (single_lines and i == num_rows) or (not single_lines and i == 1):
            this_axes.set_title(description + ' Difference')
        if is_datetime(time_one) or is_datetime(time_two):
            this_axes.set_xlabel('Date')
            assert time_units == 'datetime', 'Mismatch in the expected time units.'
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if is_diff_plot:
            this_axes.set_ylabel(description + ' Difference [rad]')
            plot_second_units_wrapper(this_axes, {prefix+'rad': leg_conv})
        else:
            this_axes.set_ylabel(description + ' Quaternion Components [dimensionless]')
        this_axes.grid(True)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'])

    return (fig_hand, err)

#%% Functions - make_difference_plot
def make_difference_plot(description, time_one, time_two, data_one, data_two, *,
        name_one='', name_two='', elements=None, units=None, time_units='sec', leg_scale='unity',
        start_date='', rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf, disp_xmax=np.inf,
        make_subplots=True, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False,
        plot_zero=False, show_rms=True, legend_loc='best', show_extra=True, second_yscale=None,
        ylabel=None, truth_name='Truth', truth_time=None, truth_data=None, data_as_rows=True):
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
    #.  Made fully function by David C. Stauffer in April 2020.

    Examples
    --------
    >>> from dstauffman import make_difference_plot, get_color_lists
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from datetime import datetime
    >>> description     = 'example'
    >>> time_one        = np.arange(11)
    >>> time_two        = np.arange(2, 13)
    >>> data_one        = 50e-6 * np.random.rand(2, 11)
    >>> data_two        = data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6 * np.random.rand(2, 11)
    >>> name_one        = 'test1'
    >>> name_two        = 'test2'
    >>> elements        = ['x', 'y']
    >>> units           = 'rad'
    >>> time_units      = 'sec'
    >>> leg_scale       = 'micro'
    >>> start_date      = str(datetime.now())
    >>> rms_xmin        = 1
    >>> rms_xmax        = 10
    >>> disp_xmin       = -2
    >>> disp_xmax       = np.inf
    >>> make_subplots   = True
    >>> single_lines    = False
    >>> color_lists     = get_color_lists()
    >>> colormap        = ListedColormap(color_lists['dbl_diff'].colors + color_lists['double'].colors)
    >>> use_mean        = False
    >>> plot_zero       = False
    >>> show_rms        = True
    >>> legend_loc      = 'best'
    >>> show_extra      = True
    >>> second_yscale   = {u'µrad': 1e6}
    >>> ylabel          = None
    >>> truth_name      = 'Truth'
    >>> truth_time      = None
    >>> truth_data      = None
    >>> data_as_rows    = True
    >>> (fig_hand, err) = make_difference_plot(description, time_one, time_two, data_one, data_two,
    ...     name_one=name_one, name_two=name_two, elements=elements, units=units, time_units=time_units, \
    ...     leg_scale=leg_scale, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, make_subplots=make_subplots, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     show_extra=show_extra, second_yscale=second_yscale, ylabel=ylabel, truth_name=truth_name, \
    ...     truth_time=truth_time, truth_data=truth_data, data_as_rows=data_as_rows)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # data checks
    assert description, 'You must give the plot a description.'

    # determine if you have the histories
    have_data_one = data_one is not None and np.any(~np.isnan(data_one))
    have_data_two = data_two is not None and np.any(~np.isnan(data_two))
    have_both     = have_data_one and have_data_two
    have_truth    = truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data))

    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        if have_data_one:
            data_one = data_one.T
        if have_data_two:
            data_two = data_two.T
        if have_truth:
            truth_data = truth_data.T

    # calculate sizes
    s1 = data_one.shape[0] if data_one is not None else 0
    s2 = data_two.shape[0] if data_two is not None else 0
    assert s1 == 0 or s2 == 0 or s1 == s2, f'Sizes of data channels must be consistent, got {s1} and {s2}.'
    num_channels = len(elements)
    assert num_channels == np.maximum(s1, s2), 'The given elements need to match the data sizes, got {} and {}.'.format(num_channels, np.maximum(s1, s2))

    #% Calculations
    # find overlapping times
    (time_overlap, d1_diff_ix, d2_diff_ix) = intersect(time_one, time_two) # TODO: add a tolerance?
    # find differences
    d1_miss_ix = np.setxor1d(np.arange(len(time_one)), d1_diff_ix)
    d2_miss_ix = np.setxor1d(np.arange(len(time_two)), d2_diff_ix)
    # build RMS indices
    ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)
    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=3*num_channels)
    # calculate the differences
    if have_both:
        nondeg_error = data_two[:, d2_diff_ix] - data_one[:, d1_diff_ix]
    # calculate the rms (or mean) values
    nans = np.full(num_channels, np.nan, dtype=float)
    if not use_mean:
        func_name = 'RMS'
        func_lamb = lambda x: rms(x, axis=1, ignore_nans=True)
    else:
        func_name = 'Mean'
        func_lamb = lambda x: np.nanmean(x, axis=1)
    data1_func    = func_lamb(data_one[:, ix['one']]) if have_data_one and np.any(ix['one']) else nans
    data2_func    = func_lamb(data_two[:, ix['two']]) if have_data_two and np.any(ix['two']) else nans
    nondeg_func   = func_lamb(nondeg_error[:, ix['overlap']]) if have_both and np.any(ix['overlap']) else nans
    # output errors
    err = {'one': data1_func, 'two': data2_func, 'diff': nondeg_func}
    # unit conversion value
    (temp, prefix) = get_factors(leg_scale)
    leg_conv = 1/temp
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
    num_axes = num_figs*num_rows*num_cols

    #% Create plots
    # create figures
    f1 = plt.figure()
    f1.canvas.set_window_title(description)
    if have_both and not make_subplots:
        f2 = plt.figure()
        f2.canvas.set_window_title(description + 'Difference')
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
                        this_label = '{} {} ({}: {} {}{})'.format(name_one, elements[j], func_name, value, prefix, units)
                    else:
                        this_label = name_one + ' ' + elements[j]
                    this_axes.plot(time_one, data_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=cm.get_color(j), zorder=3)
            if have_data_two:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(leg_conv*data2_func[j])
                        this_label = '{} {} ({}: {} {}{})'.format(name_two, elements[j], func_name, value, prefix, units)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    this_axes.plot(time_two, data_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=cm.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            for j in loop_counter:
                if single_lines and i % num_channels != j:
                    continue
                if show_rms:
                    value = _LEG_FORMAT.format(leg_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {}{})'.format(elements[j], func_name, value, prefix, units)
                else:
                    this_label = elements[j]
                this_axes.plot(time_overlap, nondeg_error[j, :], '.-', markersize=4, label=this_label, \
                    color=cm.get_color(j+2*num_channels))
            if show_extra:
                this_axes.plot(time_one[d1_miss_ix], np.zeros(len(d1_miss_ix)), 'kx', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_one[d2_miss_ix], np.zeros(len(d2_miss_ix)), 'go', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

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
            if single_lines:
                this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, markerfacecolor=_TRUTH_COLOR, \
                    linewidth=2, label=truth_name + ' ' + elements[i])
            else:
                if i == 0:
                    # TODO: add RMS to Truth data?
                    this_axes.plot(truth_time, truth_data[i, :], '.-', color=_TRUTH_COLOR, markerfacecolor=_TRUTH_COLOR, \
                        linewidth=2, label=truth_name)
        # format display of plot
        if legend_loc.lower() != 'none':
            this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description)
        elif (single_lines and i == num_rows) or (not single_lines and i == 1):
            this_axes.set_title(description + ' Difference')
        if is_datetime(time_one) or is_datetime(time_two):
            this_axes.set_xlabel('Date')
            assert time_units == 'datetime', f'Expected time units of "datetime", not "{time_units}".'
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
            if is_diff_plot and bracket > 0:
                this_axes.set_ylabel(ylabel[:bracket-1] + ' Difference ' + ylabel[bracket:])
            else:
                this_axes.set_ylabel(ylabel)
        this_axes.grid(True)
        # optionally add second Y axis
        plot_second_units_wrapper(this_axes, second_yscale)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'])

    return (fig_hand, err)

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plot_generic', exit=False)
    doctest.testmod(verbose=False)
