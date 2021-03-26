r"""
Plots related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
import doctest
import logging
import unittest

from dstauffman import get_unit_conversion, HAVE_NUMPY, HAVE_MPL, intersect, is_datetime, LogLevel, \
    rms
from dstauffman.aerospace import Kf, KfInnov, quat_angle_diff
from dstauffman.plotting.generic import make_categories_plot, make_connected_sets, make_difference_plot
from dstauffman.plotting.plotting import Opts, setup_plots
from dstauffman.plotting.support import ColorMap, COLOR_LISTS, disp_xlimits, get_nondeg_colorlists, \
    get_rms_indices, plot_second_units_wrapper, plot_vert_lines, show_zero_ylim, zoom_ylim

if HAVE_MPL:
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np
    inf = np.inf
else:
    from math import inf

#%% Globals
logger = logging.getLogger(__name__)

#%% Constants
# hard-coded values
_LEG_FORMAT  = '{:1.3f}'
_TRUTH_COLOR = 'k'

#%% Functions - make_quaternion_plot
def make_quaternion_plot(description, time_one, time_two, quat_one, quat_two, *, \
        name_one='', name_two='', time_units='sec', start_date='', plot_components=True, \
        rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, \
        make_subplots=True, single_lines=False, use_mean=False, plot_zero=False, show_rms=True, \
        legend_loc='best', show_extra=True, second_units='micro', truth_name='Truth', \
        truth_time=None, truth_data=None, data_as_rows=True, tolerance=0, return_err=False, \
        use_zoh=False, label_vert_lines=True, extra_plotter=None):
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
    second_units : str or tuple[str, float], optional, default is 'micro'
        single name and value pair to use for scaling data to a second Y axis and showing in legend
    truth_name : str, optional
        name to associate with truth data, default is 'Truth'
    truth_time : ndarray, optional
        truth time history
    truth_data : ndarray, optional
        truth quaternion history
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    tolerance : float, optional, default is zero
        Numerical tolerance on what should be considered a match between quat_one and quat_two
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles
    use_zoh : bool, optional, default is False
        Whether to plot as a zero-order hold, instead of linear interpolation between data points
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)
    extra_plotter : callable, optional
        Extra callable plotting function to add more details to the plot

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
    >>> from dstauffman.plotting import make_quaternion_plot
    >>> from dstauffman.aerospace import quat_norm
    >>> import numpy as np
    >>> from datetime import datetime
    >>> description      = 'example'
    >>> time_one         = np.arange(11)
    >>> time_two         = np.arange(2, 13)
    >>> quat_one         = quat_norm(np.random.rand(4, 11))
    >>> quat_two         = quat_norm(quat_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] + 1e-5 * np.random.rand(4, 11))
    >>> name_one         = 'test1'
    >>> name_two         = 'test2'
    >>> time_units       = 'sec'
    >>> start_date       = str(datetime.now())
    >>> plot_components  = True
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> show_extra       = True
    >>> second_units     = (u'µrad', 1e6)
    >>> truth_name       = 'Truth'
    >>> truth_time       = None
    >>> truth_data       = None
    >>> data_as_rows     = True
    >>> tolerance        = 0
    >>> return_err       = False
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> fig_hand = make_quaternion_plot(description, time_one, time_two, quat_one, quat_two,
    ...     name_one=name_one, name_two=name_two, time_units=time_units, start_date=start_date, \
    ...     plot_components=plot_components, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
    ...     disp_xmin=disp_xmin, disp_xmax=disp_xmax, make_subplots=make_subplots, \
    ...     single_lines=single_lines, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     legend_loc=legend_loc, show_extra=show_extra, second_units=second_units, \
    ...     truth_name=truth_name, truth_time=truth_time, truth_data=truth_data, \
    ...     data_as_rows=data_as_rows, tolerance=tolerance, return_err=return_err, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines, extra_plotter=extra_plotter)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # determine if you have the quaternions
    have_quat_one = quat_one is not None and np.any(~np.isnan(quat_one))
    have_quat_two = quat_two is not None and np.any(~np.isnan(quat_two))
    have_both     = have_quat_one and have_quat_two
    have_truth    = truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data))
    if not have_quat_one and not have_quat_two:
        logger.log(LogLevel.L5, f'No quaternion data was provided, so no plot was generated for "{description}".')
        # TODO: return NaNs instead of None for this case?
        out = ([], {'one': None, 'two': None, 'diff': None, 'mag': None}) if return_err else []
        return out

    # data checks
    assert description, 'You must give the plot a description.' # TODO: remove this restriction?

    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        if have_quat_one:
            quat_one = quat_one.T
        if have_quat_two:
            quat_two = quat_two.T
        if have_truth:
            truth_data = truth_data.T

    # determine which plotting function to use
    if use_zoh:
        plot_func = lambda ax, *args, **kwargs: ax.step(*args, **kwargs, where='post')
    else:
        plot_func = lambda ax, *args, **kwargs: ax.plot(*args, **kwargs)

    #% Calculations
    if have_both:
        # find overlapping times
        (time_overlap, q1_diff_ix, q2_diff_ix) = intersect(time_one, time_two, tolerance=tolerance, return_indices=True)
        # find differences
        q1_miss_ix = np.setxor1d(np.arange(len(time_one)), q1_diff_ix)
        q2_miss_ix = np.setxor1d(np.arange(len(time_two)), q2_diff_ix)
    else:
        time_overlap = None
    # build RMS indices
    ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)
    # get default plotting colors
    colororder3 = ColorMap(COLOR_LISTS['vec'])
    colororder8 = ColorMap(COLOR_LISTS['quat_diff'])
    # quaternion component names
    elements = ['X', 'Y', 'Z', 'S']
    num_channels = len(elements)
    units = 'rad'
    # calculate the difference
    if have_both:
        (nondeg_angle, nondeg_error) = quat_angle_diff(quat_one[:, q1_diff_ix], quat_two[:, q2_diff_ix])
    # calculate the rms (or mean) values
    if show_rms or return_err:
        nans = np.full(3, np.nan, dtype=float)
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x, y: np.nanmean(x, axis=y)
        q1_func     = func_lamb(quat_one[:, ix['one']], 1) if have_quat_one and np.any(ix['one']) else nans
        q2_func     = func_lamb(quat_two[:, ix['two']], 1) if have_quat_two and np.any(ix['two']) else nans
        nondeg_func = func_lamb(nondeg_error[:, ix['overlap']], 1) if have_both and np.any(ix['overlap']) else nans
        mag_func    = func_lamb(nondeg_angle[ix['overlap']], 0) if have_both and np.any(ix['overlap']) else nans[0:1]
        # output errors
        err = {'one': q1_func, 'two': q2_func, 'diff': nondeg_func, 'mag': mag_func}
    # unit conversion value
    (new_units, unit_conv) = get_unit_conversion(second_units, units)
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
                    plot_func(this_axes, time_one, quat_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+(0 if have_quat_two else num_channels)), zorder=3)
            if have_quat_two:
                for j in loop_counter:
                    if show_rms:
                        value = _LEG_FORMAT.format(q2_func[j])
                        this_label = '{} {} ({}: {})'.format(name_two, elements[j], func_name, value)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    plot_func(this_axes, time_two, quat_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            zorders = [8, 6, 5]
            for j in range(3):
                if not plot_components or (single_lines and i % num_channels != j):
                    continue
                if show_rms:
                    value = _LEG_FORMAT.format(unit_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {})'.format(elements[j], func_name, value, new_units)
                else:
                    this_label = elements[j]
                plot_func(this_axes, time_overlap, nondeg_error[j, :], '.-', markersize=4, label=this_label, zorder=zorders[j], \
                    color=colororder3.get_color(j))
            if not plot_components or (single_lines and (i + 1) % num_channels == 0):
                if show_rms:
                    value = _LEG_FORMAT.format(unit_conv*mag_func)
                    this_label = 'Angle ({}: {} {})'.format(func_name, value, new_units)
                else:
                    this_label = 'Angle'
                plot_func(this_axes, time_overlap, nondeg_angle, '.-', markersize=4, label=this_label, color=colororder3.get_color(0))
            if show_extra:
                this_axes.plot(time_one[q1_miss_ix], np.zeros(len(q1_miss_ix)), 'kx', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_two[q2_miss_ix], np.zeros(len(q2_miss_ix)), 'go', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

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
            assert time_units in {'datetime', 'numpy'}, 'Mismatch in the expected time units.'
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if is_diff_plot:
            this_axes.set_ylabel('Difference [' + units + ']')
            # optionally add second Y axis
            plot_second_units_wrapper(this_axes, (new_units, unit_conv))
        else:
            this_axes.set_ylabel('Quaternion Components [dimensionless]')
        this_axes.grid(True)
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        for fig in fig_hand:
            extra_plotter(fig=fig, ax=fig.axes)

    if return_err:
        return (fig_hand, err)
    return fig_hand

#%% plot_attitude
def plot_attitude(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""
    Plots the attitude quaternion history.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman.plotting import Opts, plot_attitude
    >>> from dstauffman.aerospace import Kf, quat_from_euler, quat_mult, quat_norm
    >>> import numpy as np

    >>> q1 = quat_norm(np.array([0.1, -0.2, 0.3, 0.4]))
    >>> dq = quat_from_euler(1e-6*np.array([-300, 100, 200]), [3, 1, 2])
    >>> q2 = quat_mult(dq, q1)

    >>> kf1      = Kf()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.att  = np.tile(q1[:, np.newaxis], (1, kf1.time.size))

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.att  = np.tile(q2[:, np.newaxis], (1, kf2.time.size))
    >>> kf2.att[3,4] += 50e-6
    >>> kf2.att = quat_norm(kf2.att)

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.quat_comp = True
    >>> opts.sub_plots = True

    >>> fig_hand = plot_attitude(kf1, kf2, opts=opts)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        truth = Kf()
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'att': 'Attitude Quaternion'}

    # alias keywords
    name_one = kwargs.pop('name_one', kf1.name)
    name_two = kwargs.pop('name_two', kf2.name)

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {'numpy', 'datetime'}

    # make local copy of opts that can be modified without changing the original
    this_opts = opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates('numpy', old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates('sec', old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    plot_comps   = kwargs.pop('plot_components', this_opts.quat_comp)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # hard-coded defaults
    second_units = kwargs.pop('second_units', 'micro')

    # initialize outputs
    figs    = []
    err     = dict()
    printed = False

    # call wrapper function for most of the details
    for (field, description) in fields.items():
        # print status
        if not printed:
            logger.log(LogLevel.L4, f'Plotting {description} plots ...')
            printed = True
        # make plots
        out = make_quaternion_plot(description, kf1.time, kf2.time, getattr(kf1, field), getattr(kf2, field), \
            name_one=name_one, name_two=name_two, time_units=time_units, start_date=start_date, \
            rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, plot_components=plot_comps, single_lines=single_lines, \
            use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
            second_units=second_units, truth_name=truth.name, truth_time=truth.time, \
            truth_data=truth.att, return_err=return_err, **kwargs)
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, '... done.')
    if return_err:
        return (figs, err)
    return figs

#%% plot_los
def plot_los(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {'los': 'LOS'}
    out = plot_attitude(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% plot_position
def plot_position(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""
    Plots the position and velocity history.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman.plotting import plot_position
    >>> from dstauffman.aerospace import Kf
    >>> import numpy as np

    >>> kf1      = Kf()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.pos  = 1e6 * np.random.rand(3, 11)
    >>> kf1.vel  = 1e3 * np.random.rand(3, 11)

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.pos  = kf1.pos[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e5
    >>> kf2.vel  = kf1.vel[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 100

    >>> fig_hand = plot_position(kf1, kf2)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        truth = Kf()
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'pos': 'Position'}

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {'numpy', 'datetime'}

    # make local copy of opts that can be modified without changing the original
    this_opts = opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates('numpy', old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates('sec', old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # hard-coded defaults
    elements      = kwargs.pop('elements', ['x', 'y', 'z'])
    default_units = 'm' if 'pos' in fields else 'm/s' if 'vel' in fields else ''
    units         = kwargs.pop('units', default_units)
    second_units  = kwargs.pop('second_units', 'kilo')
    colormap      = get_nondeg_colorlists(3)
    name_one      = kwargs.pop('name_one', kf1.name)
    name_two      = kwargs.pop('name_two', kf2.name)

    # initialize outputs
    figs = []
    err  = dict()
    printed = False

    # call wrapper function for most of the details
    for (field, description) in fields.items():
        # print status
        if not printed:
            logger.log(LogLevel.L4, f'Plotting {description} plots ...')
            printed = True
        # make plots
        out = make_difference_plot(description, kf1.time, kf2.time, getattr(kf1, field), getattr(kf2, field), \
            name_one=name_one, name_two=name_two, elements=elements, time_units=time_units, units=units, \
            start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
            single_lines=single_lines, show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, \
            return_err=return_err, **kwargs)
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, '... done.')
    if return_err:
        return (figs, err)
    return figs

#%% plot_velocity
def plot_velocity(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {'vel': 'Velocity'}
    out = plot_position(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% plot_innovations
def plot_innovations(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, \
        plot_by_status=False, plot_by_number=False, show_one=None, show_two=None, cat_names=None, \
        cat_colors=None, number_field=None, number_colors=None, **kwargs):
    r"""
    Plots the Kalman Filter innovation histories.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles
    fields : dict, optional
        Name of the innovation fields to plot
    plot_by_status : bool, optional, default is False
        Whether to make an additional plot of all innovations by status (including rejected ones)
    plot_by_number : bool, optional, default is False
        Whether to plot innovations by number (quad/SCA etc.)
    show_one : ndarray of bool, optional
        Index to the innovations to plot from kf1, shows all if not given
    show_two : ndarray of bool, optional
        Index to the innovations to plot from kf2, shows all if not given
    cat_names : dict[int, str], optional
        Name of the different possible categories for innovation status, otherwise uses their numeric values
    cat_colors : list or colormap, optional
        colors to use on the categories plot
    number_field : dict[int, str], optional
        Field name and label to use for plotting by number (quad/SCA etc.)
    number_colors : list or colormap, optional
        colors to use on the quad/SCA number plot
    kwargs : dict
        Additional arguments passed on to the lower level plotting functions

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman.plotting import Opts, plot_innovations
    >>> from dstauffman.aerospace import KfInnov
    >>> import numpy as np

    >>> num_axes   = 2
    >>> num_innovs = 11

    >>> kf1       = KfInnov()
    >>> kf1.units = 'm'
    >>> kf1.time  = np.arange(num_innovs, dtype=float)
    >>> kf1.innov = 1e-6 * np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
    >>> kf1.norm  = np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)

    >>> ix        = np.hstack((np.arange(7), np.arange(8, num_innovs)))
    >>> kf2       = KfInnov()
    >>> kf2.time  = kf1.time[ix]
    >>> kf2.innov = kf1.innov[:, ix] + 1e-8 * np.random.rand(num_axes, ix.size)
    >>> kf2.norm  = kf1.norm[:, ix] + 0.1 * np.random.rand(num_axes, ix.size)

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.sub_plots = True

    >>> fig_hand = plot_innovations(kf1, kf2, opts=opts)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = KfInnov()
    if kf2 is None:
        kf2 = KfInnov()
    if truth is None:
        pass # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'innov': 'Innovations', 'norm': 'Normalized Innovations'}
    if number_field is None:
        number_field = {'quad': 'Quad', 'sca': 'SCA'}

    # aliases and defaults
    name_one     = kwargs.pop('name_one', kf1.name)
    name_two     = kwargs.pop('name_two', kf2.name)
    description  = name_one if name_one else name_two if name_two else ''
    num_chan = 0
    for key in fields.keys():
        num_chan = max(num_chan, getattr(kf1, key).shape[0] if getattr(kf1, key) is not None else getattr(kf2, key).shape[0] if getattr(kf2, key) is not None else 0)
    elements     = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f'Channel {i+1}' for i in range(num_chan)]
    elements     = kwargs.pop('elements', elements)
    units        = kwargs.pop('units', kf1.units)
    second_units = kwargs.pop('second_units', 'micro')

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {'numpy', 'datetime'}

    # make local copy of opts that can be modified without changing the original
    this_opts = opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates('numpy', old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates('sec', old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)
    colormap     = kwargs.pop('colormap', this_opts.colormap)
    tolerance    = kwargs.pop('tolerance', 0)

    # Initialize outputs
    figs    = []
    err     = dict()
    printed = False

    #% call wrapper functions for most of the details
    for (field, sub_description) in fields.items():
        full_description = description + ' - ' + sub_description if description else sub_description
        # print status
        if not printed:
            logger.log(LogLevel.L4, f'Plotting {full_description} plots ...')
            printed = True
        # make plots
        if 'Normalized' in sub_description:
            units = u'σ'
            this_second_units = 'unity'
        else:
            this_second_units = second_units
        field_one = getattr(kf1, field)
        field_two = getattr(kf2, field)
        if field_one is not None and show_one is not None:
            t1 = kf1.time[show_one]
            f1 = field_one[:, show_one]
        else:
            t1 = kf1.time
            f1 = field_one
        if field_two is not None and show_two is not None:
            t2 = kf2.time[show_two]
            f2 = field_two[:, show_two]
        else:
            t2 = kf2.time
            f2 = field_two
        out = make_difference_plot(full_description, t1, t2, f1, f2, name_one=name_one, \
            name_two=name_two, elements=elements, units=units, time_units=time_units, start_date=start_date, \
            rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
            single_lines=single_lines, legend_loc=legend_loc, second_units=this_second_units, \
            colormap=colormap, return_err=return_err, tolerance=tolerance, **kwargs)
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out
        this_ylabel = [e + ' Innovation [' + units + ']' for e in elements]
        if plot_by_status and field_one is not None and kf1.status is not None:
            figs += make_categories_plot(full_description+' by Category', kf1.time, field_one, kf1.status, \
                name=name_one, cat_names=cat_names, elements=elements, units=units, time_units=time_units, \
                start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
                disp_xmax=disp_xmax, make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, \
                show_rms=show_rms, single_lines=single_lines, legend_loc=legend_loc, \
                second_units=this_second_units, ylabel=this_ylabel, colormap=cat_colors, **kwargs)
        if plot_by_status and field_two is not None and kf2.status is not None:
            figs += make_categories_plot(full_description+' by Category', kf2.time, field_two, kf2.status, \
                name=name_two, cat_names=cat_names, elements=elements, units=units, time_units=time_units, \
                start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
                disp_xmax=disp_xmax, make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, \
                show_rms=show_rms, single_lines=single_lines, legend_loc=legend_loc, \
                second_units=this_second_units, ylabel=this_ylabel, colormap=cat_colors, **kwargs)
        if plot_by_number and field_one is not None:
            this_number = None
            for (quad, quad_name) in number_field.items():
                if hasattr(kf1, quad):
                    this_number = getattr(kf1, quad)
                    break
            if this_number is not None:
                num_names = {num: quad_name + ' ' + str(num) for num in np.unique(this_number)}
                figs += make_categories_plot(full_description+' by '+quad_name, kf1.time, field_one, this_number, \
                    name=name_one, cat_names=num_names, elements=elements, units=units, time_units=time_units, \
                    start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
                    disp_xmax=disp_xmax, make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, \
                    show_rms=show_rms, single_lines=single_lines, legend_loc=legend_loc, \
                    second_units=this_second_units, ylabel=this_ylabel, colormap=number_colors, **kwargs)
        if plot_by_number and field_two is not None:
            this_number = None
            for (quad, quad_name) in number_field.items():
                if hasattr(kf2, quad):
                    this_number = getattr(kf2, quad)
                    break
            if this_number is not None:
                num_names = {num: quad_name + ' ' + str(num) for num in np.unique(this_number)}
                figs += make_categories_plot(full_description+' by '+quad_name, kf2.time, field_two, this_number, \
                    name=name_two, cat_names=num_names, elements=elements, units=units, time_units=time_units, \
                    start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
                    disp_xmax=disp_xmax, make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, \
                    show_rms=show_rms, single_lines=single_lines, legend_loc=legend_loc, \
                    second_units=this_second_units, ylabel=this_ylabel, colormap=number_colors, **kwargs)

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, '... done.')
    if return_err:
        return (figs, err)
    return figs

#%% plot_innov_fplocs
def plot_innov_fplocs(kf1, *, opts=None, t_bounds=None, **kwargs):
    r"""
    Plots the innovations on the focal plane, connecting the sighting and prediction with the innovation.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    opts : class Opts, optional
        Plotting options
    fields : dict, optional
        Name of the innovation fields to plot
    kwargs : dict
        Additional arguments passed on to the lower level plotting functions

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman.plotting import Opts, plot_innov_fplocs
    >>> from dstauffman.aerospace import KfInnov
    >>> import numpy as np

    >>> num_axes   = 2
    >>> num_innovs = 11

    >>> kf1       = KfInnov()
    >>> kf1.units = 'm'
    >>> kf1.time  = np.arange(num_innovs, dtype=float)
    >>> kf1.innov = np.full((num_axes, num_innovs), 5e-3) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
    >>> kf1.innov[:, :5] *= 0.1
    >>> kf1.fploc = np.full((num_axes, num_innovs), 0.05) + 0.2 * np.random.rand(num_axes, num_innovs) - 0.1

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.sub_plots = True

    >>> fig_hand = plot_innov_fplocs(kf1, opts=opts, color_by='magnitude')

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = KfInnov()
    if opts is None:
        opts = Opts()

    name = kf1.name + ' - ' if kf1.name else ''
    description = name + 'Focal Plane Sightings'
    logger.log(LogLevel.L4, f'Plotting {description} plots ...')

    # alias opts
    legend_loc = kwargs.pop('legend_loc', opts.leg_spot)

    # pull out time subset
    if t_bounds is None:
        fplocs = kf1.fploc
        innovs = kf1.innov
    else:
        ix = get_rms_indices(kf1.time, xmin=t_bounds[0], xmax=t_bounds[1])
        fplocs = kf1.fploc[:, ix['one']]
        innovs = kf1.innov[:, ix['one']]

    # call wrapper functions for most of the details
    fig = make_connected_sets(description, fplocs, innovs, units=kf1.units, \
        legend_loc=legend_loc, **kwargs)

    # Setup plots
    figs = [fig]
    setup_plots(figs, opts)
    if figs:
        logger.log(LogLevel.L4, '... done.')
    else:
        logger.log(LogLevel.L5, 'No focal plane data was provided, so no plots were generated.')
    return figs

#%% plot_covariance
def plot_covariance(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, groups=None, fields=None, **kwargs):
    r"""
    Plots the Kalman Filter square root diagonal variance value.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman.plotting import Opts, plot_covariance
    >>> from dstauffman.aerospace import Kf
    >>> import numpy as np

    >>> num_points = 11
    >>> num_states = 6

    >>> kf1        = Kf()
    >>> kf1.name   = 'KF1'
    >>> kf1.time   = np.arange(num_points, dtype=float)
    >>> kf1.covar  = 1e-6 * np.tile(np.arange(1, num_states+1, dtype=float)[:, np.newaxis], (1, num_points))
    >>> kf1.active = np.array([1, 2, 3, 4, 8, 12])

    >>> kf2        = Kf(name='KF2')
    >>> kf2.time   = kf1.time
    >>> kf2.covar  = kf1.covar + 1e-9 * np.random.rand(*kf1.covar.shape)
    >>> kf2.active = kf1.active

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.sub_plots = True

    >>> fig_hand = plot_covariance(kf1, kf2, opts=opts)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        pass # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'covar': 'Covariance'}

    # TODO: allow different sets of states in the different structures

    # aliases and defaults
    num_chan = 0
    for key in fields.keys():
        num_chan = max(num_chan, getattr(kf1, key).shape[0] if getattr(kf1, key) is not None else getattr(kf2, key).shape[0] if getattr(kf2, key) is not None else 0)
    elements     = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f'Channel {i+1}' for i in range(num_chan)]
    elements     = kwargs.pop('elements', elements)
    units        = kwargs.pop('units', 'mixed')
    second_units = kwargs.pop('second_units', 'micro')
    name_one     = kwargs.pop('name_one', kf1.name)
    name_two     = kwargs.pop('name_two', kf2.name)
    if groups is None:
        groups = [i for i in range(num_chan)]

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {'numpy', 'datetime'}

    # make local copy of opts that can be modified without changing the original
    this_opts = opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates('numpy', old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates('sec', old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)  # TODO: why do I have this line?  Need to use this_opts below?

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # initialize output
    figs    = []
    err     = dict()
    printed = False

    #% call wrapper functions for most of the details
    for (field, description) in fields.items():
        # print status
        if not printed:
            logger.log(LogLevel.L4, f'Plotting {description} plots ...')
            printed = True
        # make plots
        err[field] = {}
        for (ix, states) in enumerate(groups):
            this_units  = units if isinstance(units, str) else units[ix]
            this_2units = second_units[ix] if isinstance(second_units, list) else second_units
            this_ylabel = description + f' [{this_units}]'
            states      = np.atleast_1d(states)
            if hasattr(kf1, 'active') and kf1.active is not None:
                (this_state_nums1, this_state_rows1, _) = intersect(kf1.active, states, return_indices=True)
            else:
                this_state_nums1 = np.array([], dtype=int)
            if hasattr(kf2, 'active') and kf2.active is not None:
                (this_state_nums2, this_state_rows2, _) = intersect(kf2.active, states, return_indices=True)
            else:
                this_state_nums2 = np.array([], dtype=int)
            this_state_nums = np.union1d(this_state_nums1, this_state_nums2)
            data_one   = np.atleast_2d(getattr(kf1, field)[this_state_rows1, :]) if getattr(kf1, field) is not None else None
            data_two   = np.atleast_2d(getattr(kf2, field)[this_state_rows2, :]) if getattr(kf2, field) is not None else None
            have_data1 = data_one is not None and np.any(~np.isnan(data_one))
            have_data2 = data_two is not None and np.any(~np.isnan(data_two))
            if have_data1 or have_data2:
                this_description = description + ' for State ' + ','.join(str(x) for x in this_state_nums)
                this_elements = [elements[state] for state in this_state_nums]
                colormap = get_nondeg_colorlists(len(this_elements))
                out = make_difference_plot(this_description, kf1.time, kf2.time, data_one, data_two, \
                    name_one=name_one, name_two=name_two, elements=this_elements, units=this_units, time_units=time_units, \
                    start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
                    make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
                    single_lines=single_lines, legend_loc=legend_loc, second_units=this_2units, return_err=return_err, \
                    ylabel=this_ylabel, colormap=colormap, **kwargs)
                if return_err:
                    figs += out[0]
                    err[field][f'Group {ix}'] = out[1]
                else:
                    figs += out
    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, '... done.')
    if not figs:
        logger.log(LogLevel.L5, 'No covariance data was provided, so no plots were generated.')
    if return_err:
        return (figs, err)
    return figs

#%% plot_states
def plot_states(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Kalman Filter state histories."""
    if fields is None:
        fields = {'state': 'State Estimates'}
    out = plot_covariance(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% Unit Test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting_aerospace', exit=False)
    doctest.testmod(verbose=False)
