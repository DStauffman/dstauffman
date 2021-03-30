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

from dstauffman import DEGREE_SIGN, get_unit_conversion, HAVE_MPL, HAVE_NUMPY, intersect, \
    is_datetime, LogLevel, RAD2DEG, rms
from dstauffman.aerospace import quat_angle_diff

from dstauffman.plotting.support import COLOR_LISTS, ColorMap, DEFAULT_COLORMAP, disp_xlimits, \
    get_rms_indices, ignore_plot_data, plot_second_units_wrapper, plot_vert_lines, show_zero_ylim, \
    zoom_ylim

if HAVE_MPL:
    from matplotlib.collections import LineCollection
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

#%% Functions - make_generic_plot
def make_generic_plot(plot_type, description, time_one, data_one, *, time_two=None, data_two=None, \
        mins=None, maxs=None, name_one='', name_two='', elements=None, units='', time_units='sec', \
        start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, \
        single_lines=False, make_subplots=True, colormap=DEFAULT_COLORMAP, use_mean=False, \
        plot_zero=False, show_rms=True, ignore_empties=False, legend_loc='best', show_extra=True, \
        plot_components=True, second_units=None, ylabel=None, tolerance=0, return_err=False, \
        data_as_rows=True, extra_plotter=None, use_zoh=False, label_vert_lines=True):
    r"""
    Generic plotting function called by all the other low level plots.

    This plot is not meant to be called directly, but is an internal version.

    Parameters
    ----------
    plot_type : str
        The time of plot to create, from {'time', 'bar', 'errorbar', 'cats', 'categories', 'quat', 'quaternion'}
    description : str
        name of the data being plotted, used as title
    time_one : (A, ) array_like
        time history for channel one, [sec] or datetime64
    data_one : (A, ) or (N, A) ndarray, or (A, N) ndarray if data_as_rows is False
        vector history for channel one
    time_two : (A, ) array_like
        time history for channel two, [sec] or datetime64
    data_two : (A, ) or (N, A) ndarray, or (A, N) ndarray if data_as_rows is False
        vector history for channel two
    mins : (A, ) or (N, A) ndarray, or (A, N) ndarray if data_as_rows is False
        vector history of minimums
    maxs : (A, ) or (N, A) ndarray, or (A, N) ndarray if data_as_rows is False
        vector history of maximums
    name_one : str, optional
        name of data source one
    name_two : str, optional
        name of data source two
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec', use 'datetime' for datetime histories
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
    make_subplots : bool, optional
        flag to use subplots for differences
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    ignore_empties : bool, optional, default is False
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    legend_loc : str, optional
        location to put the legend, default is 'best', use 'none' to suppress legend
    show_extra : bool, optional
        whether to show missing data on difference plots
    plot_components : bool, optional, default is True
        whether to plot the quaternion angular differences as components or magnitude
    second_units : str or tuple of (str, float), optional
        Name and conversion factor to use for scaling data to a second Y axis and in legend
    ylabel : str or List[str], optional
        Labels to put on the Y axes, potentially by element
    tolerance : float, optional
        Tolerance for what is considered the same point in time for difference plots [sec] or [datetime64]
    return_err : bool, optional, default is False
        Whether to return the difference errors
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
    #.  Made into super complicated full-blown version in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import make_generic_plot
    >>> import numpy as np
    >>> plot_type        = 'time'
    >>> description      = 'Values vs Time'
    >>> time_one         = np.arange(-10., 10.1, 0.1)
    >>> data_one         = time_one + np.cos(time_one)
    >>> name_one         = ''
    >>> elements         = None
    >>> units            = ''
    >>> time_units       = 'sec'
    >>> start_date       = ''
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> make_subplots    = False
    >>> colormap         = 'Paired'
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> ignore_empties   = False
    >>> legend_loc       = 'best'
    >>> show_extra       = True
    >>> second_units     = None
    >>> ylabel           = None
    >>> tolerance        = 0
    >>> return_err       = False
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig = make_generic_plot(plot_type, description, time_one, data_one, name_one=name_one, \
    ...     elements=elements, units=units, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     single_lines=single_lines, make_subplots=make_subplots, colormap=colormap, \
    ...     use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, ignore_empties=ignore_empties, \
    ...     legend_loc=legend_loc, show_extra=show_extra, second_units=second_units, ylabel=ylabel, \
    ...     tolerance=tolerance, return_err=return_err, data_as_rows=data_as_rows, \
    ...     extra_plotter=extra_plotter, use_zoh=use_zoh, label_vert_lines=label_vert_lines)

    Close the plot
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    # some basic flags and checks
    assert plot_type in {'time', 'bar', 'errorbar', 'cats', 'categories', 'diff', 'differencs', \
        'quat', 'quaternion'}, f'Unexpected plot type: {plot_type}.'
    assert isinstance(description, str), 'The description should be a string, check your argument order.'
    doing_diffs = plot_type in {'diff', 'differences', 'quat', 'quaternions'}
    is_quat_diff = plot_type in {'quat', 'quaternions'}
    fig_lists = plot_type in {'cats', 'categorical', 'diff', 'differences', 'quat', 'quaternions'}
    time_is_list = isinstance(time_one, list) or isinstance(time_one, tuple)
    if time_is_list:
        assert time_two is None or isinstance(time_two, list) or isinstance(time_two, tuple), \
            'Both times must be lists if one is.'
    data_is_list = isinstance(data_one, list) or isinstance(data_one, tuple)
    dat2_is_list = isinstance(data_two, list) or isinstance(data_two, tuple)
    if doing_diffs:
        assert not data_is_list and not dat2_is_list, "Data can't be lists for diffs right now."  # TODO: remove this restriction
        have_data_one = data_one is not None and np.any(~np.isnan(data_one))
        have_data_two = data_two is not None and np.any(~np.isnan(data_two))
        have_both     = have_data_one and have_data_two
        if not have_data_one and not have_data_two:
            logger.log(LogLevel.L5, 'No %s data was provided, so no plot was generated for "%s".', plot_type, description)
            if not return_err:
                return []
            # TODO: return NaNs instead of None for this case?
            out = ([], {'one': None, 'two': None, 'diff': None})
            if is_quat_diff:
                out[1]['mag'] = None
            return out
        if have_data_one:
            assert not data_is_list
            assert data_one.ndim == 2, f'Data must be 2D, not {data_one.ndim}' # TODO: change this restriction
        if have_data_two:
            assert not dat2_is_list
            assert data_two.ndim == 2, f'Data must be 2D, not {data_two.ndim}' # TODO: change this restriction
        # convert rows/cols as necessary
        if not data_as_rows:
            # TODO: is this the best way or make branches lower?
            if have_data_one:
                data_one = data_one.T
            if have_data_two:
                data_two = data_two.T
    else:
        have_data_one = have_data_two = have_both = False
    if not time_is_list and time_one is not None:
        time_one = np.atleast_1d(time_one)
    if not data_is_list and data_one is not None:
        data_one = np.atleast_2d(data_one)
        assert data_one.ndim < 3, 'data_one must be 0d, 1d or 2d.'
    if not dat2_is_list and data_two is not None:
        data_two = np.atleast_2d(data_two)
        assert data_two.ndim < 3, 'data_two must be 0d, 1d or 2d.'

    # check for valid data
    # TODO: implement this
    if ignore_plot_data(data_one, ignore_empties) and ignore_plot_data(data_two, ignore_empties):
        raise NotImplementedError('Not yet implemented')

    # determine which plotting function to use
    if use_zoh:
        plot_func = lambda ax, *args, **kwargs: ax.step(*args, **kwargs, where='post')
    else:
        plot_func = lambda ax, *args, **kwargs: ax.plot(*args, **kwargs)

    # calculate sizes
    s0a = 0 if time_one is None else len(time_one) if time_is_list else 1
    s0b = 0 if time_two is None else len(time_two) if time_is_list else 1
    if data_one is None:
        s1 = 0
    elif data_is_list:
        s1 = len(data_one)
    elif data_as_rows:
        s1 = data_one.shape[0]
    else:
        s1 = data_one.shape[1]
        if is_quat_diff:
            assert data_one.shape[0] == 4
    if data_two is None:
        s2 = 0
    elif dat2_is_list:
        s2 = len(data_two)
    elif data_as_rows:
        s2 = data_two.shape[0]
    else:
        s2 = data_two.shape[1]

    # optional inputs
    if elements is None:
        elements = [f'Channel {i+1}' for i in range(np.max((s1, s2)))]
    # find number of elements being plotted
    num_channels = len(elements)
    assert num_channels == np.maximum(s1, s2), 'The given elements need to match the data sizes, got ' + \
        '{} and {}.'.format(num_channels, np.maximum(s1, s2))
    assert s0a == 0 or s0a == 1 or s0a == num_channels, "The time doesn't match the number of elements."
    assert s0b == 0 or s0b == 1 or s0b == num_channels, "The time doesn't match the number of elements."
    assert s1 == 0 or s2 == 0 or s1 == s2, f'Sizes of data channels must be consistent, got {s1} and {s2}.'
    if is_quat_diff:
        assert s1 == 0 or s1 == 4, 'Must be a 4-element quaternion'
        assert s2 == 0 or s2 == 4, 'Must be a 4-element quaternion'

    #% Calculations
    # build RMS indices
    if data_is_list:
        ix = {'one': [], 't_min': None, 't_max': None}
        for j in range(num_channels):
            temp_ix = get_rms_indices(time_one[j], xmin=rms_xmin, xmax=rms_xmax)
            ix['one'].append(temp_ix['one'])
            if j == 0:
                ix['pts'] = temp_ix['pts']
            else:
                ix['pts'] = [min((ix['pts'][0], temp_ix['pts'][0])), max((ix['pts'][1], temp_ix['pts'][1]))]
    elif doing_diffs:
        if have_both:
            # find overlapping times
            (time_overlap, d1_diff_ix, d2_diff_ix) = intersect(time_one, time_two, tolerance=tolerance, \
                return_indices=True)
            # find differences
            d1_miss_ix = np.setxor1d(np.arange(len(time_one)), d1_diff_ix)
            d2_miss_ix = np.setxor1d(np.arange(len(time_two)), d2_diff_ix)
        else:
            time_overlap = None
        ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)
    else:
        ix = get_rms_indices(time_one, xmin=rms_xmin, xmax=rms_xmax)
    # create a colormap
    if doing_diffs:
        if is_quat_diff:
            cm_vec = ColorMap(COLOR_LISTS['vec'])
        cm = ColorMap(colormap=colormap, num_colors=3*num_channels)
    else:
        cm = ColorMap(colormap=colormap, num_colors=num_channels)
    # calculate the differences
    if doing_diffs and have_both:
        if is_quat_diff:
            (nondeg_angle, nondeg_error) = quat_angle_diff(data_one[:, d1_diff_ix], data_two[:, d2_diff_ix])
        else:
            diffs = data_two[:, d2_diff_ix] - data_one[:, d1_diff_ix]
    # calculate the rms (or mean) values
    if show_rms or return_err:
        nans = np.full(num_channels, np.nan, dtype=float)  # TODO: num_channels should be 3 for is_quat_diff
        if not use_mean:
            func_name = 'RMS'
            func_lamb = lambda x, y: rms(x, axis=y, ignore_nans=True)
        else:
            func_name = 'Mean'
            func_lamb = lambda x, y: np.nanmean(x, axis=y)
        if not doing_diffs:
            if data_is_list:
                data_func = [func_lamb(data_one[j][ix['one'][j]], None) for j in range(num_channels)]
            elif data_as_rows:
                data_func = func_lamb(data_one[:, ix['one']], 1) if np.any(ix['one']) else np.full(num_channels, np.nan)
            else:
                data_func = func_lamb(data_one[ix['one'], :], 1) if np.any(ix['one']) else np.full(num_channels, np.nan)
        if doing_diffs:
            # TODO: combine with non diff version
            data_func  = func_lamb(data_one[:, ix['one']], 1) if have_data_one and np.any(ix['one']) else nans
            data2_func = func_lamb(data_two[:, ix['two']], 1) if have_data_two and np.any(ix['two']) else nans
            if is_quat_diff:
                nondeg_func = func_lamb(nondeg_error[:, ix['overlap']], 1) if have_both and np.any(ix['overlap']) else nans
                mag_func    = func_lamb(nondeg_angle[ix['overlap']], 0) if have_both and np.any(ix['overlap']) else nans[0:1]
            else:
                nondeg_func = func_lamb(diffs[:, ix['overlap']], 1) if have_both and np.any(ix['overlap']) else nans
            # output errors
            err = {'one': data_func, 'two': data2_func, 'diff': nondeg_func}
            if is_quat_diff:
                err['mag'] = mag_func

    # unit conversion value
    (new_units, unit_conv) = get_unit_conversion(second_units, units)
    symbol_one = '.-'
    symbol_two = '.-'
    if plot_type == 'errorbar':
        # error calculation
        # TODO: handle data_is_list and rows cases
        err_neg = data_one - mins
        err_pos = maxs - data_one
    elif plot_type == 'bar':
        # TODO: handle data_is_list and rows cases
        if data_is_list:
            bottoms = [np.cumsum(data_one[j]) for j in range(num_channels)]
        elif data_as_rows:
            bottoms = np.concatenate((np.zeros((1, len(time_one))), np.cumsum(data_one, axis=0)), axis=0)
        else:
            bottoms = np.concatenate((np.zeros((len(time_one), 1)), np.cumsum(data_one, axis=1)), axis=1)
    elif doing_diffs:
        if have_both:
            symbol_one = '^-'
            symbol_two = 'v:'
    # get the number of axes to make
    if plot_type == 'bar':
        num_rows = num_cols = 1
    elif doing_diffs:
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
    else:
        num_figs = 1
        num_rows = num_channels if single_lines else 1
        num_cols = 1
    num_axes = num_figs * num_rows * num_cols

    #% Create plots
    # create figures
    fig = plt.figure()
    if is_quat_diff and not make_subplots:
        fig.canvas.set_window_title(description + ' Quaternion Components')
    else:
        fig.canvas.set_window_title(description)
    if doing_diffs:
        if have_both and not make_subplots:
            f2 = plt.figure()
            f2.canvas.set_window_title(description + ' Difference')
            figs = [fig, f2]
        else:
            figs = [fig]
    # create axes
    ax = []
    ax_prim = None
    for i in range(num_figs):
        for j in range(num_cols):
            for k in range(num_rows):
                temp_axes = fig.add_subplot(num_rows, num_cols, k*num_cols + j + 1, sharex=ax_prim)
                if ax_prim is None:
                    ax_prim = temp_axes
                ax.append(temp_axes)
    assert num_axes == len(ax), 'There is a mismatch in the number of axes.'
    # plot data
    for (i, this_axes) in enumerate(ax):
        is_diff_plot = doing_diffs and (i > num_rows-1 or (not single_lines and make_subplots and i == 1))
        if plot_type == 'bar':
            loop_counter = reversed(range(num_channels))
        elif single_lines:
            if is_diff_plot:
                if is_quat_diff:
                    loop_counter = range(3)
                else:
                    loop_counter = [i - num_rows]
            else:
                loop_counter = [i]
        else:
            loop_counter = range(num_channels) if not is_quat_diff else range(3)
        if not is_diff_plot:
            # standard plot
            for j in loop_counter:
                this_label = f'{name_one} {elements[j]}' if name_one else str(elements[j])
                if show_rms:
                    value = _LEG_FORMAT.format(unit_conv*data_func[j])
                    if new_units:
                        this_label += f' ({func_name}: {value} {new_units})'
                    else:
                        this_label += f' ({func_name}: {value})'
                if not doing_diffs or (doing_diffs and have_data_one):
                    this_time = time_one[j] if time_is_list else time_one
                    this_data = data_one[j] if data_is_list else data_one[j, :] if data_as_rows else data_one[:, j]
                if plot_type == 'errorbar':
                    this_zorder = 3
                elif doing_diffs:
                    this_zorder = 3 if is_quat_diff else 4
                else:
                    this_zorder = 9
                if plot_type == 'bar':
                    this_bottom1 = bottoms[j] if data_is_list else bottoms[j, :] if data_as_rows else bottoms[:, j]
                    this_bottom2 = bottoms[j+1] if data_is_list else bottoms[j+1, :] if data_as_rows else bottoms[:, j+1]
                    if not ignore_plot_data(this_data, ignore_empties):
                        # Note: The performance of ax.bar is really slow with large numbers of bars (>20), so
                        # fill_between is a better alternative
                        this_axes.fill_between(this_time, this_bottom1, this_bottom2, step='mid', \
                            label=this_label, color=cm.get_color(j), edgecolor='none')
                else:
                    if not doing_diffs or (doing_diffs and have_data_one):
                        if is_quat_diff and not have_data_two:
                            # TODO: get rid of this special case or rework into colormap?
                            this_color = cm.get_color(j + num_channels)
                        else:
                            this_color = cm.get_color(j)
                        plot_func(this_axes, this_time, this_data, symbol_one, markersize=4, label=this_label, \
                            color=this_color, zorder=this_zorder)
                    if doing_diffs and have_data_two:
                        this_data2 = data_two[j] if data_is_list else data_two[j, :] if data_as_rows else data_two[:, j]
                        this_label2 = f'{name_two} {elements[j]}' if name_two else str(elements[j])
                        if show_rms:
                            value = _LEG_FORMAT.format(unit_conv*data2_func[j])
                            if new_units:
                                this_label2 += f' ({func_name}: {value} {new_units})'
                            else:
                                this_label2 += f' ({func_name}: {value})'
                        plot_func(this_axes, time_two, this_data2, symbol_two, markersize=4, label=this_label2, \
                            color=cm.get_color(j+num_channels), zorder=this_zorder+1)
                if plot_type == 'errorbar':
                    # plot error bars
                    this_axes.errorbar(this_time, this_data, yerr=np.vstack((err_neg[j, :], err_pos[j, :])), \
                        color='None', ecolor=cm.get_color(j), zorder=5, capsize=2)
        else:
            #% Difference plot
            for j in loop_counter:
                if single_lines and i % num_channels != j and not is_quat_diff or (is_quat_diff and not plot_components):
                    continue
                if show_rms:
                    value = _LEG_FORMAT.format(unit_conv*nondeg_func[j])
                    this_label = f'{elements[j]} ({func_name}: {value}) {new_units})'
                else:
                    this_label = elements[j]
                this_data = nondeg_error[j, :] if is_quat_diff else diffs[j, :]
                this_zorder = [8, 6, 5][j] if is_quat_diff else 5
                this_color = cm_vec.get_color(j) if is_quat_diff else cm.get_color(j+2*num_channels)
                plot_func(this_axes, time_overlap, this_data, '.-', markersize=4, label=this_label, \
                    color=this_color)
            if is_quat_diff and not plot_components or (single_lines and (i + 1) % num_channels == 0):
                if show_rms:
                    value = _LEG_FORMAT.format(unit_conv*mag_func)
                    this_label = f'Angle ({func_name}: {value} {new_units})'
                else:
                    this_label = 'Angle'
                plot_func(this_axes, time_overlap, nondeg_angle, '.-', markersize=4, label=this_label, color=cm_vec.get_color(0))
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
        if plot_type == 'bar':
            # TODO: generalize this
            this_axes.set_ylim(0, 100)
        else:
            zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # format display of plot
        if i == 0:
            if is_quat_diff:
                this_axes.set_title(description + ' Quaternion Components')
            else:
                this_axes.set_title(description)
        elif doing_diffs and ((single_lines and i == num_rows) or (not single_lines and i == 1)):
            this_axes.set_title(description + ' Difference')
        if (time_is_list and is_datetime(time_one[0])) or is_datetime(time_one) or is_datetime(time_two):
            this_axes.set_xlabel('Date')
            assert time_units in {'datetime', 'numpy'}, 'Expected time units of "datetime" or "numpy", ' + \
                'not "{}".'.format(time_units)
        else:
            this_axes.set_xlabel(f'Time [{time_units}]{start_date}')
        if ylabel is None:
            if is_diff_plot:
                if is_quat_diff:
                    this_axes.set_ylabel('Quaternion Components [dimensionless]')
                else:
                    this_axes.set_ylabel(f'{description} Difference [{units}]')
            else:
                this_axes.set_ylabel(f'{description} [{units}]')
        else:
            this_ylabel = ylabel[i] if isinstance(ylabel, list) else ylabel
            if is_diff_plot:
                bracket = this_ylabel.find('[')
                if bracket > 0:
                    this_axes.set_ylabel(this_ylabel[:bracket-1] + ' Difference ' + this_ylabel[bracket:])
                else:
                    this_axes.set_ylabel(this_ylabel + ' Difference')
            else:
                this_axes.set_ylabel(this_ylabel)
        this_axes.grid(True)
        # optionally add second Y axis
        plot_second_units_wrapper(this_axes, (new_units, unit_conv))
        # plot RMS lines
        if show_rms:
            plot_vert_lines(this_axes, ix['pts'], show_in_legend=label_vert_lines)

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        if fig_lists:
            for fig in figs:
                extra_plotter(fig=fig, ax=fig.axes)
        else:
            extra_plotter(fig=fig, ax=ax)

    # add legend at the very end once everything has been done
    if legend_loc.lower() != 'none':
        for this_axes in ax:
            this_axes.legend(loc=legend_loc)

    if return_err:
        if fig_lists:
            return (figs, err)
        return (fig, err)
    if fig_lists:
        return figs
    return fig

#%% Functions - make_time_plot
def make_time_plot(description, time, data, *, name='', elements=None, units='', time_units='sec', \
        start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, \
        single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, plot_zero=False, \
        show_rms=True, ignore_empties=False, legend_loc='best', second_units=None, ylabel=None, \
        data_as_rows=True, extra_plotter=None, use_zoh=False, label_vert_lines=True):
    r"""
    Generic data versus time plotting routine.

    See make_generic_plot for input details.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

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
    >>> ignore_empties   = False
    >>> legend_loc       = 'best'
    >>> second_units     = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig = make_time_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
    ...     disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     ignore_empties=ignore_empties, legend_loc=legend_loc, second_units=second_units, \
    ...     ylabel=ylabel, data_as_rows=data_as_rows, extra_plotter=extra_plotter, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines)

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    return make_generic_plot(plot_type='time', description=description, time_one=time, data_one=data, \
        name_one=name, elements=elements, units=units, time_units=time_units, start_date=start_date, \
        rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
        single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
        show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, ylabel=ylabel, \
        data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
        label_vert_lines=label_vert_lines)

#%% Functions - make_error_bar_plot
def make_error_bar_plot(description, time, data, mins, maxs, *, elements=None, units='', \
        time_units='sec', start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, \
        disp_xmax=inf, single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, \
        plot_zero=False, show_rms=True, legend_loc='best', second_units=None, ylabel=None, \
        data_as_rows=True, extra_plotter=None, use_zoh=False, label_vert_lines=True):
    r"""
    Generic plotting routine to make error bars.

    See make_generic_plot for input details.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

    See Also
    --------
    make_generic_plot

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Made fully functional by David C. Stauffer in April 2020.
    #.  Wrapped to the generic do everything version by David C. Stauffer in March 2021

    Examples
    --------
    >>> from dstauffman.plotting import make_error_bar_plot
    >>> import numpy as np
    >>> from datetime import datetime
    >>> description      = 'Random Data Error Bars'
    >>> time             = np.arange(11)
    >>> data             = np.array([[3.], [-2.], [5]]) + np.random.rand(3, 11)
    >>> mins             = data - 0.5 * np.random.rand(3, 11)
    >>> maxs             = data + 1.5 * np.random.rand(3, 11)
    >>> elements         = ['x', 'y', 'z']
    >>> units            = 'rad'
    >>> time_units       = 'sec'
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
    >>> second_units     = 'milli'
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig              = make_error_bar_plot(description, time, data, mins, maxs, \
    ...     elements=elements, units=units, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
    ...     label_vert_lines=label_vert_lines)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    return make_generic_plot('errorbar', description=description, time_one=time, data_one=data, \
        mins=mins, maxs=maxs, elements=elements, units=units, time_units=time_units, \
        start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
        disp_xmax=disp_xmax, single_lines=single_lines, colormap=colormap, use_mean=use_mean, \
        plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, \
        ylabel=ylabel, data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
        label_vert_lines=label_vert_lines)

#%% Functions - make_difference_plot
def make_difference_plot(description, time_one, time_two, data_one, data_two, *, \
        name_one='', name_two='', elements=None, units='', time_units='sec', start_date='', \
        rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, make_subplots=True, \
        single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=False, plot_zero=False, \
        show_rms=True, legend_loc='best', show_extra=True, second_units=None, ylabel=None, \
        data_as_rows=True, tolerance=0, return_err=False, use_zoh=False, label_vert_lines=True, \
        extra_plotter=None):
    r"""
    Generic difference comparison plot for use in other wrapper functions.
    Plots two vector histories over time, along with a difference from one another.

    See make_generic_plot for input details.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle
    err : Dict
        Differences

    See Also
    --------
    make_generic_plot

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Made fully functional by David C. Stauffer in April 2020.
    #.  Wrapped to the generic do everything version by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import make_difference_plot, get_nondeg_colorlists
    >>> import numpy as np
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
    >>> start_date       = str(datetime.now())
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> colormap         = get_nondeg_colorlists(2)
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = 'best'
    >>> show_extra       = True
    >>> second_units     = (u'µrad', 1e6)
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> tolerance        = 0
    >>> return_err       = False
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> fig_hand = make_difference_plot(description, time_one, time_two, data_one, data_two, \
    ...     name_one=name_one, name_two=name_two, elements=elements, units=units, \
    ...     start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     time_units=time_units, disp_xmax=disp_xmax, make_subplots=make_subplots, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, show_extra=show_extra, \
    ...     second_units=second_units, ylabel=ylabel, data_as_rows=data_as_rows, \
    ...     tolerance=tolerance, return_err=return_err, use_zoh=use_zoh, \
    ...     label_vert_lines=label_vert_lines, extra_plotter=extra_plotter)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    return make_generic_plot('diff', description=description, time_one=time_one, data_one=data_one, \
        time_two=time_two, data_two=data_two, name_one=name_one, name_two=name_two, \
        elements=elements, units=units, time_units=time_units, start_date=start_date, \
        rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
        single_lines=single_lines, make_subplots=make_subplots, colormap=colormap, use_mean=use_mean, \
        plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, show_extra=show_extra, \
        second_units=second_units, ylabel=ylabel, tolerance=tolerance, return_err=return_err, \
        data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
        label_vert_lines=label_vert_lines)

#%% Functions - make_categories_plot
def make_categories_plot(description, time, data, cats, *, cat_names=None, name='', elements=None, \
        units='', time_units='sec', start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, \
        disp_xmax=inf, make_subplots=True, single_lines=False, colormap=DEFAULT_COLORMAP, \
        use_mean=False, plot_zero=False, show_rms=True, legend_loc='best', second_units=None, \
        ylabel=None, data_as_rows=True, use_zoh=False, label_vert_lines=True, extra_plotter=None):
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
    second_units : str or tuple of (str, float), optional
        Name and conversion factor to use for scaling data to a second Y axis and in legend
    ylabel : str, optional
        Labels to put on the Y axes, potentially by element
    data_as_rows : bool, optional, default is True
        Whether the data has each channel as a row vector when 2D, vs a column vector
    use_zoh : bool, optional, default is False
        Whether to plot as a zero-order hold, instead of linear interpolation between data points
    label_vert_lines : bool, optional, default is True
        Whether to label the RMS start/stop lines in the legend (if legend is shown)
    extra_plotter : callable, optional
        Extra callable plotting function to add more details to the plot

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
    >>> second_units     = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> figs = make_categories_plot(description, time, data, cats, cat_names=cat_names, name=name, \
    ...     elements=elements, units=units, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     make_subplots=make_subplots, single_lines=single_lines, colormap=colormap, \
    ...     use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     second_units=second_units, ylabel=ylabel, data_as_rows=data_as_rows, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines, extra_plotter=extra_plotter)

    Close plots
    >>> import matplotlib.pyplot as plt
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
    (new_units, unit_conv) = get_unit_conversion(second_units, units)
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
                value = _LEG_FORMAT.format(unit_conv*data_func[cat][ix_data])
                if new_units:
                    this_label = f'{root_label} {this_cat_name} ({func_name}: {value} {new_units})'
                else:
                    this_label = f'{root_label} {this_cat_name} ({func_name}: {value})'
            else:
                this_label = f'{root_label} {this_cat_name}'
            this_cats = cats == cat
            this_linestyle = '-' if single_lines else 'none'
            # Note: Use len(cat_keys) here instead of num_cats so that potentially missing categories
            # won't mess up the color scheme by skipping colors
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
        plot_second_units_wrapper(this_axes, (new_units, unit_conv))
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

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        for fig in figs:
            extra_plotter(fig=fig, ax=fig.axes)

    return figs

#%% Functions - make_bar_plot
def make_bar_plot(description, time, data, *, name='', elements=None, units='', time_units='sec', \
        start_date='', rms_xmin=-inf, rms_xmax=inf, disp_xmin=-inf, disp_xmax=inf, \
        single_lines=False, colormap=DEFAULT_COLORMAP, use_mean=True, plot_zero=False, \
        show_rms=True, ignore_empties=False, legend_loc='best', second_units=None, ylabel=None, \
        data_as_rows=True, extra_plotter=None, use_zoh=False, label_vert_lines=True):
    r"""
    Plots a filled bar chart, using methods optimized for larger data sets.

    See make_generic_plot for input details.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

    See Also
    --------
    make_generic_plot

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import make_bar_plot
    >>> import numpy as np
    >>> description      = 'Test vs Time'
    >>> time             = np.arange(0, 5, 1./12) + 2000
    >>> data             = np.random.rand(5, len(time))
    >>> mag              = np.sum(data, axis=0)
    >>> data             = 100 * data / mag
    >>> name             = ''
    >>> elements         = None
    >>> units            = '%'
    >>> time_units       = 'sec'
    >>> start_date       = ''
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = 'Paired'
    >>> use_mean         = True
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> ignore_empties   = False
    >>> legend_loc       = 'best'
    >>> second_units     = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig = make_bar_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
    ...     disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     ignore_empties=ignore_empties, legend_loc=legend_loc, second_units=second_units, \
    ...     ylabel=ylabel, data_as_rows=data_as_rows, extra_plotter=extra_plotter, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines)

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    return make_generic_plot('bar', description=description, time_one=time, data_one=data, \
        name_one=name, elements=elements, units=units, time_units=time_units, start_date=start_date, \
        rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
        single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
        show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, ylabel=ylabel, \
        data_as_rows=data_as_rows, extra_plotter=extra_plotter, use_zoh=use_zoh, \
        label_vert_lines=label_vert_lines)

#%% make_connected_sets
def make_connected_sets(description, points, innovs, *, color_by='none', center_origin=False, \
        legend_loc='best', units='', mag_ratio=None, leg_scale='unity', colormap=None):
    r"""
    Plots two sets of X-Y pairs, with lines drawn between them.

    Parameters
    ----------
    description : str
        Plot description
    points : (2, N) ndarray
        Focal plane sightings
    innovs : (2, N) ndarray
        Innovations (implied to be in focal plane frame)
    color_by : str
        How to color the innovations, 'none' for same calor, 'magnitude' to color by innovation
        magnitude, or 'direction' to color by polar direction
    center_origin : bool, optional, default is False
        Whether to center the origin in the plot
    legend_loc : str, optional, default is 'best'
        Location of the legend in the plot
    units : str, optional
        Units to label on the plot
    mag_ratio : float, optional
        Percentage highest innovation magnitude to use, typically 0.95-1.0, but lets you exclude
        outliers that otherwise make the colorbar less useful
    leg_scale : str, optional, default is 'micro'
        Amount to scale the colorbar legend
    colormap : str, optional
        Name to use instead of the default colormaps, which depend on the mode

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle

    Examples
    --------
    >>> from dstauffman.plotting import make_connected_sets
    >>> import numpy as np
    >>> description = 'Focal Plane Sightings'
    >>> points = np.array([[0.1, 0.6, 0.7], [1.1, 1.6, 1.7]])
    >>> innovs = 5*np.array([[0.01, 0.02, 0.03], [-0.01, -0.015, -0.01]])
    >>> fig = make_connected_sets(description, points, innovs)

    >>> points2 = 2 * np.random.rand(2, 100) - 1.
    >>> innovs2 = 0.1 * np.random.randn(*points2.shape)
    >>> fig2 = make_connected_sets(description, points2, innovs2, color_by='direction')

    >>> fig3 = make_connected_sets(description, points2, innovs2, color_by='magnitude', \
    ...     leg_scale='milli', units='m')

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    >>> plt.close(fig2)
    >>> plt.close(fig3)

    """
    # hard-coded defaults
    colors_meas = 'xkcd:black'
    if color_by == 'none':
        colors_line = 'xkcd:red'
        colors_pred = 'xkcd:blue' if colormap is None else colormap
        extra_text  = ''
    elif color_by == 'direction':
        polar_ang   = RAD2DEG * np.arctan2(innovs[1, :], innovs[0, :])
        innov_cmap  = ColorMap('hsv' if colormap is None else colormap, low=-180, high=180)  # hsv or twilight?
        colors_line = tuple(innov_cmap.get_color(x) for x in polar_ang)
        colors_pred = colors_line
        extra_text  = ' (Colored by Direction)'
    elif color_by == 'magnitude':
        (new_units, unit_conv) = get_unit_conversion(leg_scale, units)
        innov_mags  = unit_conv * np.sqrt(np.sum(innovs**2, axis=0))
        if mag_ratio is None:
            max_innov = np.max(innov_mags)
        else:
            sorted_innovs = np.sort(innov_mags)
            max_innov = sorted_innovs[int(np.ceil(mag_ratio * innov_mags.size)) - 1]
        innov_cmap  = ColorMap(colormap='autumn_r' if colormap is None else colormap, low=0, high=max_innov)
        colors_line = tuple(innov_cmap.get_color(x) for x in innov_mags)
        colors_pred = colors_line
        extra_text  = ' (Colored by Magnitude)'
    else:
        raise ValueError(f'Unexpected value for color_by of "{color_by}"')

    # calculations
    predicts = points - innovs

    # create figure
    fig = plt.figure()
    fig.canvas.set_window_title(description + extra_text)
    ax = fig.add_subplot(1, 1, 1)
    # plot endpoints
    ax.plot(points[0, :], points[1, :], '.', color=colors_meas, label='Sighting', zorder=5)
    ax.scatter(predicts[0, :], predicts[1, :], c=colors_pred, marker='.', label='Predicted', zorder=8)
    # create fake line to add to legend
    ax.plot(np.nan, np.nan, '-', color='xkcd:black', label='Innov')
    if color_by != 'none':
        cbar = fig.colorbar(innov_cmap.get_smap())
        cbar_units = DEGREE_SIGN if color_by == 'direction' else new_units
        cbar.ax.set_ylabel('Innovation ' + color_by.capitalize() + ' [' + cbar_units + ']')
    # create segments
    segments = np.zeros((points.shape[1], 2, 2))
    segments[:, 0, :] = points.T
    segments[:, 1, :] = predicts.T
    lines = LineCollection(segments, colors=colors_line, zorder=3)
    ax.add_collection(lines)
    ax.set_title(description + extra_text)
    ax.legend(loc=legend_loc)
    ax.set_xlabel('FP X Loc [' + units + ']')  # TODO: pass in X,Y labels
    ax.set_ylabel('FP Y Loc [' + units + ']')
    ax.grid(True)
    if center_origin:
        xlims = np.max(np.abs(ax.get_xlim()))
        ylims = np.max(np.abs(ax.get_ylim()))
        ax.set_xlim(-xlims, xlims)
        ax.set_ylim(-ylims, ylims)
    ax.set_aspect('equal', 'box')

    return fig

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting_generic', exit=False)
    doctest.testmod(verbose=False)
