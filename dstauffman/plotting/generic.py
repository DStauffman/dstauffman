r"""
Defines low-level plotting routines meant to be wrapped in higher level ones.

Notes
-----
#.  Written by David C. Stauffer in May 2020.

"""  # pylint: disable=too-many-lines

# %% Imports
from __future__ import annotations

import datetime
import doctest
from itertools import repeat
import logging
from typing import Any, Callable, Iterable, Literal, overload, TYPE_CHECKING
import unittest

from slog import LogLevel

from dstauffman import (
    DEGREE_SIGN,
    get_unit_conversion,
    HAVE_DS,
    HAVE_MPL,
    HAVE_NUMPY,
    HAVE_PANDAS,
    intersect,
    is_datetime,
    RAD2DEG,
    rms,
)
from dstauffman.aerospace import quat_angle_diff
from dstauffman.plotting.support import (
    add_datashaders,
    ColorMap,
    COLOR_LISTS,
    DEFAULT_COLORMAP,
    disp_xlimits,
    ExtraPlotter,
    fig_ax_factory,
    get_rms_indices,
    ignore_plot_data,
    plot_second_units_wrapper,
    plot_vert_lines,
    show_zero_ylim,
    zoom_ylim,
)

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Colormap, ListedColormap
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

    inf = np.inf
else:
    from math import inf
if HAVE_PANDAS:
    import pandas as pd

# %% Constants
# hard-coded values
_LEG_FORMAT = "{:1.3f}"
_TRUTH_COLOR = "k"

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _M = NDArray[np.floating]  # 2D
    _Q = NDArray[np.floating]
    _CM = str | Colormap | ListedColormap | ColorMap
    _Data = int | float | np.floating | _I | _N | _M | list[_I] | list[_N] | list[_I | _N] | tuple[_I, ...] | tuple[_N, ...] | tuple[_I | _N, ...]  # fmt: skip
    _Time = int | float | np.floating | datetime.datetime | datetime.date | np.datetime64 | None
    _Times = int | float | np.floating | datetime.datetime | np.datetime64 | _D | _I | _N | list[_N] | list[_D] | tuple[_N, ...] | tuple[_D, ...]  # fmt: skip
    _DeltaTime = int | float | np.floating | np.timedelta64
    _Figs = list[Figure]
    _FuncLamb = Callable[[_I | _N, int | None], np.floating | _N]
    _SecUnits = str | int | float | np.floating | tuple[str, float] | tuple[str, np.floating] | None

# %% Globals
logger = logging.getLogger(__name__)


# %% Functions - _is_a_list
def _is_a_list(time: _Times | None, data: _Data | None) -> tuple[bool, bool]:
    """Determine if the inputs are lists/tuples or not."""
    time_is_list = isinstance(time, (list, tuple))
    data_is_list = isinstance(data, (list, tuple))
    if not time_is_list and time is not None:
        time = np.atleast_1d(time)  # type: ignore[arg-type, assignment]
    if not data_is_list and data is not None:
        data = np.atleast_2d(data)
        if data.ndim >= 3:
            raise AssertionError("data_one must be 0d, 1d or 2d.")
    return (time_is_list, data_is_list)


# %% Functions - _check_sizes
def _check_sizes(time, data, time_is_list, data_is_list, data_as_rows, *, is_quat: bool = False, num_channels: int | None = None):
    # calculate sizes
    s0 = 0 if time is None else len(time) if time_is_list else 1  # type: ignore[arg-type]
    if data is None:
        s1 = 0
    elif data_is_list:
        s1 = len(data)  # type: ignore[arg-type]
    elif data_as_rows:
        s1 = data.shape[0]  # type: ignore[union-attr]
    else:
        s1 = data.shape[1]  # type: ignore[union-attr]
        if is_quat:
            assert data.shape[0] == 4  # type: ignore[union-attr]
    # optional inputs
    if num_channels is None:
        num_channels = s1
    elif num_channels != s1:
        raise AssertionError(f"The given elements need to match the data sizes, got {num_channels} and {s1}.")
    if s0 not in (0, 1, num_channels):
        raise AssertionError("The time doesn't match the number of elements.")
    if is_quat:
        if s1 not in (0, 4):
            raise AssertionError("Must be a 4-element quaternion")

    return num_channels

# %% Functions - _build_indices
def _build_indices(time_is_list, data_is_list, time, num_channels, rms_xmin, rms_xmax) -> dict[str, list[int] | _B | None]:
    if data_is_list:
        ix: dict[str, list[int] | _B | None] = {"one": [], "t_min": None, "t_max": None}
        for j in range(num_channels):
            if time_is_list:
                temp_ix = get_rms_indices(time[j], xmin=rms_xmin, xmax=rms_xmax)  # type: ignore[index]
            else:
                temp_ix = get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)
            ix["one"].append(temp_ix["one"])  # type: ignore[arg-type, union-attr]
            if j == 0:
                ix["pts"] = temp_ix["pts"]
            else:
                ix["pts"] = [min((ix["pts"][0], temp_ix["pts"][0])), max((ix["pts"][1], temp_ix["pts"][1]))]  # type: ignore[index]
        return ix
    return get_rms_indices(time, xmin=rms_xmin, xmax=rms_xmax)


# %% Functions - _calc_rms
def _calc_rms(
    data, ix, *, num_channels: int, use_mean: bool, data_is_list: bool, data_as_rows: bool
) -> tuple[list[np.floating] | np.floating | _N, str]:
    """Calculates the RMS/mean."""

    # possible RMS/mean functions
    def _nan_rms(x: _I | _N, axis: int | None) -> np.floating | _N:
        """Calculate the RMS while ignoring NaNs."""
        try:
            return rms(x, axis=axis, ignore_nans=True)
        except TypeError:
            return np.array(np.nan)

    def _nan_mean(x: _I | _N, axis: int | None) -> np.floating | _N:
        """Calculate the mean while ignoring NaNs."""
        try:
            return np.nanmean(x, axis=axis)
        except TypeError:
            return np.array(np.nan)

    func_lamb: _FuncLamb
    data_func: list[np.floating] | np.floating | _N
    if not use_mean:
        func_name = "RMS"
        func_lamb = _nan_rms
    else:
        func_name = "Mean"
        func_lamb = _nan_mean
    if data_is_list:
        data_func = [func_lamb(data[j][ix[j]], None) for j in range(num_channels)]  # type: ignore[misc, index]
    elif data_as_rows:
        data_func = func_lamb(data[:, ix], 1) if np.any(ix) else np.full(num_channels, np.nan)  # type: ignore[assignment, call-overload, index]
    else:
        data_func = func_lamb(data[ix, :], 1) if np.any(ix) else np.full(num_channels, np.nan)  # type: ignore[assignment, call-overload, index]
    return data_func, func_name


# %% Functions - _get_units
def _get_units(units, second_units, leg_scale):
    """Get all the unit conversions."""
    (new_units, unit_conv) = get_unit_conversion(second_units, units)
    if leg_scale is not None:
        (leg_units, leg_conv) = get_unit_conversion(leg_scale, units)
    else:
        leg_units = new_units
        leg_conv = unit_conv
    return (new_units, unit_conv, leg_units, leg_conv)


# %% Functions - _get_ylabels
def _get_ylabels(
    num_channels: int, ylabel: list[str] | str | None, elements: list[str], *, single_lines: bool, description: str, units: str
) -> list[str]:
    if ylabel is None:
        if single_lines:
            ylabels = [f"{elements[i]} [{units}]" for i in range(num_channels)]
        else:
            ylabels = [""] * (num_channels - 1) + [f"{description} [{units}]"]
    else:
        if isinstance(ylabels, list):
            ylabels = ylabel
        else:
            ylabels = [ylabel] * num_channels if single_lines else [""] * (num_channels - 1) + [ylabel]
    return ylabels


# %% Functions - _create_figure
def _create_figure(
    num_figs: int, num_rows: int, num_cols: int, *, description: str = ""
) -> tuple[tuple[Figure, Axes], ...]:
    """Create or passthrough the given figures."""
    # % Create plots
    if num_cols == 1:
        fig_ax = fig_ax_factory(num_figs=num_figs, num_axes=num_rows, layout="rows", sharex=True)
    elif num_rows == 1:
        fig_ax = fig_ax_factory(num_figs=num_figs, num_axes=num_cols, layout="cols", sharex=True)
    else:
        layout = "colwise"  # TODO: colwise or rowwise?
        fig_ax = fig_ax_factory(num_figs=num_figs, num_axes=[num_rows, num_cols], layout=layout, sharex=True)  # type: ignore[call-overload]
    if description:
        fig_ax[0][0].canvas.manager.set_window_title(description)
    return fig_ax


# %% Functions - _plot_linear
def _plot_linear(ax: Axes, time: _Times | None, data: _Data | None, symbol: str, *args: Any, **kwargs: Any) -> None:
    """Plots a normal linear plot with passthrough options."""
    if len(args) != 0:
        raise AssertionError("Unexpected positional arguments.")
    assert time is not None
    assert data is not None
    try:
        if np.all(np.isnan(data)):
            return
    except TypeError:
        pass  # like categorical data that cannot be safely coerced to NaNs
    ax.plot(time, data, symbol, markerfacecolor="none", **kwargs)  # type: ignore[arg-type]


# %% Functions - _plot_zoh
def _plot_zoh(ax: Axes, time: _Times | None, data: _Data | None, symbol: str, *args: Any, **kwargs: Any) -> None:
    """Plots a zero-order hold step plot with passthrough options."""
    if len(args) != 0:
        raise AssertionError("Unexpected positional arguments.")
    assert time is not None
    assert data is not None
    try:
        if np.all(np.isnan(data)):
            return
    except TypeError:
        pass  # like categorical data that cannot be safely coerced to NaNs
    ax.step(time, data, symbol, where="post", markerfacecolor="none", **kwargs)  # type: ignore[arg-type]


# %% Functions - _label_x
def _label_x(this_axes, xlim, disp_xmin, disp_xmax, time_is_date, time_units, start_date):
    if xlim is None:
        disp_xlimits(this_axes, xmin=disp_xmin, xmax=disp_xmax)
        xlim = this_axes.get_xlim()
    if time_is_date:  # type: ignore[arg-type, index]
        this_axes.set_xlabel("Date")
        assert time_units in {"datetime", "numpy"}, f'Expected time units of "datetime" or "numpy", not "{time_units}".'
    else:
        this_axes.set_xlabel(f"Time [{time_units}]{start_date}")
    return xlim


# %% Functions - _get_diff_flags
def _get_diff_flags(
    time_one, time_two, data_one, data_two, *, time_is_list, tim2_is_list, data_is_list, dat2_is_list, data_as_rows
) -> tuple[bool, bool, bool, _N, _N]:
    """Determine which data you have, and do some consistency checks."""
    have_data_one = data_one is not None and np.any(~np.isnan(data_one))
    have_data_two = data_two is not None and np.any(~np.isnan(data_two))
    have_both = have_data_one and have_data_two
    if have_data_one:
        assert not data_is_list
        # TODO: remove this following restriction
        assert data_one.ndim == 2, f"Data must be 2D, not {data_one.ndim}"  # type: ignore[union-attr]
    if have_data_two:
        assert not dat2_is_list
        # TODO: remove this following restriction
        assert data_two.ndim == 2, f"Data must be 2D, not {data_two.ndim}"  # type: ignore[union-attr]
    # convert rows/cols as necessary
    if not data_as_rows:
        # TODO: is this the best way or make branches lower?
        if have_data_one:
            data_one = data_one.T  # type: ignore[union-attr]
        if have_data_two:
            data_two = data_two.T  # type: ignore[union-attr]
    if not time_is_list and time_one is not None:
        time_one = np.atleast_1d(time_one)  # type: ignore[arg-type, assignment]
    if not tim2_is_list and time_two is not None:
        time_two = np.atleast_1d(time_two)  # type: ignore[arg-type, assignment]
    if not data_is_list and data_one is not None:
        data_one = np.atleast_2d(data_one)
        if data_one.ndim >= 3:
            raise AssertionError("data_one must be 0d, 1d or 2d.")
    if not dat2_is_list and data_two is not None:
        data_two = np.atleast_2d(data_two)
        if data_two.ndim >= 3:
            raise AssertionError("data_two must be 0d, 1d or 2d.")
    return (have_data_one, have_data_two, have_both, data_one, data_two)


# %% Functions - make_time_plot
def make_time_plot(
    description: str,
    time: _Times | None,
    data: _Data | None,
    *,
    name: str = "",
    elements: list[str] | tuple[str, ...] | None = None,
    units: str = "",
    time_units: str = "sec",
    start_date: str = "",
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    single_lines: bool = False,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool = False,
    plot_zero: bool = False,
    show_rms: bool = True,
    ignore_empties: bool = False,
    legend_loc: str = "best",
    second_units: _SecUnits = None,
    leg_scale: _SecUnits = None,
    ylabel: str | list[str] | None = None,
    ylims: tuple[int, int] | tuple[float, float] | None = None,
    data_as_rows: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    use_datashader: bool = False,
    fig_ax: tuple[Figure, Axes] | None = None,
    plot_type: str = "time",  # {"time", "scatter"}
) -> Figure:
    r"""
    Generic data versus time plotting routine.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

    Examples
    --------
    >>> from dstauffman.plotting import make_time_plot
    >>> import numpy as np
    >>> description      = "Values vs Time"
    >>> time             = np.arange(-10., 10.1, 0.1)
    >>> data             = time + np.cos(time)
    >>> name             = ""
    >>> elements         = None
    >>> units            = ""
    >>> time_units       = "sec"
    >>> start_date       = ""
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = "Paired"
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> ignore_empties   = False
    >>> legend_loc       = "best"
    >>> second_units     = None
    >>> leg_scale        = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> use_datashader   = False
    >>> fig_ax           = None
    >>> fig = make_time_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
    ...     disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     ignore_empties=ignore_empties, legend_loc=legend_loc, second_units=second_units, \
    ...     leg_scale=leg_scale, ylabel=ylabel, data_as_rows=data_as_rows, \
    ...     extra_plotter=extra_plotter, use_zoh=use_zoh, label_vert_lines=label_vert_lines, \
    ...     use_datashader=use_datashader, fig_ax=fig_ax)

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    # hard-coded values
    return_err = False  # TODO: remove this restriction
    # get information on inputs
    time_is_list, data_is_list = _is_a_list(time, data)
    time_is_date = (time_is_list and len(time) > 0 and is_datetime(time[0]))

    # check for valid data
    # TODO: implement this
    if ignore_plot_data(data, ignore_empties):
        raise NotImplementedError("Not yet implemented.")

    # force 2D data
    if not data_is_list:
        data = np.atleast_2d(data)
        if data.ndim >= 3:
            raise AssertionError("data_one must be 0d, 1d or 2d.")

    # check sizing information
    num_channels = _check_sizes(time, data, time_is_list, data_is_list, data_as_rows, num_channels=None if elements is None else len(elements))

    # optional inputs
    if elements is None:
        elements = [f"Channel {i + 1}" for i in range(num_channels)]

    # build RMS indices
    if show_rms or return_err:
        ix = _build_indices(time_is_list, data_is_list, time, num_channels, rms_xmin, rms_xmax)

    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=num_channels)

    # calculate the rms (or mean) values
    if show_rms or return_err:
        data_func, func_name = _calc_rms(data, ix["one"], num_channels=num_channels, use_mean=use_mean, data_is_list=data_is_list, data_as_rows=data_as_rows)

    # unit conversion value
    (new_units, unit_conv, leg_units, leg_conv) = _get_units(units, second_units, leg_scale)

    # plotting options
    plot_func = _plot_zoh if use_zoh else _plot_linear
    if plot_type == "time":
        symbol = ".-"
    elif plot_type == "scatter":
        symbol = "."
    else:
        raise ValueError(f"Unexpected plot_type of {plot_type}.")

    # build data
    times = time if time_is_list else repeat(time, num_channels)
    if data_is_list:
        datum = data
    elif data_as_rows:
        datum = [data[i, :] for i in range(num_channels)]
    else:
        datum = [data[:, i] for i in range(num_channels)]

    ylabels = _get_ylabels(num_channels, ylabel, elements=elements, single_lines=single_lines, description=description, units=units)

    if fig_ax is None:
        # get the number of figures and axes to make
        num_figs = 1
        num_rows = num_channels if single_lines else 1
        num_cols = 1
        fig_ax = _create_figure(num_figs, num_rows, num_cols, description=description)
        if not single_lines:
            fig_ax = fig_ax * num_channels
    assert len(fig_ax) == num_channels, "Expecting a (figure, axes) pair for each channel in data."
    fig = fig_ax[0][0]  # type: ignore[index]
    ax = [f_a[1] for f_a in fig_ax] if single_lines else [fig_ax[0][1]]

    xlim: tuple[float, float] | None = None
    for i, ((this_fig, this_axes), this_time, this_data, this_ylabel),  in enumerate(zip(fig_ax, times, datum, ylabels)):
        this_label = str(elements[i])
        if show_rms:
            value = _LEG_FORMAT.format(leg_conv * data_func[i])  # type: ignore[index, operator]
            if leg_units:
                this_label += f" ({func_name}: {value} {leg_units})"
            else:
                this_label += f" ({func_name}: {value})"
        this_color = cm.get_color(i)
        plot_func(
            this_axes,
            this_time,
            this_data,
            symbol,
            markersize=4,
            label=this_label,
            color=this_color,
            zorder=9,
        )
        xlim = _label_x(this_axes, xlim, disp_xmin, disp_xmax, time_is_date, time_units, start_date)
        zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        if plot_zero:
            show_zero_ylim(this_axes)
        if ylims is not None:
            this_axes.set_ylims(ylims)
        if i == 0:
            this_axes.set_title(description)
        if bool(this_ylabel):
            this_axes.set_ylabel(this_ylabel)
            this_axes.grid(True)
            # optionally add second Y axis
            plot_second_units_wrapper(this_axes, (new_units, unit_conv))
            # plot RMS lines
            if show_rms:
                vert_labels = None if not use_mean else ["Mean Start Time", "Mean Stop Time"]
                plot_vert_lines(this_axes, ix["pts"], show_in_legend=label_vert_lines, labels=vert_labels)  # type: ignore[arg-type]

    if single_lines:
        fig.supylabel(description)

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        extra_plotter(fig=fig, ax=ax)

    # add legend at the very end once everything has been done
    if legend_loc.lower() != "none":
        for this_axes in ax:
            this_axes.legend(loc=legend_loc)

    return fig  # type: ignore[no-any-return]


# %% Functions - make_difference_plot
@overload
def make_difference_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    data_one: _Data | None,
    data_two: _Data | None,
    *,
    name_one: str,
    name_two: str,
    elements: list[str] | tuple[str, ...] | None,
    units: str,
    time_units: str,
    start_date: str,
    rms_xmin: _Time,
    rms_xmax: _Time,
    disp_xmin: _Time,
    disp_xmax: _Time,
    make_subplots: bool,
    single_lines: bool,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool,
    plot_zero: bool,
    show_rms: bool,
    legend_loc: str,
    show_extra: bool,
    second_units: _SecUnits,
    leg_scale: _SecUnits,
    ylabel: str | list[str] | None,
    ylims: tuple[int, int] | tuple[float, float] | None,
    data_as_rows: bool,
    tolerance: _DeltaTime,
    return_err: Literal[False] = ...,
    use_zoh: bool,
    label_vert_lines: bool,
    extra_plotter: ExtraPlotter | None,
    use_datashader: bool,
    fig_ax: tuple[Figure, Axes] | None,
) -> _Figs: ...
@overload
def make_difference_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    data_one: _Data | None,
    data_two: _Data | None,
    *,
    name_one: str,
    name_two: str,
    elements: list[str] | tuple[str, ...] | None,
    units: str,
    time_units: str,
    start_date: str,
    rms_xmin: _Time,
    rms_xmax: _Time,
    disp_xmin: _Time,
    disp_xmax: _Time,
    make_subplots: bool,
    single_lines: bool,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool,
    plot_zero: bool,
    show_rms: bool,
    legend_loc: str,
    show_extra: bool,
    second_units: _SecUnits,
    leg_scale: _SecUnits,
    ylabel: str | list[str] | None,
    ylims: tuple[int, int] | tuple[float, float] | None,
    data_as_rows: bool,
    tolerance: _DeltaTime,
    return_err: Literal[True],
    use_zoh: bool,
    label_vert_lines: bool,
    extra_plotter: ExtraPlotter | None,
    use_datashader: bool,
    fig_ax: tuple[Figure, Axes] | None,
) -> tuple[_Figs, dict[str, _N]]: ...
def make_difference_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    data_one: _Data | None,
    data_two: _Data | None,
    *,
    name_one: str = "",
    name_two: str = "",
    elements: list[str] | tuple[str, ...] | None = None,
    units: str = "",
    time_units: str = "sec",
    start_date: str = "",
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    make_subplots: bool = True,
    single_lines: bool = False,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool = False,
    plot_zero: bool = False,
    show_rms: bool = True,
    legend_loc: str = "best",
    show_extra: bool = True,
    second_units: _SecUnits = None,
    leg_scale: _SecUnits = None,
    ylabel: str | list[str] | None = None,
    ylims: tuple[int, int] | tuple[float, float] | None = None,
    data_as_rows: bool = True,
    tolerance: _DeltaTime = 0,
    return_err: bool = False,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_datashader: bool = False,
    fig_ax: tuple[Figure, Axes] | None = None,
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""
    Generic difference comparison plot for use in other wrapper functions.

    Plots two vector histories over time, along with a difference from one another.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle
    err : dict
        Differences

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
    >>> prng             = np.random.default_rng()
    >>> description      = "example"
    >>> time_one         = np.arange(11)
    >>> time_two         = np.arange(2, 13)
    >>> data_one         = 50e-6 * prng.random((2, 11))
    >>> data_two         = data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6 * prng.random((2, 11))
    >>> name_one         = "test1"
    >>> name_two         = "test2"
    >>> elements         = ["x", "y"]
    >>> units            = "rad"
    >>> time_units       = "sec"
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
    >>> legend_loc       = "best"
    >>> show_extra       = True
    >>> second_units     = ("µrad", 1e6)
    >>> leg_scale        = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> tolerance        = 0
    >>> return_err       = False
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> use_datashader   = False
    >>> fig_ax           = None
    >>> fig_hand = make_difference_plot(description, time_one, time_two, data_one, data_two, \
    ...     name_one=name_one, name_two=name_two, elements=elements, units=units, \
    ...     start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     time_units=time_units, disp_xmax=disp_xmax, make_subplots=make_subplots, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, show_extra=show_extra, \
    ...     second_units=second_units, leg_scale=leg_scale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, tolerance=tolerance, return_err=return_err, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines, extra_plotter=extra_plotter, \
    ...     use_datashader=use_datashader, fig_ax=fig_ax)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # get information on inputs
    time_is_list, data_is_list = _is_a_list(time_one, data_one)
    time_is_date = (time_is_list and len(time_one) > 0 and is_datetime(time_one[0]))
    tim2_is_list, dat2_is_list = _is_a_list(time_two, data_two)
    tim2_is_date = (time_is_list and len(time_two) > 0 and is_datetime(time_two[0]))
    assert not (time_is_list ^ tim2_is_list), "Both times must be lists if one is."
    if data_is_list or dat2_is_list:
        raise AssertionError("Data can't be lists for diffs right now.")  # TODO: remove this restriction
    have_data_one, have_data_two, have_both, data_one, data_two = _get_diff_flags(
        time_one,
        time_two,
        data_one,
        data_two,
        time_is_list=time_is_list,
        tim2_is_list=tim2_is_list,
        data_is_list=data_is_list,
        dat2_is_list=dat2_is_list,
        data_as_rows=data_as_rows,
    )

    # check for valid data
    if not have_data_one and not have_data_two:
        logger.log(LogLevel.L5, 'No differences data was provided, so no plot was generated for "%s".', description)
        if not return_err:
            return []
        # TODO: return NaNs instead of None for this case?
        out: tuple[_Figs, dict[str, float | None]] = ([], {"one": None, "two": None, "diff": None})
        return out  # type: ignore[return-value]

    # check sizing information
    num_channels = _check_sizes(time_one, data_two, time_is_list, data_is_list, data_as_rows, num_channels=None if elements is None else len(elements))
    num_channel2 = _check_sizes(time_two, data_two, tim2_is_list, dat2_is_list, data_as_rows, num_channels=None if elements is None else len(elements))
    if num_channels != num_channel2:
        raise AssertionError(f"The given elements need to match the data sizes, got {num_channels} and {num_channel2}.")

    # optional inputs
    if elements is None:
        elements = [f"Channel {i + 1}" for i in range(num_channels)]

    # build RMS indices
    if show_rms or return_err:
        if have_both:
            # find overlapping times
            (time_overlap, d1_diff_ix, d2_diff_ix) = intersect(time_one, time_two, tolerance=tolerance, return_indices=True)  # type: ignore[call-overload, misc]
            # find differences
            d1_miss_ix = np.setxor1d(np.arange(len(time_one)), d1_diff_ix)  # type: ignore[arg-type]
            d2_miss_ix = np.setxor1d(np.arange(len(time_two)), d2_diff_ix)  # type: ignore[arg-type]
            diffs = data_two[:, d2_diff_ix] - data_one[:, d1_diff_ix]  # type: ignore[call-overload, index]
        else:
            time_overlap = None
        ix = get_rms_indices(time_one, time_two, time_overlap, xmin=rms_xmin, xmax=rms_xmax)  # type: ignore[assignment]

    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=3 * num_channels)

    # calculate the rms (or mean) values
    if show_rms or return_err:
        data_func, func_name = _calc_rms(data_one, ix["one"], num_channels=num_channels, use_mean=use_mean, data_is_list=data_is_list, data_as_rows=data_as_rows)
        data2_func, _ = _calc_rms(data_two, ix["two"], num_channels=num_channels, use_mean=use_mean, data_is_list=dat2_is_list, data_as_rows=data_as_rows)
        if np.any(ix["overlap"]):
            nondeg_func, _ = _calc_rms(diffs, ix["overlap"], num_channels=num_channels, use_mean=use_mean, data_is_list=data_is_list, data_as_rows=data_as_rows)
        else:
            nondeg_func = np.full(num_channels, np.nan)
        # output errors
        err = {"one": data_func, "two": data2_func, "diff": nondeg_func}

    # unit conversion value
    (new_units, unit_conv, leg_units, leg_conv) = _get_units(units, second_units, leg_scale)

    # plotting options
    plot_func = _plot_zoh if use_zoh else _plot_linear

    # build data
    times1 = time_one if time_is_list else repeat(time_one, num_channels)
    times2 = time_two if time_is_list else repeat(time_two, num_channels)
    if data_is_list:
        datum1 = data_one
    elif data_as_rows:
        datum1 = [data_one[i, :] for i in range(num_channels)]
    else:
        datum1 = [data_one[:, i] for i in range(num_channels)]  # TODO: eliminates earlier transpose!!
    if dat2_is_list:
        datum2 = data_two
    elif data_as_rows:
        datum2 = [data_two[i, :] for i in range(num_channels)]
    else:
        datum2 = [data_two[:, i] for i in range(num_channels)]  # TODO: eliminates earlier transpose!!

    ylabels = _get_ylabels(num_channels, ylabel, elements=elements, single_lines=single_lines, description=description, units=units)


# %% Functions - make_quaternion_plot
@overload
def make_quaternion_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    name_one: str,
    name_two: str,
    time_units: str,
    start_date: str,
    plot_components: bool,
    rms_xmin: _Time,
    rms_xmax: _Time,
    disp_xmin: _Time,
    disp_xmax: _Time,
    make_subplots: bool,
    single_lines: bool,
    use_mean: bool,
    plot_zero: bool,
    show_rms: bool,
    legend_loc: str,
    show_extra: bool,
    second_units: _SecUnits,
    leg_scale: _SecUnits,
    data_as_rows: bool,
    tolerance: _DeltaTime,
    return_err: Literal[False] = ...,
    use_zoh: bool,
    label_vert_lines: bool,
    extra_plotter: ExtraPlotter | None,
    use_datashader: bool,
) -> _Figs: ...
@overload
def make_quaternion_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    name_one: str,
    name_two: str,
    time_units: str,
    start_date: str,
    plot_components: bool,
    rms_xmin: _Time,
    rms_xmax: _Time,
    disp_xmin: _Time,
    disp_xmax: _Time,
    make_subplots: bool,
    single_lines: bool,
    use_mean: bool,
    plot_zero: bool,
    show_rms: bool,
    legend_loc: str,
    show_extra: bool,
    second_units: _SecUnits,
    leg_scale: _SecUnits,
    data_as_rows: bool,
    tolerance: _DeltaTime,
    return_err: Literal[True],
    use_zoh: bool,
    label_vert_lines: bool,
    extra_plotter: ExtraPlotter | None,
    use_datashader: bool,
) -> tuple[_Figs, dict[str, _N]]: ...
def make_quaternion_plot(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    name_one: str = "",
    name_two: str = "",
    time_units: str = "sec",
    start_date: str = "",
    plot_components: bool = True,
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    make_subplots: bool = True,
    single_lines: bool = False,
    use_mean: bool = False,
    plot_zero: bool = False,
    show_rms: bool = True,
    legend_loc: str = "best",
    show_extra: bool = True,
    second_units: _SecUnits = "micro",
    leg_scale: _SecUnits = None,
    data_as_rows: bool = True,
    tolerance: _DeltaTime = 0,
    return_err: bool = False,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_datashader: bool = False,
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""
    Generic quaternion comparison plot for use in other wrapper functions.

    Plots two quaternion histories over time, along with a difference from one another.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle
    err : dict
        Differences

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in December 2018.
    #.  Made fully functional by David C. Stauffer in March 2019.
    #.  Wrapped to the generic do everything version by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import make_quaternion_plot
    >>> from dstauffman.aerospace import quat_norm
    >>> import numpy as np
    >>> from datetime import datetime
    >>> prng = np.random.default_rng()
    >>> description      = "example"
    >>> time_one         = np.arange(11)
    >>> time_two         = np.arange(2, 13)
    >>> quat_one         = quat_norm(prng.random((4, 11)))
    >>> quat_two         = quat_norm(quat_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] + 1e-5 * prng.random((4, 11)))
    >>> name_one         = "test1"
    >>> name_two         = "test2"
    >>> time_units       = "sec"
    >>> start_date       = str(datetime.now())
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = "best"
    >>> show_extra       = True
    >>> plot_components  = True
    >>> second_units     = ("µrad", 1e6)
    >>> leg_scale        = None
    >>> data_as_rows     = True
    >>> tolerance        = 0
    >>> return_err       = False
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> use_datashader   = False
    >>> fig_hand = make_quaternion_plot(description, time_one, time_two, quat_one, quat_two,
    ...     name_one=name_one, name_two=name_two, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     make_subplots=make_subplots, single_lines=single_lines, use_mean=use_mean, \
    ...     plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, show_extra=show_extra, \
    ...     plot_components=plot_components, second_units=second_units, leg_scale=leg_scale, \
    ...     data_as_rows=data_as_rows, tolerance=tolerance, return_err=return_err, use_zoh=use_zoh, \
    ...     label_vert_lines=label_vert_lines, extra_plotter=extra_plotter, use_datashader=use_datashader)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    colormap = ColorMap(COLOR_LISTS["quat_comp"])
    return None
    # return make_generic_plot(  # type: ignore[return-value]
    #     "quat",
    #     description=description,
    #     time_one=time_one,
    #     data_one=quat_one,
    #     time_two=time_two,
    #     data_two=quat_two,
    #     name_one=name_one,
    #     name_two=name_two,
    #     elements=("X", "Y", "Z", "S"),
    #     units="rad",
    #     time_units=time_units,
    #     start_date=start_date,
    #     rms_xmin=rms_xmin,
    #     rms_xmax=rms_xmax,
    #     disp_xmin=disp_xmin,
    #     disp_xmax=disp_xmax,
    #     single_lines=single_lines,
    #     make_subplots=make_subplots,
    #     colormap=colormap,
    #     use_mean=use_mean,
    #     plot_zero=plot_zero,
    #     show_rms=show_rms,
    #     legend_loc=legend_loc,
    #     show_extra=show_extra,
    #     plot_components=plot_components,
    #     second_units=second_units,
    #     leg_scale=leg_scale,
    #     tolerance=tolerance,
    #     return_err=return_err,
    #     data_as_rows=data_as_rows,
    #     extra_plotter=extra_plotter,
    #     use_zoh=use_zoh,
    #     label_vert_lines=label_vert_lines,
    #     use_datashader=use_datashader,
    # )


# %% Functions - make_error_bar_plot
def make_error_bar_plot(
    description: str,
    time: _Times | None,
    data: _Data | None,
    mins: _N | _M | None,
    maxs: _N | _M | None,
    *,
    elements: list[str] | tuple[str, ...] | None = None,
    units: str = "",
    time_units: str = "sec",
    start_date: str = "",
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    single_lines: bool = False,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool = False,
    plot_zero: bool = False,
    show_rms: bool = True,
    legend_loc: str = "best",
    second_units: _SecUnits = None,
    leg_scale: _SecUnits = None,
    ylabel: str | list[str] | None = None,
    ylims: tuple[int, int] | tuple[float, float] | None = None,
    data_as_rows: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    fig_ax: tuple[Figure, Axes] | None = None,
) -> Figure:
    r"""
    Generic plotting routine to make error bars.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

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
    >>> prng             = np.random.default_rng()
    >>> description      = "Random Data Error Bars"
    >>> time             = np.arange(11)
    >>> data             = np.array([[3.], [-2.], [5]]) + prng.random((3, 11))
    >>> mins             = data - 0.5 * prng.random((3, 11))
    >>> maxs             = data + 1.5 * prng.random((3, 11))
    >>> elements         = ["x", "y", "z"]
    >>> units            = "rad"
    >>> time_units       = "sec"
    >>> start_date       = "  t0 = " + str(datetime.now())
    >>> rms_xmin         = 1
    >>> rms_xmax         = 10
    >>> disp_xmin        = -2
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = "tab10"
    >>> use_mean         = False
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = "best"
    >>> second_units     = "milli"
    >>> leg_scale        = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig_ax           = None
    >>> fig              = make_error_bar_plot(description, time, data, mins, maxs, \
    ...     elements=elements, units=units, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     single_lines=single_lines, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, \
    ...     show_rms=show_rms, legend_loc=legend_loc, second_units=second_units, \
    ...     leg_scale=leg_scale, ylabel=ylabel, data_as_rows=data_as_rows, \
    ...     extra_plotter=extra_plotter, use_zoh=use_zoh, label_vert_lines=label_vert_lines, \
    ...     fig_ax=fig_ax)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    # hard-coded values
    return_err = False  # TODO: remove this restriction
    ignore_empties = False  # TODO: remove this restriction
    # get information on inputs
    time_is_list, data_is_list = _is_a_list(time, data)
    time_is_date = (time_is_list and len(time) > 0 and is_datetime(time[0]))

    # check for valid data
    # TODO: implement this
    if ignore_plot_data(data, ignore_empties):
        raise NotImplementedError("Not yet implemented.")

    # check sizing information
    num_channels = _check_sizes(time, data, time_is_list, data_is_list, data_as_rows, num_channels=None if elements is None else len(elements))

    # optional inputs
    if elements is None:
        elements = [f"Channel {i + 1}" for i in range(num_channels)]

    # build RMS indices
    if show_rms or return_err:
        ix = _build_indices(time_is_list, data_is_list, time, num_channels, rms_xmin, rms_xmax)

    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=num_channels)

    # calculate the rms (or mean) values
    if show_rms or return_err:
        data_func, func_name = _calc_rms(data, ix["one"], num_channels=num_channels, use_mean=use_mean, data_is_list=data_is_list, data_as_rows=data_as_rows)

    # unit conversion value
    (new_units, unit_conv, leg_units, leg_conv) = _get_units(units, second_units, leg_scale)

    # plotting options
    plot_func = _plot_zoh if use_zoh else _plot_linear
    symbol = ".-"

    # extra errorbar calculations
    assert not data_is_list  # TODO: handle data_is_list and rows cases
    err_neg = data - mins  # type: ignore[call-overload, operator]
    err_pos = maxs - data  # type: ignore[call-overload, operator]
    yerrs = [np.vstack([err_neg[i, :], err_pos[i, :]]) for i in range(num_channels)]
    times = time if time_is_list else repeat(time, num_channels)
    if data_is_list:
        datum = data
        yerrs = [np.vstack([d - mn, mx - d]) for d, mn, mx in zip(data, mins, maxs)]
    elif data_as_rows:
        err_neg = data - mins  # type: ignore[call-overload, operator]
        err_pos = maxs - data  # type: ignore[call-overload, operator]
        datum = [data[i, :] for i in range(num_channels)]
        yerrs = [np.vstack([err_neg[i, :], err_pos[i, :]]) for i in range(num_channels)]
    else:
        err_neg = data - mins  # type: ignore[call-overload, operator]
        err_pos = maxs - data  # type: ignore[call-overload, operator]
        datum = [data[:, i] for i in range(num_channels)]
        yerrs = [np.vstack([err_neg[:, i], err_pos[:, i]]) for i in range(num_channels)]

    ylabels = _get_ylabels(num_channels, ylabel, elements=elements, single_lines=single_lines, description=description, units=units)

    if fig_ax is None:
        # get the number of figures and axes to make
        num_figs = 1
        num_rows = num_channels if single_lines else 1
        num_cols = 1
        fig_ax = _create_figure(num_figs, num_rows, num_cols, description=description)
        if not single_lines:
            fig_ax = fig_ax * num_channels
    assert len(fig_ax) == num_channels, "Expecting a (figure, axes) pair for each channel in data."
    fig = fig_ax[0][0]  # type: ignore[index]
    ax = [f_a[1] for f_a in fig_ax] if single_lines else [fig_ax[0][1]]

    xlim: tuple[float, float] | None = None
    for i, ((this_fig, this_axes), this_time, this_data, this_yerr, this_ylabel),  in enumerate(zip(fig_ax, times, datum, yerrs, ylabels)):
        this_label = str(elements[i])
        if show_rms:
            value = _LEG_FORMAT.format(leg_conv * data_func[i])  # type: ignore[index, operator]
            if leg_units:
                this_label += f" ({func_name}: {value} {leg_units})"
            else:
                this_label += f" ({func_name}: {value})"
        this_color = cm.get_color(i)
        plot_func(
            this_axes,
            this_time,
            this_data,
            symbol,
            markersize=4,
            label=this_label,
            color=this_color,
            zorder=3,
        )
        # plot error bars
        this_axes.errorbar(
            this_time,
            this_data,
            yerr=this_yerr,
            color="None",
            ecolor=cm.get_color(i),
            zorder=5,
            capsize=2,
        )
        xlim = _label_x(this_axes, xlim, disp_xmin, disp_xmax, time_is_date, time_units, start_date)
        zoom_ylim(this_axes, t_start=xlim[0], t_final=xlim[1])
        if plot_zero:
            show_zero_ylim(this_axes)
        if ylims is not None:
            this_axes.set_ylims(ylims)
        if i == 0:
            this_axes.set_title(description)
        if bool(this_ylabel):
            this_axes.set_ylabel(this_ylabel)
            this_axes.grid(True)
            # optionally add second Y axis
            plot_second_units_wrapper(this_axes, (new_units, unit_conv))
            # plot RMS lines
            if show_rms:
                vert_labels = None if not use_mean else ["Mean Start Time", "Mean Stop Time"]
                plot_vert_lines(this_axes, ix["pts"], show_in_legend=label_vert_lines, labels=vert_labels)  # type: ignore[arg-type]

    if single_lines:
        fig.supylabel(description)

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        extra_plotter(fig=fig, ax=ax)

    # add legend at the very end once everything has been done
    if legend_loc.lower() != "none":
        for this_axes in ax:
            this_axes.legend(loc=legend_loc)

    return fig  # type: ignore[no-any-return]


# %% Functions - make_bar_plot
def make_bar_plot(
    description: str,
    time: _Times | None,
    data: _Data | None,
    *,
    name: str = "",
    elements: list[str] | tuple[str, ...] | None = None,
    units: str = "",
    time_units: str = "sec",
    start_date: str = "",
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    single_lines: bool = False,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool = True,
    plot_zero: bool = False,
    show_rms: bool = True,
    ignore_empties: bool = False,
    legend_loc: str = "best",
    second_units: _SecUnits = None,
    ylabel: str | list[str] | None = None,
    ylims: tuple[int, int] | tuple[float, float] | None = None,
    data_as_rows: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    fig_ax: tuple[Figure, Axes] | None = None,
) -> Figure:
    r"""
    Plots a filled bar chart, using methods optimized for larger data sets.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle

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
    >>> description      = "Test vs Time"
    >>> time             = np.arange(0, 5, 1./12) + 2000
    >>> data             = np.random.default_rng().random((5, len(time)))
    >>> mag              = np.sum(data, axis=0)
    >>> data             = 100 * data / mag
    >>> name             = ""
    >>> elements         = None
    >>> units            = "%"
    >>> time_units       = "sec"
    >>> start_date       = ""
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> single_lines     = False
    >>> colormap         = "Paired"
    >>> use_mean         = True
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> ignore_empties   = False
    >>> legend_loc       = "best"
    >>> second_units     = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> extra_plotter    = None
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> fig_ax           = None
    >>> fig = make_bar_plot(description, time, data, name=name, elements=elements, units=units, \
    ...     time_units=time_units, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, \
    ...     disp_xmin=disp_xmin, disp_xmax=disp_xmax, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, \
    ...     ignore_empties=ignore_empties, legend_loc=legend_loc, second_units=second_units, \
    ...     ylabel=ylabel, data_as_rows=data_as_rows, extra_plotter=extra_plotter, \
    ...     use_zoh=use_zoh, label_vert_lines=label_vert_lines, fig_ax=fig_ax)

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    # hard-coded values
    return_err = False  # TODO: remove this restriction
    ignore_empties = False  # TODO: remove this restriction
    leg_scale = ("%", 1.0)

    # get information on inputs
    time_is_list, data_is_list = _is_a_list(time, data)
    time_is_date = (time_is_list and len(time) > 0 and is_datetime(time[0]))

    # check for valid data
    # TODO: implement this
    if ignore_plot_data(data, ignore_empties):
        raise NotImplementedError("Not yet implemented.")
    if single_lines:
        raise ValueError("Bar plots are not valid with single_lines.")

    # check sizing information
    num_channels = _check_sizes(time, data, time_is_list, data_is_list, data_as_rows, num_channels=None if elements is None else len(elements))

    # optional inputs
    if elements is None:
        elements = [f"Channel {i + 1}" for i in range(num_channels)]

    # build RMS indices
    if show_rms or return_err:
        ix = _build_indices(time_is_list, data_is_list, time, num_channels, rms_xmin, rms_xmax)

    # create a colormap
    cm = ColorMap(colormap=colormap, num_colors=num_channels)

    # calculate the rms (or mean) values
    if show_rms or return_err:
        data_func, func_name = _calc_rms(data, ix["one"], num_channels=num_channels, use_mean=use_mean, data_is_list=data_is_list, data_as_rows=data_as_rows)

    # unit conversion value
    (new_units, unit_conv, leg_units, leg_conv) = _get_units(units, second_units, leg_scale)

    # extra bar calculations
    bottoms: list[_N] | _N | _M
    if data_is_list:
        bottoms = [np.cumsum(np.ma.masked_invalid(data[j])) for j in range(num_channels)]  # type: ignore[index, no-untyped-call]
    elif data_as_rows:
        bottoms = np.concatenate((np.zeros((1, len(time))), np.cumsum(np.ma.masked_invalid(data), axis=0)), axis=0)  # type: ignore[arg-type, no-untyped-call]
    else:
        bottoms = np.concatenate((np.zeros((len(time), 1)), np.cumsum(np.ma.masked_invalid(data), axis=1)), axis=1)  # type: ignore[arg-type, no-untyped-call]

    ylabels = _get_ylabels(num_channels, ylabel, elements=elements, single_lines=single_lines, description=description, units=units)
    if not single_lines:
        ylabels = ylabels[::-1]

    if fig_ax is None:
        # get the number of axes to make
        num_figs = 1
        num_rows = 1
        num_cols = 1
        fig_ax = _create_figure(num_figs, num_rows, num_cols, description=description)
        if not single_lines:
            fig_ax = fig_ax * num_channels
    assert len(fig_ax) == num_channels, "Expecting a (figure, axes) pair for each channel in data."
    fig = fig_ax[0][0]  # type: ignore[index]
    ax = [fig_ax[0][1]]

    xlim: tuple[float, float] | None = None
    for i in reversed(range(num_channels)):
        this_fig, this_axes = fig_ax[i]
        this_time = time[i] if time_is_list else time
        this_data = data[i] if data_is_list else data[i, :] if data_as_rows else data[:, i]  # type: ignore[call-overload, index]
        this_ylabel = ylabels[i]
        this_label = str(elements[i])
        if show_rms:
            value = _LEG_FORMAT.format(leg_conv * data_func[i])  # type: ignore[index, operator]
            if leg_units:
                this_label += f" ({func_name}: {value} {leg_units})"
            else:
                this_label += f" ({func_name}: {value})"
        this_bottom1 = bottoms[i] if data_is_list else bottoms[i, :] if data_as_rows else bottoms[:, i]  # type: ignore[call-overload]
        this_bottom2 = bottoms[i + 1] if data_is_list else bottoms[i + 1, :] if data_as_rows else bottoms[:, i + 1]  # type: ignore[call-overload]
        if not ignore_plot_data(this_data, ignore_empties):
            # Note: The performance of ax.bar is really slow with large numbers of bars (>20), so
            # fill_between is a better alternative
            this_axes.fill_between(
                this_time,
                this_bottom1,
                this_bottom2,
                step="mid",
                label=this_label,
                color=cm.get_color(i),
                edgecolor="none",
            )
        xlim = _label_x(this_axes, xlim, disp_xmin, disp_xmax, time_is_date, time_units, start_date)
        this_axes.set_ylim(0, 100)
        if plot_zero:
            show_zero_ylim(this_axes)
        if ylims is not None:
            this_axes.set_ylims(ylims)
        if i == num_channels - 1:
            this_axes.set_title(description)
        if bool(this_ylabel):
            this_axes.set_ylabel(this_ylabel)
            this_axes.grid(True)
            # optionally add second Y axis
            plot_second_units_wrapper(this_axes, (new_units, unit_conv))
            # plot RMS lines
            if show_rms:
                vert_labels = None if not use_mean else ["Mean Start Time", "Mean Stop Time"]
                plot_vert_lines(this_axes, ix["pts"], show_in_legend=label_vert_lines, labels=vert_labels)  # type: ignore[arg-type]

    if single_lines:
        fig.supylabel(description)

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        extra_plotter(fig=fig, ax=ax)

    # add legend at the very end once everything has been done
    if legend_loc.lower() != "none":
        for this_axes in ax:
            this_axes.legend(loc=legend_loc)

    return fig  # type: ignore[no-any-return]


# %% Functions - make_categories_plot
def make_categories_plot(
    description: str,
    time: _Times | None,
    data: _Data | None,
    cats: Iterable[Any] | None,
    *,
    cat_names: dict[Any, str] | None = None,
    name: str = "",
    elements: list[str] | tuple[str, ...] | None = None,
    units: str = "",
    time_units: str = "sec",
    start_date: str = "",
    rms_xmin: _Time = -inf,
    rms_xmax: _Time = inf,
    disp_xmin: _Time = -inf,
    disp_xmax: _Time = inf,
    make_subplots: bool = True,
    single_lines: bool = False,
    colormap: _CM | None = DEFAULT_COLORMAP,
    use_mean: bool = False,
    plot_zero: bool = False,
    show_rms: bool = True,
    legend_loc: str = "best",
    second_units: _SecUnits = None,
    leg_scale: _SecUnits = None,
    ylabel: str | list[str] | None = None,
    ylims: tuple[int, int] | tuple[float, float] | None = None,
    data_as_rows: bool = True,
    use_zoh: bool = False,
    label_vert_lines: bool = True,
    extra_plotter: ExtraPlotter | None = None,
    use_datashader: bool = False,
    fig_ax: tuple[Figure, Axes] | None = None,
) -> _Figs:
    r"""
    Data versus time plotting routine when grouped into categories.

    Returns
    -------
    figs : list of class matplotlib.Figure
        Figure handles

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.
    #.  Wrapped to the generic do everything version by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import make_categories_plot
    >>> import numpy as np
    >>> description      = "Values vs Time"
    >>> time             = np.arange(-10., 10.1, 0.1)
    >>> data             = np.vstack((time + np.cos(time), np.ones(time.shape)))
    >>> data[1, 60:85]   = 2
    >>> MeasStatus       = type("MeasStatus", (object,), {"rejected": 0, "accepted": 1})
    >>> cats             = np.full(time.shape, MeasStatus.accepted, dtype=int)
    >>> cats[50:100]     = MeasStatus.rejected
    >>> cat_names        = {0: "rejected", 1: "accepted"}
    >>> name             = ""
    >>> elements         = None
    >>> units            = ""
    >>> time_units       = "sec"
    >>> start_date       = ""
    >>> rms_xmin         = -np.inf
    >>> rms_xmax         = np.inf
    >>> disp_xmin        = -np.inf
    >>> disp_xmax        = np.inf
    >>> make_subplots    = True
    >>> single_lines     = False
    >>> colormap         = "Paired"
    >>> use_mean         = True
    >>> plot_zero        = False
    >>> show_rms         = True
    >>> legend_loc       = "best"
    >>> second_units     = None
    >>> leg_scale        = None
    >>> ylabel           = None
    >>> data_as_rows     = True
    >>> use_zoh          = False
    >>> label_vert_lines = True
    >>> extra_plotter    = None
    >>> use_datashader   = False
    >>> fig_ax           = None
    >>> figs = make_categories_plot(description, time, data, cats, cat_names=cat_names, name=name, \
    ...     elements=elements, units=units, time_units=time_units, start_date=start_date, \
    ...     rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
    ...     make_subplots=make_subplots, single_lines=single_lines, colormap=colormap, \
    ...     use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     second_units=second_units, leg_scale=leg_scale, ylabel=ylabel, \
    ...     data_as_rows=data_as_rows, use_zoh=use_zoh, label_vert_lines=label_vert_lines, \
    ...     extra_plotter=extra_plotter, use_datashader=use_datashader, fig_ax=fig_ax)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in figs:
    ...     plt.close(fig)

    """
    return None
    # return make_generic_plot(  # type: ignore[return-value]
    #     plot_type="cats",
    #     description=description,
    #     time_one=time,
    #     data_one=data,
    #     cats=cats,
    #     cat_names=cat_names,
    #     name_one=name,
    #     elements=elements,
    #     units=units,
    #     time_units=time_units,
    #     start_date=start_date,
    #     rms_xmin=rms_xmin,
    #     rms_xmax=rms_xmax,
    #     disp_xmin=disp_xmin,
    #     disp_xmax=disp_xmax,
    #     make_subplots=make_subplots,
    #     single_lines=single_lines,
    #     colormap=colormap,
    #     use_mean=use_mean,
    #     plot_zero=plot_zero,
    #     show_rms=show_rms,
    #     legend_loc=legend_loc,
    #     second_units=second_units,
    #     leg_scale=leg_scale,
    #     ylabel=ylabel,
    #     ylims=ylims,
    #     data_as_rows=data_as_rows,
    #     use_zoh=use_zoh,
    #     label_vert_lines=label_vert_lines,
    #     extra_plotter=extra_plotter,
    #     use_datashader=use_datashader,
    #     fig_ax=fig_ax,
    # )

# %% make_connected_sets
def make_connected_sets(  # noqa: C901
    description: str,
    points: _M,
    innovs: _M | None,
    *,
    color_by: str = "none",
    hide_innovs: bool = False,
    center_origin: bool = False,
    legend_loc: str = "best",
    units: str = "",
    mag_ratio: float | None = None,
    leg_scale: _SecUnits = "unity",
    colormap: str | ColorMap | None = None,
    use_datashader: bool = False,
    add_quiver: bool = False,
    quiver_scale: float | None = None,
    fig_ax: tuple[Figure, Axes] | None = None,
) -> Figure:
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
    hide_innovs : bool, optional, default is False
        Whether to hide the innovations and only show the sightings
    color_by : str
        How to color the innovations, "none" for same calor, "magnitude" to color by innovation
        magnitude, or "direction" to color by polar direction
    center_origin : bool, optional, default is False
        Whether to center the origin in the plot
    legend_loc : str, optional, default is "best"
        Location of the legend in the plot
    units : str, optional
        Units to label on the plot
    mag_ratio : float, optional
        Percentage highest innovation magnitude to use, typically 0.95-1.0, but lets you exclude
        outliers that otherwise make the colorbar less useful
    leg_scale : str, optional, default is "micro"
        Amount to scale the colorbar legend
    colormap : str, optional
        Name to use instead of the default colormaps, which depend on the mode
    add_quiver : bool, optional, default is False
        Whether to add matplotlib quiver lines to the plot
    quiver_scale : float, optional
        quiver line scale factor
    fig_ax : (fig, ax) tuple, optional
        Figure and axis to use, otherwise create new ones

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle

    Examples
    --------
    >>> from dstauffman.plotting import make_connected_sets
    >>> import numpy as np
    >>> description = "Focal Plane Sightings"
    >>> points = np.array([[0.1, 0.6, 0.7], [1.1, 1.6, 1.7]])
    >>> innovs = 5*np.array([[0.01, 0.02, 0.03], [-0.01, -0.015, -0.01]])
    >>> fig = make_connected_sets(description, points, innovs)

    >>> prng = np.random.default_rng()
    >>> points2 = 2 * prng.uniform(-1.0, 0.0, (2, 100))
    >>> innovs2 = 0.1 * prng.normal(size=points2.shape)
    >>> fig2 = make_connected_sets(description, points2, innovs2, color_by="direction")

    >>> fig3 = make_connected_sets(description, points2, innovs2, color_by="magnitude", \
    ...     leg_scale="milli", units="m")

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    >>> plt.close(fig2)
    >>> plt.close(fig3)

    """
    # hard-coded defaults
    datashader_pts = 2000  # Plot this many points on top of datashader plots, or skip if fewer exist
    colors_meas = "xkcd:black"
    null_options = {"none", "density"}

    # calculations
    if innovs is None:
        assert color_by in null_options, 'If no innovations are given, then you must color by "none" or "density".'
        plot_innovs = False
    elif hide_innovs:
        plot_innovs = False
    else:
        plot_innovs = True
        predicts = points - innovs
    datashaders = []

    # get index to subset of points for datashading
    if use_datashader:
        assert HAVE_PANDAS and HAVE_DS, "You must have pandas and datashader to run datashader plots."
        if points.shape[1] < datashader_pts:
            ix = np.arange(points.shape[1])
        else:
            ix = np.round(np.linspace(0, points.shape[1] - 1, datashader_pts // 10)).astype(int)
            # include the mins and maxes in both axes
            ix_xmin = np.argmin(points[0, :])
            ix_xmax = np.argmax(points[0, :])
            ix_ymin = np.argmin(points[1, :])
            ix_ymax = np.argmax(points[1, :])
            ix = np.union1d(ix, np.array([ix_xmin, ix_xmax, ix_ymin, ix_ymax]))
            if plot_innovs:
                ix_xmin = np.argmin(predicts[0, :])
                ix_xmax = np.argmax(predicts[0, :])
                ix_ymin = np.argmin(predicts[1, :])
                ix_ymax = np.argmax(predicts[1, :])
                ix = np.union1d(ix, np.array([ix_xmin, ix_xmax, ix_ymin, ix_ymax]))
    else:
        ix = np.arange(points.shape[1])

    # color options
    colors_line: str | ColorMap | tuple[Any, ...]
    colors_pred: str | ColorMap | tuple[Any, ...]
    ds_value: _N | None
    # fmt: off
    if color_by in null_options:
        colors_line = "xkcd:red"
        colors_pred = "xkcd:blue" if colormap is None else colormap
        if color_by == "none":
            extra_text = ""
            ds_value   = np.zeros(points.shape[1])
        else:
            extra_text = " (Colored by Density)"
            ds_value   = None
        ds_low      = None
        ds_high     = None
        ds_color    = "xkcd:blue"
    elif color_by == "direction":
        polar_ang   = RAD2DEG * np.arctan2(innovs[1, :], innovs[0, :])  # type: ignore[index]
        innov_cmap  = ColorMap("hsv" if colormap is None else colormap, low=-180, high=180)  # hsv or twilight?
        colors_line = tuple(innov_cmap.get_color(x) for x in polar_ang[ix])
        colors_pred = colors_line
        extra_text  = " (Colored by Direction)"
        ds_value    = polar_ang
        ds_low      = -180
        ds_high     = 180
        ds_color = "hsv" if not isinstance(colormap, str) else colormap
    elif color_by == "magnitude":
        (new_units, unit_conv) = get_unit_conversion(leg_scale, units)
        innov_mags = unit_conv * np.sqrt(np.sum(innovs**2, axis=0))  # type: ignore[operator]
        if mag_ratio is None:
            max_innov = np.max(innov_mags)
        else:
            sorted_innovs = np.sort(innov_mags)
            max_innov = sorted_innovs[int(np.ceil(mag_ratio * innov_mags.size)) - 1] if innov_mags.size > 0 else 0
        innov_cmap  = ColorMap(colormap="autumn_r" if colormap is None else colormap, low=0, high=max_innov)
        colors_line = tuple(innov_cmap.get_color(x) for x in innov_mags[ix])
        colors_pred = colors_line
        extra_text  = " (Colored by Magnitude)"
        ds_value    = innov_mags
        ds_low      = 0
        ds_high     = max_innov
        ds_color    = "autumn_r" if not isinstance(colormap, str) else colormap
    else:
        raise ValueError(f'Unexpected value for color_by of "{color_by}"')
    # fmt: on

    # create figure and axes (needs to be done before building datashader information)
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        (fig, ax) = fig_ax
    assert fig.canvas.manager is not None
    if (sup := fig._suptitle) is None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
        fig.canvas.manager.set_window_title(description + extra_text)
    else:
        fig.canvas.manager.set_window_title(sup.get_text())

    # build datashader information for use later
    color_key = "color" if ds_color.startswith("xkcd") else "colormap"
    if use_datashader and points.shape[1] >= datashader_pts:
        datashaders.append(
            {
                "time": points[0, :],
                "data": points[1, :],
                "ax": ax,
                color_key: ds_color,
                "vmin": ds_low,
                "vmax": ds_high,
                "value": ds_value,
                "norm": "eq_hist",
                "aspect": "equal",
            }
        )
        if plot_innovs:
            datashaders.append(
                {"time": predicts[0, :], "data": predicts[1, :], "ax": ax, "color": "xkcd:black", "aspect": "equal"}
            )

    # populate the normal plot, potentially with a subset of points
    if plot_innovs:
        ax.plot(points[0, ix], points[1, ix], ".", color=colors_meas, label="Sighting", zorder=5)
        ax.scatter(predicts[0, ix], predicts[1, ix], c=colors_pred, marker=".", label="Predicted", zorder=8)  # type: ignore[arg-type]
        # create fake line to add to legend
        line_leg_color = colors_line if isinstance(colors_line, str) else "xkcd:black"
        ax.plot(np.nan, np.nan, "-", color=line_leg_color, label="Innov")
        # create segments
        segments = np.zeros((ix.size, 2, 2))
        segments[:, 0, :] = points[:, ix].T
        segments[:, 1, :] = predicts[:, ix].T
        lines = LineCollection(segments, colors=colors_line, zorder=3)  # type: ignore[arg-type]
        ax.add_collection(lines)
    else:
        ax.scatter(points[0, ix], points[1, ix], c=colors_pred, marker=".", label="Sighting", zorder=5)  # type: ignore[arg-type]
    if add_quiver:
        ax.quiver(points[0, ix], points[1, ix], innovs[0, ix], innovs[1, ix], color="xkcd:black", units="x", scale=quiver_scale)  # type: ignore[index]
    if color_by not in null_options:
        cbar = fig.colorbar(innov_cmap.get_smap(), ax=ax, shrink=0.9)
        cbar_units = DEGREE_SIGN if color_by == "direction" else new_units
        cbar.ax.set_ylabel("Innovation " + color_by.capitalize() + " [" + cbar_units + "]")
    ax.set_title(description + extra_text)
    ax.set_xlabel("FP X Loc [" + units + "]")  # TODO: pass in X, Y labels
    ax.set_ylabel("FP Y Loc [" + units + "]")
    ax.grid(True)
    if center_origin:
        xlims = np.max(np.abs(ax.get_xlim()))
        ylims = np.max(np.abs(ax.get_ylim()))
        ax.set_xlim(-xlims, xlims)
        ax.set_ylim(-ylims, ylims)
    ax.set_aspect("equal", "box")

    if bool(datashaders):
        add_datashaders(datashaders)
    if legend_loc.lower() != "none":
        ax.legend(loc=legend_loc)

    return fig


# %% Unit test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_generic", exit=False)
    doctest.testmod(verbose=False)
