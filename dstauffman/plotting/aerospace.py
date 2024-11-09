r"""
Plots related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""  # pylint: disable=too-many-lines

# %% Imports
from __future__ import annotations

import datetime
import doctest
import logging
from pathlib import Path
from typing import Any, Literal, NotRequired, overload, TYPE_CHECKING, TypedDict, Unpack
import unittest

from slog import LogLevel

from dstauffman import HAVE_MPL, HAVE_NUMPY, intersect, is_datetime, np_digitize
from dstauffman.aerospace import Kf, KfInnov
from dstauffman.plotting.generic import (
    make_categories_plot,
    make_connected_sets,
    make_difference_plot,
    make_generic_plot,
    make_time_plot,
)
from dstauffman.plotting.plotting import Opts, plot_histogram, setup_plots
from dstauffman.plotting.support import (
    COLOR_LISTS,
    ColorMap,
    ExtraPlotter,
    fig_ax_factory,
    get_nondeg_colorlists,
    get_rms_indices,
    plot_phases,
)

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap
    from matplotlib.dates import date2num
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

    inf = np.inf
else:
    from math import inf

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.float64]
    _M = NDArray[np.float64]  # 2D
    _Q = NDArray[np.float64]
    _CM = str | Colormap | ListedColormap | ColorMap
    _Data = int | float | _I | _N | _M | list[_I] | list[_N] | list[_I | _N] | tuple[_I, ...] | tuple[_N, ...] | tuple[_I | _N, ...]  # fmt: skip
    _Time = int | float | datetime.datetime | datetime.date | np.datetime64 | np.int_ | np.float64 | None
    _Times = int | float | datetime.datetime | np.datetime64 | _D | _I | _N | list[_N] | list[_D] | tuple[_N, ...] | tuple[_D, ...]  # fmt: skip
    _DeltaTime = int | float | np.timedelta64
    _Figs = list[Figure]
    _SecUnits = str | int | float | tuple[str, float] | None

    class _KfQuatKwargs(TypedDict):
        name_one: NotRequired[str]
        name_two: NotRequired[str]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Path | None]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        make_subplots: NotRequired[bool]
        plot_components: NotRequired[bool]  # quat-only
        single_lines: NotRequired[bool]
        use_mean: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[_SecUnits]
        show_extra: NotRequired[bool]
        leg_scale: NotRequired[_SecUnits]
        data_as_rows: NotRequired[bool]
        tolerance: NotRequired[_DeltaTime]
        use_zoh: NotRequired[bool]
        extra_plotter: NotRequired[ExtraPlotter | None]
        use_datashader: NotRequired[bool]
        classify: NotRequired[str]

    class _KfDiffKwargs(TypedDict):
        name_one: NotRequired[str]
        name_two: NotRequired[str]
        elements: NotRequired[list[str] | tuple[str, ...] | None]  # diff-only
        units: NotRequired[str]  # diff-only
        save_plot: NotRequired[bool]
        save_path: NotRequired[Path | None]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        make_subplots: NotRequired[bool]
        single_lines: NotRequired[bool]
        colormap: NotRequired[_CM | None]  # diff-only
        use_mean: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[_SecUnits]
        show_extra: NotRequired[bool]
        leg_scale: NotRequired[_SecUnits]
        data_as_rows: NotRequired[bool]
        tolerance: NotRequired[_DeltaTime]
        use_zoh: NotRequired[bool]
        extra_plotter: NotRequired[ExtraPlotter | None]
        use_datashader: NotRequired[bool]
        fig_ax: NotRequired[tuple[Figure, Axes] | None]  # diff-only
        classify: NotRequired[str]

    class _SetsKwargs(TypedDict):
        color_by: NotRequired[str]
        legend_loc: NotRequired[str]
        skip_setup_plots: NotRequired[bool]
        hide_innovs: NotRequired[bool]
        center_origin: NotRequired[bool]
        units: NotRequired[str]
        mag_ratio: NotRequired[float | None]
        leg_scale: NotRequired[_SecUnits]
        colormap: NotRequired[_CM | None]
        use_datashader: NotRequired[bool]
        add_quiver: NotRequired[bool]
        quiver_scale: NotRequired[float | None]
        fig_ax: NotRequired[tuple[Figure, Axes] | None]


# %% Globals
logger = logging.getLogger(__name__)

# %% Constants
# hard-coded values
_LEG_FORMAT = "{:1.3f}"
_TRUTH_COLOR = "k"


# %% Functions - minimize_names
def minimize_names(names: tuple[str, ...], *, sep: str = ",") -> str:
    r"""
    Combine the unique characters in a name to make them simpler.

    Parameters
    ----------
    names : tuple[str, ...]
        Names of the fields to be combined
    sep : str, optional, default is ","
        String to combine between the shortened names

    Returns
    -------
    str
        Combined string of shortened names

    See Also
    --------
    plot_covariance

    Notes
    -----
    #.  Written by David C. Stauffer in July 2024

    Examples
    --------
    >>> from dstauffman.plotting import minimize_names
    >>> names = ("Gyro Bias 1", "Gyro Bias 2", "Gyro Bias 3")
    >>> print(minimize_names(names))
    Gyro Bias 1,2,3

    """
    if len(names) == 0:
        return ""
    if len(names) == 1:
        return names[0]
    all_chars = set()
    for name in names:
        all_chars |= set(name)
    if len(all_chars - frozenset({"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"})) == 0:
        return sep.join(names)
    min_len = min(len(name) for name in names)
    for i in range(min_len):
        if names[0][i].isdigit():
            match = i
            break
        shorts = {name[: i + 1] for name in names}
        if len(shorts) > 1:
            match = i
            break
    else:
        match = i + 1
    return names[0][:match] + sep.join(name[match:] for name in names)


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
    See make_generic_plot for input details.

    Returns
    -------
    fig : class matplotlib.Figure
        figure handle
    err : dict
        Differences

    See Also
    --------
    make_generic_plot

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
    >>> description      = "example"
    >>> time_one         = np.arange(11)
    >>> time_two         = np.arange(2, 13)
    >>> quat_one         = quat_norm(np.random.rand(4, 11))
    >>> quat_two         = quat_norm(quat_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] + 1e-5 * np.random.rand(4, 11))
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
    return make_generic_plot(  # type: ignore[return-value]
        "quat",
        description=description,
        time_one=time_one,
        data_one=quat_one,
        time_two=time_two,
        data_two=quat_two,
        name_one=name_one,
        name_two=name_two,
        elements=("X", "Y", "Z", "S"),
        units="rad",
        time_units=time_units,
        start_date=start_date,
        rms_xmin=rms_xmin,
        rms_xmax=rms_xmax,
        disp_xmin=disp_xmin,
        disp_xmax=disp_xmax,
        single_lines=single_lines,
        make_subplots=make_subplots,
        colormap=colormap,
        use_mean=use_mean,
        plot_zero=plot_zero,
        show_rms=show_rms,
        legend_loc=legend_loc,
        show_extra=show_extra,
        plot_components=plot_components,
        second_units=second_units,
        leg_scale=leg_scale,
        tolerance=tolerance,
        return_err=return_err,
        data_as_rows=data_as_rows,
        extra_plotter=extra_plotter,
        use_zoh=use_zoh,
        label_vert_lines=label_vert_lines,
        use_datashader=use_datashader,
    )


# %% plot_quaternion
@overload
def plot_quaternion(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    opts: Opts | None,
    skip_setup_plots: bool,
    return_err: Literal[False] = ...,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs: ...
@overload
def plot_quaternion(
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    opts: Opts | None,
    skip_setup_plots: bool,
    return_err: Literal[True],
    **kwargs: Unpack[_KfQuatKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_quaternion(  # noqa: C901
    description: str,
    time_one: _Times | None,
    time_two: _Times | None,
    quat_one: _Q | None,
    quat_two: _Q | None,
    *,
    opts: Opts | None = None,
    skip_setup_plots: bool = False,
    return_err: bool = False,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""
    Plots the attitude quaternion history without making explicit Kf classes.

    Parameters
    ----------
    description : str
        name of the data being plotted, used as title
    time_one : (A,) array_like
        time history for channel one, [sec] or datetime64
    time_two : (B,) array_like
        time history for channel two, [sec] or datetime64
    quat_one : (4,) or (4, A) ndarray, or (A, 4) ndarray if data_as_rows is False
        quaternion history for channel one
    quat_two : (4,) or (4, B) ndarray, or (B, 4) ndarray if data_as_rows is False
        quaternion history for channel two
    opts : class Opts, optional
        Plotting options
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis
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
    >>> from dstauffman.plotting import Opts, plot_quaternion
    >>> from dstauffman.aerospace import quat_from_euler, quat_mult, quat_norm
    >>> import numpy as np

    >>> q1 = quat_norm(np.array([0.1, -0.2, 0.3, 0.4]))
    >>> dq = quat_from_euler(1e-6*np.array([-300, 100, 200]), [3, 1, 2])
    >>> q2 = quat_mult(dq, q1)

    >>> time_one = np.arange(11)
    >>> quat_one = np.tile(q1[:, np.newaxis], (1, time_one.size))

    >>> time_two = np.arange(2, 13)
    >>> quat_two = np.tile(q2[:, np.newaxis], (1, time_two.size))
    >>> quat_two[3,4] += 50e-6
    >>> quat_two = quat_norm(quat_two)

    >>> opts = Opts()
    >>> opts.case_name = "test_plot"
    >>> opts.quat_comp = True
    >>> opts.sub_plots = True

    >>> fig_hand = plot_quaternion("Quaternion", time_one, time_two, quat_one, quat_two, \
    ...                            opts=opts, name_one="KF1", name_two="KF2")

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()

    # determine if converting units
    is_date_1 = is_datetime(time_one)
    is_date_2 = is_datetime(time_two)
    is_date_o = opts.time_unit in {"numpy", "datetime"}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates("numpy", old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates("sec", old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    sub_plots    = kwargs.pop("make_subplots", this_opts.sub_plots)
    plot_comps   = kwargs.pop("plot_components", this_opts.quat_comp)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    show_extra   = kwargs.pop("show_extra", this_opts.show_xtra)
    name_one, name_two = this_opts.get_name_one_and_two(kwargs)
    # fmt: on

    # hard-coded defaults
    second_units = kwargs.pop("second_units", "micro")

    # initialize outputs
    figs: _Figs = []

    # print status
    logger.log(LogLevel.L4, "Plotting %s plots ...", description)

    # make plots
    out = make_quaternion_plot(  # type: ignore[call-overload, misc]
        description,
        time_one,
        time_two,
        quat_one,
        quat_two,
        name_one=name_one,
        name_two=name_two,
        time_units=time_units,
        start_date=start_date,
        rms_xmin=rms_xmin,
        rms_xmax=rms_xmax,
        disp_xmin=disp_xmin,
        disp_xmax=disp_xmax,
        make_subplots=sub_plots,
        plot_components=plot_comps,
        single_lines=single_lines,
        use_mean=use_mean,
        label_vert_lines=lab_vert,
        plot_zero=plot_zero,
        show_rms=show_rms,
        legend_loc=legend_loc,
        show_extra=show_extra,
        second_units=second_units,
        return_err=return_err,
        **kwargs,
    )
    if return_err:
        figs += out[0]
        err: dict[str, _N] = out[1]
    else:
        figs += out

    # Setup plots
    if not skip_setup_plots:
        setup_plots(figs, opts)
    logger.log(LogLevel.L4, "... done.")
    if return_err:
        return (figs, err)
    return figs


# %% plot_attitude
@overload
def plot_attitude(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs: ...
@overload
def plot_attitude(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_attitude(  # noqa: C901
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
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
    >>> kf1.name = "KF1"
    >>> kf1.time = np.arange(11)
    >>> kf1.att  = np.tile(q1[:, np.newaxis], (1, kf1.time.size))

    >>> kf2      = Kf()
    >>> kf2.name = "KF2"
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.att  = np.tile(q2[:, np.newaxis], (1, kf2.time.size))
    >>> kf2.att[3,4] += 50e-6
    >>> kf2.att = quat_norm(kf2.att)

    >>> opts = Opts()
    >>> opts.case_name = "test_plot"
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
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {"att": "Attitude Quaternion"}

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {"numpy", "datetime"}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates("numpy", old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates("sec", old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    sub_plots    = kwargs.pop("make_subplots", this_opts.sub_plots)
    plot_comps   = kwargs.pop("plot_components", this_opts.quat_comp)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    name_one, name_two = this_opts.get_name_one_and_two(kwargs, kf1=kf1, kf2=kf2)
    # fmt: on

    # hard-coded defaults
    second_units = kwargs.pop("second_units", "micro")

    # initialize outputs
    figs: _Figs = []
    err: dict[str, _N] = {}
    printed = False

    if truth is not None:
        raise NotImplementedError("Truth manipulations are not yet implemented.")

    # call wrapper function for most of the details
    for field, description in fields.items():
        # print status
        if not printed:
            logger.log(LogLevel.L4, "Plotting %s plots ...", description)
            printed = True
        # make plots
        out = make_quaternion_plot(  # type: ignore[call-overload, misc]
            description,
            kf1.time,
            kf2.time,
            getattr(kf1, field),
            getattr(kf2, field),
            name_one=name_one,
            name_two=name_two,
            time_units=time_units,
            start_date=start_date,
            rms_xmin=rms_xmin,
            rms_xmax=rms_xmax,
            disp_xmin=disp_xmin,
            disp_xmax=disp_xmax,
            make_subplots=sub_plots,
            plot_components=plot_comps,
            single_lines=single_lines,
            use_mean=use_mean,
            label_vert_lines=lab_vert,
            plot_zero=plot_zero,
            show_rms=show_rms,
            legend_loc=legend_loc,
            second_units=second_units,
            return_err=return_err,
            **kwargs,
        )
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, "... done.")
    if return_err:
        return (figs, err)
    return figs


# %% plot_los
@overload
def plot_los(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs: ...
@overload
def plot_los(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_los(
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfQuatKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {"los": "LOS"}
    out = plot_attitude(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)  # type: ignore[call-overload]
    return out  # type: ignore[no-any-return]


# %% plot_position
@overload
def plot_position(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs: ...
@overload
def plot_position(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_position(  # noqa: C901
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
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
    >>> kf1.name = "KF1"
    >>> kf1.time = np.arange(11)
    >>> kf1.pos  = 1e6 * np.random.rand(3, 11)
    >>> kf1.vel  = 1e3 * np.random.rand(3, 11)

    >>> kf2      = Kf()
    >>> kf2.name = "KF2"
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
        fields = {"pos": "Position"}

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {"numpy", "datetime"}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates("numpy", old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates("sec", old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    sub_plots    = kwargs.pop("make_subplots", this_opts.sub_plots)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    show_extra   = kwargs.pop("show_extra", this_opts.show_xtra)

    # hard-coded defaults
    elements      = kwargs.pop("elements", ["x", "y", "z"])
    default_units = "m" if "pos" in fields else "m/s" if "vel" in fields else ""
    units         = kwargs.pop("units", default_units)
    second_units  = kwargs.pop("second_units", "kilo")
    colormap      = get_nondeg_colorlists(3)
    name_one, name_two = this_opts.get_name_one_and_two(kwargs, kf1=kf1, kf2=kf2)
    # fmt: on

    # initialize outputs
    figs: _Figs = []
    err = {}
    printed = False

    # call wrapper function for most of the details
    for field, description in fields.items():
        # print status
        if not printed:
            logger.log(LogLevel.L4, "Plotting %s plots ...", description)
            printed = True
        # make plots
        out = make_difference_plot(  # type: ignore[call-overload, misc]
            description,
            kf1.time,
            kf2.time,
            getattr(kf1, field),
            getattr(kf2, field),
            name_one=name_one,
            name_two=name_two,
            elements=elements,
            time_units=time_units,
            units=units,
            start_date=start_date,
            rms_xmin=rms_xmin,
            rms_xmax=rms_xmax,
            disp_xmin=disp_xmin,
            disp_xmax=disp_xmax,
            make_subplots=sub_plots,
            colormap=colormap,
            use_mean=use_mean,
            label_vert_lines=lab_vert,
            plot_zero=plot_zero,
            single_lines=single_lines,
            show_rms=show_rms,
            legend_loc=legend_loc,
            show_extra=show_extra,
            second_units=second_units,
            return_err=return_err,
            **kwargs,
        )
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, "... done.")
    if return_err:
        return (figs, err)
    return figs


# %% plot_velocity
@overload
def plot_velocity(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs: ...
@overload
def plot_velocity(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_velocity(
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {"vel": "Velocity"}
    out = plot_position(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)  # type: ignore[call-overload]
    return out  # type: ignore[no-any-return]


# %% plot_innovations
@overload
def plot_innovations(
    kf1: KfInnov | None,
    kf2: KfInnov | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    fields: dict[str, str] | None,
    plot_by_status: bool,
    plot_by_number: bool,
    show_one: _B | None,
    show_two: _B | None,
    cat_names: dict[Any, str] | None,
    cat_colors: _CM | None,
    number_field: dict[str, str] | None,
    number_colors: _CM | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs: ...
@overload
def plot_innovations(
    kf1: KfInnov | None,
    kf2: KfInnov | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    fields: dict[str, str] | None,
    plot_by_status: bool,
    plot_by_number: bool,
    show_one: _B | None,
    show_two: _B | None,
    cat_names: dict[Any, str] | None,
    cat_colors: _CM | None,
    number_field: dict[str, str] | None,
    number_colors: _CM | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_innovations(  # noqa: C901
    kf1: KfInnov | None = None,
    kf2: KfInnov | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    fields: dict[str, str] | None = None,
    plot_by_status: bool = False,
    plot_by_number: bool = False,
    show_one: _B | None = None,
    show_two: _B | None = None,
    cat_names: dict[Any, str] | None = None,
    cat_colors: _CM | None = None,
    number_field: dict[str, str] | None = None,
    number_colors: _CM | None = None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
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
    >>> kf1.units = "m"
    >>> kf1.time  = np.arange(num_innovs, dtype=float)
    >>> kf1.innov = 1e-6 * np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
    >>> kf1.norm  = np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)

    >>> ix        = np.hstack((np.arange(7), np.arange(8, num_innovs)))
    >>> kf2       = KfInnov()
    >>> kf2.time  = kf1.time[ix]
    >>> kf2.innov = kf1.innov[:, ix] + 1e-8 * np.random.rand(num_axes, ix.size)
    >>> kf2.norm  = kf1.norm[:, ix] + 0.1 * np.random.rand(num_axes, ix.size)

    >>> opts = Opts()
    >>> opts.case_name = "test_plot"
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
        pass  # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {"innov": "Innovations", "norm": "Normalized Innovations"}
    if number_field is None:
        number_field = {"quad": "Quad", "sca": "SCA"}

    # aliases and defaults
    num_chan = 0
    for key in fields.keys():
        if getattr(kf1, key) is not None:
            temp = getattr(kf1, key).shape[0]
        elif getattr(kf2, key) is not None:
            temp = getattr(kf2, key).shape[0]
        else:
            temp = 0
        num_chan = max(num_chan, temp)
    # fmt: off
    elements: list[str] | tuple[str, ...] | None
    elements     = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f"Channel {i+1}" for i in range(num_chan)]
    elements     = kwargs.pop("elements", elements)
    units        = kwargs.pop("units", kf1.units)
    second_units = kwargs.pop("second_units", "micro")
    # fmt: on

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {"numpy", "datetime"}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates("numpy", old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates("sec", old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    sub_plots    = kwargs.pop("make_subplots", this_opts.sub_plots)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    show_extra   = kwargs.pop("show_extra", this_opts.show_xtra)
    colormap     = kwargs.pop("colormap", this_opts.colormap)
    tolerance    = kwargs.pop("tolerance", 0)
    name_one, name_two = this_opts.get_name_one_and_two(kwargs, kf1=kf1, kf2=kf2)
    description = name_one if name_one else name_two if name_two else ""
    # fmt: on

    # Initialize outputs
    figs: _Figs = []
    err = {}
    printed = False

    # % call wrapper functions for most of the details
    for field, sub_description in fields.items():
        full_description = description + " - " + sub_description if description else sub_description
        # print status
        if not printed:
            logger.log(LogLevel.L4, "Plotting %s plots ...", full_description)
            printed = True
        # make plots
        if "Normalized" in sub_description:
            units = "σ"
            this_second_units = "unity"
        else:
            this_second_units = second_units  # type: ignore[assignment]
        field_one = getattr(kf1, field)
        field_two = getattr(kf2, field)
        if field_one is not None and show_one is not None:
            t1 = kf1.time[show_one]  # type: ignore[index]
            f1 = field_one[:, show_one]
        else:
            t1 = kf1.time
            f1 = field_one
        if field_two is not None and show_two is not None:
            t2 = kf2.time[show_two]  # type: ignore[index]
            f2 = field_two[:, show_two]
        else:
            t2 = kf2.time
            f2 = field_two
        out = make_difference_plot(  # type: ignore[call-overload, misc]
            full_description,
            t1,
            t2,
            f1,
            f2,
            name_one=name_one,
            name_two=name_two,
            elements=elements,
            units=units,
            time_units=time_units,
            start_date=start_date,
            rms_xmin=rms_xmin,
            rms_xmax=rms_xmax,
            disp_xmin=disp_xmin,
            disp_xmax=disp_xmax,
            make_subplots=sub_plots,
            use_mean=use_mean,
            label_vert_lines=lab_vert,
            plot_zero=plot_zero,
            show_rms=show_rms,
            single_lines=single_lines,
            legend_loc=legend_loc,
            show_extra=show_extra,
            second_units=this_second_units,
            colormap=colormap,
            return_err=return_err,
            tolerance=tolerance,
            **kwargs,
        )
        if return_err:
            figs += out[0]
            err[field] = out[1]
        else:
            figs += out
        this_ylabel = [e + " Innovation [" + units + "]" for e in elements] if elements is not None else None
        if plot_by_status and field_one is not None and kf1.status is not None:
            figs += make_categories_plot(  # type: ignore[misc]
                full_description + " by Category",
                kf1.time,
                field_one,
                kf1.status,
                name=name_one,
                cat_names=cat_names,
                elements=elements,
                units=units,
                time_units=time_units,
                start_date=start_date,
                rms_xmin=rms_xmin,
                rms_xmax=rms_xmax,
                disp_xmin=disp_xmin,
                disp_xmax=disp_xmax,
                make_subplots=sub_plots,
                use_mean=use_mean,
                label_vert_lines=lab_vert,
                plot_zero=plot_zero,
                show_rms=show_rms,
                single_lines=single_lines,
                legend_loc=legend_loc,
                second_units=this_second_units,
                ylabel=this_ylabel,
                colormap=cat_colors,
                **kwargs,
            )
        if plot_by_status and field_two is not None and kf2.status is not None:
            figs += make_categories_plot(  # type: ignore[misc]
                full_description + " by Category",
                kf2.time,
                field_two,
                kf2.status,
                name=name_two,
                cat_names=cat_names,
                elements=elements,
                units=units,
                time_units=time_units,
                start_date=start_date,
                rms_xmin=rms_xmin,
                rms_xmax=rms_xmax,
                disp_xmin=disp_xmin,
                disp_xmax=disp_xmax,
                make_subplots=sub_plots,
                use_mean=use_mean,
                label_vert_lines=lab_vert,
                plot_zero=plot_zero,
                show_rms=show_rms,
                single_lines=single_lines,
                legend_loc=legend_loc,
                second_units=this_second_units,
                ylabel=this_ylabel,
                colormap=cat_colors,
                **kwargs,
            )
        if plot_by_number and field_one is not None and ~np.all(np.isnan(field_one)):
            this_number = None
            quad_name: str | None = None
            for quad, quad_name in number_field.items():
                if hasattr(kf1, quad):
                    this_number = getattr(kf1, quad)
                    break
            if this_number is not None:
                assert isinstance(quad_name, str), "quad_name should have been set in earlier for loop."
                num_names = {num: quad_name + " " + str(num) for num in np.unique(this_number)}
                figs += make_categories_plot(  # type: ignore[misc]
                    full_description + " by " + quad_name,
                    kf1.time,
                    field_one,
                    this_number,
                    name=name_one,
                    cat_names=num_names,
                    elements=elements,
                    units=units,
                    time_units=time_units,
                    start_date=start_date,
                    rms_xmin=rms_xmin,
                    rms_xmax=rms_xmax,
                    disp_xmin=disp_xmin,
                    disp_xmax=disp_xmax,
                    make_subplots=sub_plots,
                    use_mean=use_mean,
                    label_vert_lines=lab_vert,
                    plot_zero=plot_zero,
                    show_rms=show_rms,
                    single_lines=single_lines,
                    legend_loc=legend_loc,
                    second_units=this_second_units,
                    ylabel=this_ylabel,
                    colormap=number_colors,
                    **kwargs,
                )
        if plot_by_number and field_two is not None and ~np.all(np.isnan(field_two)):
            this_number = None
            for quad, quad_name in number_field.items():
                if hasattr(kf2, quad):
                    this_number = getattr(kf2, quad)
                    break
            if this_number is not None:
                num_names = {num: quad_name + " " + str(num) for num in np.unique(this_number)}
                figs += make_categories_plot(  # type: ignore[misc]
                    full_description + " by " + quad_name,
                    kf2.time,
                    field_two,
                    this_number,
                    name=name_two,
                    cat_names=num_names,
                    elements=elements,
                    units=units,
                    time_units=time_units,
                    start_date=start_date,
                    rms_xmin=rms_xmin,
                    rms_xmax=rms_xmax,
                    disp_xmin=disp_xmin,
                    disp_xmax=disp_xmax,
                    make_subplots=sub_plots,
                    use_mean=use_mean,
                    label_vert_lines=lab_vert,
                    plot_zero=plot_zero,
                    show_rms=show_rms,
                    single_lines=single_lines,
                    legend_loc=legend_loc,
                    second_units=this_second_units,
                    ylabel=this_ylabel,
                    colormap=number_colors,
                    **kwargs,
                )

    # Setup plots
    setup_plots(figs, opts)
    if printed:
        logger.log(LogLevel.L4, "... done.")
    if return_err:
        return (figs, err)
    return figs


# %% plot_innov_fplocs
def plot_innov_fplocs(
    kf1: KfInnov | None,
    *,
    opts: Opts | None = None,
    t_bounds: list[_Time] | None = None,
    mask: _B | None = None,
    **kwargs: Unpack[_SetsKwargs],
) -> _Figs:
    r"""
    Plots the innovations on the focal plane, connecting the sighting and prediction with the innovation.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    opts : class Opts, optional
        Plotting options
    t_bounds : (2,) ndarray, optional
        Minimum and maximum time bounds to plot
    mask : (N,) ndarray, optional
        Mask array
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
    >>> kf1.units = "m"
    >>> kf1.time  = np.arange(num_innovs, dtype=float)
    >>> kf1.innov = np.full((num_axes, num_innovs), 5e-3) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
    >>> kf1.innov[:, :5] *= 0.1
    >>> kf1.fploc = np.full((num_axes, num_innovs), 0.05) + 0.2 * np.random.rand(num_axes, num_innovs) - 0.1

    >>> opts = Opts()
    >>> opts.case_name = "test_plot"
    >>> opts.sub_plots = True

    >>> fig_hand = plot_innov_fplocs(kf1, opts=opts, color_by="magnitude")

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

    name = kf1.name + " - " if kf1.name else ""
    description = name + "Focal Plane Sightings"
    extra_text = f'(by {kwargs["color_by"]}) ' if "color_by" in kwargs and kwargs["color_by"] != "none" else ""
    logger.log(LogLevel.L4, "Plotting %s plots %s...", description, extra_text)

    # check for data
    if kf1.fploc is None:
        logger.log(LogLevel.L5, "No focal plane data was provided, so no plots were generated.")
        return []

    # alias opts
    legend_loc = kwargs.pop("legend_loc", opts.leg_spot)
    skip_setup_plots = kwargs.pop("skip_setup_plots", False)

    # pull out time subset
    if t_bounds is None:
        if mask is None:
            fplocs = kf1.fploc
            innovs = kf1.innov
        else:
            fplocs = kf1.fploc[:, mask]
            innovs = kf1.innov[:, mask]  # type: ignore[index]
    else:
        ix = get_rms_indices(kf1.time, xmin=t_bounds[0], xmax=t_bounds[1])
        if mask is None:
            this_mask = ix["one"]
        else:
            this_mask = mask & ix["one"]
        fplocs = kf1.fploc[:, this_mask]
        innovs = kf1.innov[:, this_mask]  # type: ignore[index]

    # call wrapper functions for most of the details
    fig = make_connected_sets(description, fplocs, innovs, units=kf1.units, legend_loc=legend_loc, **kwargs)  # type: ignore[arg-type, misc]

    # Setup plots
    figs = [fig]
    if not skip_setup_plots:
        setup_plots(figs, opts)
    logger.log(LogLevel.L4, "... done.")
    return figs


# %% plot_innov_hist
def plot_innov_hist(
    kf1: KfInnov | None,
    bins: _N | list[float],
    *,
    opts: Opts | None = None,
    fields: dict[str, str] | None = None,
    normalize_spacing: bool = False,
    use_exact_counts: bool = False,
    show_cdf: bool = False,
    cdf_x: float | list[float] | None = None,
    cdf_y: float | list[float] | None = None,
) -> _Figs:
    r"""Plots the innovation histogram."""
    # check optional inputs
    if kf1 is None:
        kf1 = KfInnov()
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {"innov": "Innovations", "norm": "Normalized Innovations"}

    description = kf1.name if kf1.name else ""
    logger.log(LogLevel.L4, "Plotting %s plots ...", description)

    # check for data
    if kf1.innov is None:
        logger.log(LogLevel.L5, "No innovation data was provided, so no plots were generated.")
        return []
    data = kf1.innov[0, :]

    figs: _Figs = []
    printed = False

    # % call wrapper functions for most of the details
    for field, sub_description in fields.items():
        full_description = (
            description + " - " + sub_description + " Histogram" if description else sub_description + " Histogram"
        )
        # print status
        if not printed:
            logger.log(LogLevel.L4, "Plotting %s plots ...", full_description)
            printed = True
        data = getattr(kf1, field)
        for i in range(data.shape[0]):
            fig = plot_histogram(
                full_description,
                data[i, :],
                bins,
                opts=opts,
                color="#1f77b4",
                xlabel="Data",
                ylabel="Number",
                second_ylabel="Distribution [%]",
                normalize_spacing=normalize_spacing,
                use_exact_counts=use_exact_counts,
                show_cdf=show_cdf,
                cdf_x=cdf_x,
                cdf_y=cdf_y,
            )
            figs.append(fig)
    return figs


# %% plot_covariance
@overload
def plot_covariance(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    groups: list[int | _I] | tuple[int, ...] | None,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs: ...
@overload
def plot_covariance(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    groups: list[int | _I] | tuple[int, ...] | None,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> tuple[_Figs, dict[str, dict[str, _N]]]: ...
def plot_covariance(  # noqa: C901
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    groups: list[int | _I] | tuple[int, ...] | None = None,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs | tuple[_Figs, dict[str, dict[str, _N]]]:
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
    >>> kf1.name   = "KF1"
    >>> kf1.time   = np.arange(num_points, dtype=float)
    >>> kf1.covar  = 1e-6 * np.tile(np.arange(1, num_states+1, dtype=float)[:, np.newaxis], (1, num_points))
    >>> kf1.active = np.array([1, 2, 3, 4, 8, 12])

    >>> kf2        = Kf(name="KF2")
    >>> kf2.time   = kf1.time
    >>> kf2.covar  = kf1.covar + 1e-9 * np.random.rand(*kf1.covar.shape)
    >>> kf2.active = kf1.active

    >>> opts = Opts()
    >>> opts.case_name = "test_plot"
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
        pass  # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {"covar": "Covariance"}

    # TODO: allow different sets of states in the different structures

    # aliases and defaults
    num_chan = 0
    for key in fields.keys():
        if getattr(kf1, key) is not None:
            temp = getattr(kf1, key).shape[0]
        elif getattr(kf2, key) is not None:
            temp = getattr(kf2, key).shape[0]
        else:
            temp = 0
        num_chan = max(num_chan, temp)
    # fmt: off
    elements: list[str] | tuple[str, ...] | None
    elements     = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f"Channel {i+1}" for i in range(num_chan)]
    elements     = kwargs.pop("elements", elements)
    units        = kwargs.pop("units", "mixed")
    second_units = kwargs.pop("second_units", "micro")
    # fmt: on
    if groups is None:
        groups = list(range(num_chan))

    # determine if converting units
    is_date_1 = is_datetime(kf1.time)
    is_date_2 = is_datetime(kf2.time)
    is_date_o = opts.time_unit in {"numpy", "datetime"}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # allow opts to convert as necessary
    if is_date_1 or is_date_2 and not is_date_o:
        this_opts.convert_dates("numpy", old_form=opts.time_base)
    elif is_date_o and not is_date_1 and not is_date_2:
        this_opts.convert_dates("sec", old_form=opts.time_base)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    sub_plots    = kwargs.pop("make_subplots", this_opts.sub_plots)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    show_extra   = kwargs.pop("show_extra", this_opts.show_xtra)
    name_one, name_two = this_opts.get_name_one_and_two(kwargs, kf1=kf1, kf2=kf2)
    # fmt: on

    # initialize output
    figs: _Figs = []
    err: dict[str, dict[str, _N]] = {}

    # % call wrapper functions for most of the details
    for field, description in fields.items():
        logger.log(LogLevel.L4, "Plotting %s plots ...", description)
        err[field] = {}
        for ix, states in enumerate(groups):
            this_units = units if isinstance(units, str) else units[ix]
            this_2units = second_units[ix] if isinstance(second_units, list) else second_units  # type: ignore[index]
            this_ylabel = description + f" [{this_units}]"
            states = np.atleast_1d(states)
            if hasattr(kf1, "active") and kf1.active is not None:
                (this_state_nums1, this_state_rows1, found_rows1) = intersect(kf1.active, states, return_indices=True)  # type: ignore[call-overload]
            else:
                this_state_nums1 = np.array([], dtype=int)
            if hasattr(kf2, "active") and kf2.active is not None:
                (this_state_nums2, this_state_rows2, found_rows2) = intersect(kf2.active, states, return_indices=True)  # type: ignore[call-overload]
            else:
                this_state_nums2 = np.array([], dtype=int)
            this_state_nums = np.union1d(this_state_nums1, this_state_nums2)
            data_one = np.atleast_2d(getattr(kf1, field)[this_state_rows1, :]) if getattr(kf1, field) is not None else None
            data_two = np.atleast_2d(getattr(kf2, field)[this_state_rows2, :]) if getattr(kf2, field) is not None else None
            have_data1 = data_one is not None and np.any(~np.isnan(data_one))
            have_data2 = data_two is not None and np.any(~np.isnan(data_two))
            if have_data1 and this_state_nums1.size < this_state_nums.size:
                temp = np.full((this_state_nums.size, data_one.shape[1]), np.nan)  # pyright: ignore[reportOptionalMemberAccess]
                temp[found_rows1, :] = data_one
                data_one = temp
            if have_data2 and this_state_nums2.size < this_state_nums.size:
                temp = np.full((this_state_nums.size, data_two.shape[1]), np.nan)  # pyright: ignore[reportOptionalMemberAccess]
                temp[found_rows2, :] = data_two
                data_two = temp
            if have_data1 or have_data2:
                state_names = tuple(str(state) if elements is None else elements[state] for state in this_state_nums)
                this_description = description + " for State " + minimize_names(state_names)
                this_elements = [elements[state] for state in this_state_nums] if elements is not None else None
                colormap = get_nondeg_colorlists(len(this_elements)) if this_elements is not None else None
                out = make_difference_plot(  # type: ignore[call-overload, misc]
                    this_description,
                    kf1.time,
                    kf2.time,
                    data_one,
                    data_two,
                    name_one=name_one,
                    name_two=name_two,
                    elements=this_elements,
                    units=this_units,
                    time_units=time_units,
                    start_date=start_date,
                    rms_xmin=rms_xmin,
                    rms_xmax=rms_xmax,
                    disp_xmin=disp_xmin,
                    disp_xmax=disp_xmax,
                    make_subplots=sub_plots,
                    use_mean=use_mean,
                    label_vert_lines=lab_vert,
                    plot_zero=plot_zero,
                    show_rms=show_rms,
                    single_lines=single_lines,
                    legend_loc=legend_loc,
                    show_extra=show_extra,
                    second_units=this_2units,
                    return_err=return_err,
                    ylabel=this_ylabel,
                    colormap=colormap,
                    **kwargs,
                )
                if return_err:
                    figs += out[0]
                    err[field][f"Group {ix+1}"] = out[1]
                else:
                    figs += out
        logger.log(LogLevel.L4, "... done.")

    # Setup plots
    setup_plots(figs, opts)
    if not figs:
        logger.log(  # pylint: disable=logging-fstring-interpolation
            LogLevel.L5, f"No {'/'.join(fields.values())} data was provided, so no plots were generated."
        )
    if return_err:
        return (figs, err)
    return figs


# %% plot_states
@overload
def plot_states(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[False] = ...,
    groups: list[int | _I] | tuple[int, ...] | None,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs: ...
@overload
def plot_states(
    kf1: Kf | None,
    kf2: Kf | None,
    *,
    truth: Kf | None,
    opts: Opts | None,
    return_err: Literal[True],
    groups: list[int | _I] | tuple[int, ...] | None,
    fields: dict[str, str] | None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> tuple[_Figs, dict[str, _N]]: ...
def plot_states(
    kf1: Kf | None = None,
    kf2: Kf | None = None,
    *,
    truth: Kf | None = None,
    opts: Opts | None = None,
    return_err: bool = False,
    groups: list[int | _I] | tuple[int, ...] | None = None,
    fields: dict[str, str] | None = None,
    **kwargs: Unpack[_KfDiffKwargs],
) -> _Figs | tuple[_Figs, dict[str, _N]]:
    r"""Plots the Kalman Filter state histories."""
    if fields is None:
        fields = {"state": "State Estimates"}
    out = plot_covariance(  # type: ignore[call-overload, misc]
        kf1, kf2, truth=truth, opts=opts, return_err=return_err, groups=groups, fields=fields, **kwargs
    )
    return out  # type: ignore[no-any-return]


# %% Functions - plot_tci
def plot_tci(
    time: _D,
    data: _N,
    *,
    solar_cycles: _D | None = None,
    solar_labels: list[str] | None = None,
    opts: Opts | None = None,
) -> Figure:
    """
    Plots the Thermosphere Climate Index (TCI).

    Parameters
    ----------
    time : (N,)
        Time
    data : (N,)
        Thermosphere Climate Index data
    solar_cyles : (A,) optional
        Solar cycle start times
    solar_labels : (A,) list of str
        Solar cycle labels
    opts : class Opts, optional
        Plotting options

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in October 2023.

    Examples
    --------
    >>> from dstauffman.plotting import plot_tci
    >>> from dstauffman.aerospace import read_solar_cycles, read_tci_data
    >>> from dstauffman import get_data_dir
    >>> from matplotlib.dates import date2num
    >>> import numpy as np
    >>> folder = get_data_dir()
    >>> tci_file = folder / "tci_info.txt"
    >>> tci_data = read_tci_data(tci_file)
    >>> solar_file = folder / "Solar_Cycles.txt"
    >>> solar_data = read_solar_cycles(solar_file)
    >>> time = tci_data.Date.to_numpy()
    >>> data = tci_data.TCI.to_numpy()
    >>> data[data < 0.0] = np.nan
    >>> solar_cycles = date2num(solar_data.Start[16:].to_numpy())
    >>> solar_labels = [f"SC {name}" for name in solar_data.Solar_Cycle[16:]]
    >>> fig = plot_tci(time, data, solar_cycles=solar_cycles, solar_labels=solar_labels)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    if opts is None:
        opts = Opts()
    # Find the quintiles
    quintiles = np.nanpercentile(data, [20, 40, 60, 80, 100])
    quintile_names = ("Cold", "Cool", "Neutral", "Warm", "Hot")
    quintile_colors = ("xkcd:royal blue", "xkcd:azure", "xkcd:grey", "xkcd:tangerine", "xkcd:bright red")
    # Create the figure
    fig_ax: tuple[tuple[Figure, Axes]] = fig_ax_factory(1, 1)  # type: ignore[call-overload]
    fig, ax = fig_ax[0]
    # Plot the basic data
    title = "Thermosphere Climate Index"
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(title)
    # fmt: off
    make_time_plot(
        title, time, data, units="W", second_units=("10^11 W", 1e-11), time_units="numpy",
        fig_ax=fig_ax[0], show_rms=False, legend_loc="none", ylabel="Power [W]",
    )
    # fmt: on
    # ax.set_xlabel("Year")
    ax.set_ylim(0.0, 6.0e11)
    fig.axes[1].set_ylim(0.0, 6.0)
    if solar_cycles is not None:
        plot_phases(ax, solar_cycles, labels=solar_labels)
    for name, color, value in zip(quintile_names, quintile_colors, quintiles):
        ax.axhline(value, label=name, color=color)
        ax.annotate(name, (time[0], value), color=color, fontsize=16, verticalalignment="top", zorder=10)
    setup_plots(fig, opts=opts)
    return fig


# %% Functions - plot_kp
def plot_kp(
    time: _D,
    data: _N,
    *,
    opts: Opts | None = None,
) -> Figure:
    """
    Plots the K planetary index 3-hourly data.

    Parameters
    ----------
    time : (N,)
        Time
    data : (N,)
        Planetary K index
    opts : class Opts, optional
        Plotting options

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in May 2024.

    Examples
    --------
    >>> from dstauffman.plotting import plot_kp
    >>> from dstauffman.aerospace import read_kp_ap_nowcast
    >>> from dstauffman import get_data_dir
    >>> import numpy as np
    >>> folder = get_data_dir()
    >>> kp_file = folder / "Kp_ap_nowcast.txt"
    >>> kp_data = read_kp_ap_nowcast(kp_file)
    >>> time = kp_data.GMT.to_numpy()
    >>> data = kp_data.Kp.to_numpy()
    >>> data[data < 0.0] = np.nan
    >>> fig = plot_kp(time, data)

    Close plots
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    if opts is None:
        opts = Opts()
    # calculate the colors
    kp_bins = np.array([-1e-5, 5.0, 6.0, 7.0, 8.0, 8.99, 100.0])
    delta_time = 0.120  # days
    data_clean = np.where(np.isnan(data) | (data < 0.0), 0.0, data)
    kp_color_ix = np_digitize(data_clean, kp_bins, right=True)
    kp_colormap = ListedColormap(
        ("xkcd:easter green", "xkcd:dandelion", "xkcd:tangerine", "xkcd:orange", "xkcd:bright red", "xkcd:red")
    )
    # create the figure
    fig_ax: tuple[tuple[Figure, Axes]] = fig_ax_factory(1, 1)  # type: ignore[call-overload]
    fig, ax = fig_ax[0]
    # plot the data
    title = "Estimated Planetary K index (3 hour data)"
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(title)
    ax.plot([time[0], time[-1]], [data[0], data[-1]], ".", alpha=0.05)
    for t, d, b in zip(date2num(time), data, kp_color_ix):
        if ~np.isnan(d):
            ax.add_patch(
                Rectangle(
                    (t, 0),
                    delta_time,
                    d,
                    facecolor=kp_colormap.colors[b],  # type: ignore[index]
                    edgecolor="none",
                )
            )
    ax.set_ylabel("Kp index")
    ax.grid(True)
    ax.set_ylim(0.0, 9.0)
    ax.set_title(title)

    setup_plots(fig, opts=opts)
    return fig


# %% Unit Test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_aerospace", exit=False)
    doctest.testmod(verbose=False)
