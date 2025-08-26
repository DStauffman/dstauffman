r"""
Plotting functions related to Batch Parameter Estimation.

Notes
-----
#.  Written by David C. Stauffer in July 2016.
#.  Moved to joint plotting submodule by David C. Stauffer in July 2020.

"""

# %% Imports
from __future__ import annotations

import datetime
import doctest
from pathlib import Path
from typing import NotRequired, Sequence, TYPE_CHECKING, TypedDict, Unpack
import unittest

from dstauffman import HAVE_MPL, HAVE_NUMPY
from dstauffman.estimation import BpeResults
from dstauffman.plotting.plotting import ExtraPlotter, Opts, plot_correlation_matrix, plot_time_history, setup_plots
from dstauffman.plotting.support import ColorMap

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _CM = str | Colormap | ListedColormap | ColorMap
    _Figs = list[Figure]
    _Time = int | float | np.floating | datetime.datetime | datetime.date | np.datetime64 | None

    class _BpeKwArgs(TypedDict):
        name: NotRequired[str]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Path | None]
        units: NotRequired[str]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        single_lines: NotRequired[bool]
        colormap: NotRequired[_CM | None]
        use_mean: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[str | int | float | tuple[str, float] | None]
        leg_scale: NotRequired[str | int | float | tuple[str, float] | None]
        ylabel: NotRequired[str | list[str] | None]
        data_as_rows: NotRequired[bool]
        extra_plotter: NotRequired[ExtraPlotter | None]
        use_zoh: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        use_datashader: NotRequired[bool]
        fig_ax: NotRequired[tuple[tuple[Figure, Axes], ...] | None]
        plot_type: NotRequired[str]


# %% Functions - plot_bpe_convergence
def plot_bpe_convergence(
    costs: Sequence,
    *,
    opts: Opts | None = None,
    fig_ax: tuple[Figure, Axes] | None = None,
    skip_setup_plots: bool = False,
) -> Figure:
    r"""
    Plot the BPE convergence rate by iteration on a log scale.

    Parameters
    ----------
    costs : array_like
        Costs for the beginning run, each iteration, and final run
    opts : class Opts, optional
        Plotting options
    fig_ax : (fig, ax) tuple, optional
        Figure and axis to use, otherwise create new ones
    skip_setup_plots : bool, default is False
        Whether to skip the setup_plots step within this function

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in July 2016.

    Examples
    --------
    >>> from dstauffman.plotting import plot_bpe_convergence
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> costs = np.array([1, 0.1, 0.05, 0.01])
    >>> fig = plot_bpe_convergence(costs)

    Close plot
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()

    # get number of iterations
    num_iters = len(costs) - 2
    time = np.arange(len(costs)) if HAVE_NUMPY else list(range(len(costs)))
    labels = ["Begin"] + [str(x + 1) for x in range(num_iters)] + ["Final"]

    # alias the title
    this_title = "Convergence by Iteration"
    if fig_ax is None:
        # create the figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        (fig, ax) = fig_ax
    assert fig.canvas.manager is not None
    if (sup := fig._suptitle) is None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
        fig.canvas.manager.set_window_title(this_title)
    else:
        fig.canvas.manager.set_window_title(sup.get_text())
    # add an axis and plot the data
    ax.semilogy(time, costs, "b.-", linewidth=2)
    # add labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(this_title)
    ax.set_xticks(time)
    if len(costs) > 0:
        ax.set_xticklabels(labels)
    # show a grid
    ax.grid(True)
    # Setup plots
    if not skip_setup_plots:
        setup_plots(fig, opts)
    return fig


# %% plot_bpe_results
def plot_bpe_results(  # noqa: C901
    bpe_results: BpeResults,
    *,
    opts: Opts | None = None,
    plots: dict[str, bool] | None = None,
    **kwargs: Unpack[_BpeKwArgs],
) -> _Figs:
    r"""Plot the results of estimation."""
    # hard-coded options
    label_values = False

    # alias the names
    if bpe_results.param_names is not None:
        names = [name.decode("utf-8") for name in bpe_results.param_names]
    else:
        names = []

    # defaults for which plots to make
    default_plots = {"innovs": False, "convergence": False, "correlation": False, "info_svd": False, "covariance": False}

    # check for optional variables
    if opts is None:
        opts = Opts()
    if plots is None:
        plots = default_plots
    else:
        # check for unexpected keys
        for key in plots:
            if key not in default_plots:
                raise ValueError(f'Unexpected plotting option: "{key}".')
        # overlay any missing default plots
        missing_keys = set(default_plots.keys()) - set(plots.keys())
        for key in missing_keys:
            plots[key] = default_plots[key]

    # colormap information
    kw_colormap = kwargs.pop("colormap", None)

    # preallocate output
    figs: _Figs = []

    # time based plots
    if plots["innovs"]:
        if bpe_results.begin_innovs is not None and bpe_results.final_innovs is not None:
            time = np.arange(len(bpe_results.begin_innovs))
            data = np.vstack((bpe_results.begin_innovs, bpe_results.final_innovs))
            colormap = kw_colormap if kw_colormap is not None else "bwr_r"
            temp_opts = opts.__class__(opts)
            temp_opts.disp_xmin = temp_opts.disp_xmax = temp_opts.rms_xmin = temp_opts.rms_xmax = None
            new_figs = plot_time_history(  # type: ignore[misc]
                "Innovs Before and After",
                time,
                data,
                opts=temp_opts,
                elements=["Before", "After"],
                colormap=colormap,
                skip_setup_plots=True,
                **kwargs,
            )
            for fig in new_figs:
                fig.axes[0].set_xlabel("Innovation Number")
            setup_plots(new_figs, opts=temp_opts)
            figs += new_figs
        else:
            print("Data isn't available for Innovations plot.")
    if plots["convergence"]:
        if len(bpe_results.costs) != 0:
            fig = plot_bpe_convergence(bpe_results.costs, opts=opts)
            figs.append(fig)
        else:
            print("Data isn't available for convergence plot.")

    if plots["correlation"]:
        if bpe_results.correlation is not None:
            colormap = kw_colormap if kw_colormap is not None else "bwr"
            fig = plot_correlation_matrix(
                bpe_results.correlation,
                labels=names,
                opts=opts,
                matrix_name="Correlation Matrix",
                cmin=-1,
                plot_lower_only=True,
                label_values=label_values,
                colormap=colormap,
            )
            figs.append(fig)
        else:
            print("Data isn't available for correlation plot.")

    if plots["info_svd"]:
        if bpe_results.info_svd is not None:
            colormap = kw_colormap if kw_colormap is not None else "cool"
            fig = plot_correlation_matrix(
                np.abs(bpe_results.info_svd),
                opts=opts,
                cmin=0,
                matrix_name="Information SVD Matrix",
                label_values=label_values,
                labels=[[f"{i + 1}" for i in range(len(names))], names],
                colormap=colormap,
            )
            figs.append(fig)
        else:
            print("Data isn't available for information SVD plot.")

    if plots["covariance"]:
        if bpe_results.covariance is not None:
            max_mag = np.nanmax(np.abs(bpe_results.covariance))
            colormap = kw_colormap if kw_colormap is not None else "bwr"
            fig = plot_correlation_matrix(
                bpe_results.covariance,
                labels=names,
                opts=opts,
                matrix_name="Covariance Matrix",
                cmin=-max_mag,
                cmax=max_mag,
                plot_lower_only=True,
                label_values=label_values,
                colormap=colormap,
            )
            figs.append(fig)
        else:
            print("Data isn't available for covariance plot.")
    return figs


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_plotting_batch", exit=False)
    doctest.testmod(verbose=False)
