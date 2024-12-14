r"""
Plotting functions related to displaying Earth maps.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
#.  Incorporated into dstauffman by David C. Stauffer in December 2024.
"""

# %% Imports
from __future__ import annotations

import doctest
import json
from pathlib import Path
from typing import NotRequired, TYPE_CHECKING, TypedDict, Unpack
import unittest

from dstauffman import DEG2RAD, get_data_dir, HAVE_MPL, HAVE_NUMPY, RAD2DEG, unit
from dstauffman.plotting.plotting import ColorMap, ExtraPlotter, Opts, setup_plots

if HAVE_MPL:
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _N = NDArray[np.float64]
    _M = NDArray[np.float64]  # 2D

    class _PlotMapKwargs(TypedDict):
        save_plot: NotRequired[bool]
        save_path: NotRequired[Path | None]
        classify: NotRequired[str]
        background: NotRequired[str]
        border: NotRequired[str]
        land_colormap: NotRequired[str]


# %% Functions - get_map_data
def get_map_data(source: str = "110m") -> tuple[dict[str, _M], dict[str, tuple[float, float]], dict[str, int]]:
    """
    Loads data for coastlines from one of the specified sources.

    Examples
    --------
    >>> from dstauffman.plotting import get_map_data
    >>> map_data, map_labels, map_colors = get_map_data()
    >>> print(sorted(map_data.keys())[:5])
    ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antarctica']

    """
    if source == "110m":
        filename = get_data_dir() / "earth_110m.geo.json"
    elif source == "50m":
        filename = get_data_dir() / "earth_50m.geo.json"
    else:
        raise ValueError(f"Unknown data source: {source}")
    with open(filename, encoding="utf-8") as file:
        raw_data = json.load(file)
    features = raw_data["features"]

    map_data: dict[str, _M] = {}
    map_labels: dict[str, tuple[float, float]] = {}
    map_colors: dict[str, int] = {}
    nan2 = np.array([np.nan, np.nan])
    for this in features:
        name = this["properties"]["name_en"]
        data = this["geometry"]["coordinates"]
        latlon: list[_N] = []
        for this_data in data:
            latlon.append(nan2)
            latlon.append(np.squeeze(np.asarray(this_data)))
        label_x = this["properties"]["label_x"]
        label_y = this["properties"]["label_y"]
        color = this["properties"]["mapcolor9"]
        temp = np.vstack(latlon)
        map_data[name] = np.vstack([temp[:, 1], temp[:, 0]])
        map_labels[name] = (label_x, label_y)
        map_colors[name] = color

    # TODO: add return options?
    return map_data, map_labels, map_colors


# %% Functions - plot_map
def plot_map(  # noqa: C901
    map_data: dict[str, _M] | None = None,
    lat: _N | None = None,
    lon: _N | None = None,
    *,
    title: str = "Ground Track",
    units: str = "deg",
    latlon_units: str = "rad",
    opts: Opts | None = None,
    xy_events: tuple[tuple[float, float], ...] | None = None,
    xy_annotations: tuple[str, ...] | None = None,
    xy_colors: tuple[str, ...] | None = None,
    land_colors: str = "same",
    color_by: tuple[str, _N] | None = None,
    cbar_colormap: str | None = None,
    map_labels: dict[str, tuple[float, float]] | None = None,
    map_colors: dict[str, int | str] | None = None,
    dir_skip: int | None = None,
    extra_plotter: ExtraPlotter | None = None,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_PlotMapKwargs],
) -> Figure:
    """
    Plots the given map.

    Parameters
    ----------
    map_data : dict by name of Lat/Lon data
        Map data to plot
    lat : (N, )
        Additional Latitude track to plot, optional [rad]
    lon : (N, )
        Additional Longitude track to plot, optional [rad]

    Returns
    -------
    fig : matplot.figure.Figure
        Map figure

    Examples
    --------
    >>> from dstauffman.plotting import plot_map
    >>> fig = plot_map()  # doctest: +SKIP

    """
    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # checks
    assert units in {"deg", "rad"}, "Unexpected units"
    assert latlon_units in {"deg", "rad"}, "Unexpected lat/lon units"
    if units == "deg":
        latlon_scale = 1.0 if latlon_units == "deg" else RAD2DEG
    else:
        latlon_scale = 1.0 if latlon_units == "rad" else DEG2RAD

    # load data
    if map_data is None:
        map_data, _, _map_colors = get_map_data()
    else:
        _map_colors = None
    if map_colors is None:
        if _map_colors is None:
            _, _, map_colors = get_map_data()  # type: ignore[assignment]
        else:
            map_colors = _map_colors  # type: ignore[assignment]
    assert map_colors is not None
    land_names = list(map_data.keys())
    map_lat = [map_data[name][0, ...] for name in land_names]
    map_lon = [map_data[name][1, ...] for name in land_names]

    fig = plt.figure()
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(title)
    ax = fig.add_subplot(1, 1, 1)
    if land_colors == "none":
        land_cmap = ColorMap(ListedColormap((kwargs.pop("land_colormap", "xkcd:white"),)))
        background = kwargs.pop("background", "xkcd:white")
        border = kwargs.pop("border", "C0")
        linewidth = 0.3
    elif land_colors == "same":
        land_cmap = ColorMap(ListedColormap((kwargs.pop("land_colormap", "xkcd:light green"),)))
        background = kwargs.pop("background", "xkcd:pale blue")
        border = kwargs.pop("border", "xkcd:green")
        linewidth = 0.3
    elif land_colors == "multi":
        land_colormap = kwargs.pop("land_colormap", "rainbow_r")  # use tab10 as default?
        land_10_colors = ColorMap(num_colors=10, colormap=land_colormap)
        list_colors: list[str | tuple[float, float, float, float]] = []
        for name in land_names:
            if isinstance(mcolor := map_colors[name], str):
                list_colors.append(mcolor)
            else:
                list_colors.append(land_10_colors.get_color(mcolor))
        land_cmap = ColorMap(ListedColormap(list_colors), num_colors=len(land_names))
        background = kwargs.pop("background", "xkcd:pale blue")
        border = kwargs.pop("border", "xkcd:black")
        linewidth = 0.3
    else:
        raise ValueError(f"Unexpected value for land_colors: {land_colors}")
    ax.set_facecolor(background)
    colors = tuple(land_cmap.get_color(x) for x in range(len(land_names)))
    for y, x, c in zip(map_lat, map_lon, colors):
        ax.fill(x, y, color=c, edgecolor=border, linewidth=linewidth)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Longitude [{units}]")
    ax.set_ylabel(f"Latitude [{units}]")

    if lat is not None and lon is not None:
        if color_by is not None:
            cbar_title = color_by[0]
            cbar_data = color_by[1]
            cbar_cmap = ColorMap(
                colormap="autumn_r" if cbar_colormap is None else cbar_colormap,
                low=float(np.nanmin(cbar_data)),
                high=float(np.nanmax(cbar_data)),
            )
            colors = tuple(cbar_cmap.get_color(x) for x in cbar_data)
        else:
            colors = "xkcd:orange"  # type: ignore[assignment]
        ax.scatter(latlon_scale * lon, latlon_scale * lat, s=10, c=colors, label="Trace", zorder=6)
        if dir_skip is not None:
            ang = unit(np.vstack([np.diff(lon), np.diff(lat)]), axis=0)
            ax.quiver(
                latlon_scale * lon[0:-1:dir_skip],
                latlon_scale * lat[0:-1:dir_skip],
                ang[0, ::dir_skip],
                ang[1, ::dir_skip],
                zorder=8,
                angles="xy",
                units="dots",
                scale_units="xy",
                scale=0.5,
                minshaft=0.2,
                minlength=0.2,
                pivot="tail",
                width=3,
                headwidth=4.5,
                headlength=4.5,
                headaxislength=4.5,
            )
            # pivot="tail", headwidth=3, headlength=5, headaxislength=4.5)
        if color_by is not None:
            cbar = fig.colorbar(cbar_cmap.get_smap(), ax=ax, orientation="horizontal")
            cbar.ax.set_ylabel(cbar_title)
    if xy_events is not None:
        for i in range(len(xy_events)):  # pylint: disable=consider-using-enumerate
            xy_event = tuple(np.multiply(latlon_scale, xy_events[i]))
            xy_color = xy_colors[i] if xy_colors is not None else "xkcd:green"
            xy_annot = xy_annotations[i] if xy_annotations is not None else None
            circle = plt.Circle(xy_event, 5.0, color=xy_color, fill=False, linewidth=3.0, zorder=10)
            ax.add_patch(circle)
            if xy_annot is not None:
                ax.annotate(xy_annot, xy=xy_event, fontweight="bold", fontsize=12)
    if map_labels is not None:
        for map_label, map_xy in map_labels.items():
            ax.annotate(map_label, xy=map_xy, fontsize=8, horizontalalignment="center")

    # set limits
    ax.set_xlim((-180.0, 180.0) if units == "deg" else (-np.pi, np.pi))
    ax.set_ylim((-90.0, 90.0) if units == "deg" else (-np.pi / 2, np.pi / 2))

    # plot any extra information through a generic callable
    if extra_plotter is not None:
        extra_plotter(fig=fig, ax=[ax])

    # call advanced tools
    if not skip_setup_plots:
        setup_plots(fig, this_opts)

    return fig


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_plotting_maps", exit=False)
    doctest.testmod(verbose=False)
