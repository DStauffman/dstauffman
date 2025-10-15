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
from typing import TYPE_CHECKING, TypedDict

try:
    from typing import NotRequired, Unpack
except ImportError:
    from typing_extensions import NotRequired, Unpack  # for Python v3.10
import unittest

from dstauffman import DEG2RAD, get_data_dir, HAVE_MPL, HAVE_NUMPY, M2FT, RAD2DEG, unit
from dstauffman.aerospace import EARTH, geod2ecf, sph2cart
from dstauffman.plotting.plotting import ColorMap, ExtraPlotter, Opts, setup_plots

if HAVE_MPL:
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _F = float | np.floating
    _N = NDArray[np.floating]
    _M = NDArray[np.floating]  # 2D

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
    with filename.open(encoding="utf-8") as file:
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
def plot_map(
    map_data: dict[str, _M] | None = None,
    lat: _N | None = None,
    lon: _N | None = None,
    *,
    title: str = "Ground Track",
    units: str = "deg",
    latlon_units: str = "rad",
    opts: Opts | None = None,
    xy_events: tuple[tuple[_F, _F], ...] | list[tuple[_F, _F]] | None = None,
    xy_annotations: tuple[str, ...] | list[str] | None = None,
    xy_colors: tuple[str, ...] | list[str] | None = None,
    land_colors: str = "same",
    color_by: tuple[str, _N] | None = None,
    cbar_colormap: str | None | ColorMap = None,
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
    title : str, optional, default is "Orbit"
        Title to put on the plot
    units : str, optional, default is "deg", from {"deg", "rad"}
        Units for map data
    latlon_units : str, optional, default is "rad", from {"deg", "rad"}
        Units for lattitude and longitude inputs
    opts : class Opts
        Plotting options
    xy_events : iterable of tuple[float, float], optional
        Series of lat/lon event locations
    xy_annotations : iterable of str, optional
        Series of event labels
    xy_colors : iterable of str, optional
        Series of event color strings
    land_colors : str, optional, default is "same"
        Whether to color all the countries the same or no colors or multicolored
    color_by : tuple[str, _N], optional
        What data to color the orbit by, usually altitude
    cbar_colormap: str| ColorMap, optional
        Colormap used to color the orbit
    map_labels : dict[str, tuple[float, float]], optional
        Dict of label and lat/lon positions
    map_colors : dict[str, int | str], optional
        Colors or index to colors for each country location
    dir_skip : int, optional
        How many points to skip when subsampling the orbit
    extra_plotter : protocol ExtraPlotter
        A function to pass into plotting routines to plot vertical lines at the given events
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots call
    **kwargs : dict
        Additional arguments for saving, classification, and map options

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
    latlon_scale = (1.0 if latlon_units == "deg" else RAD2DEG) if units == "deg" else 1.0 if latlon_units == "rad" else DEG2RAD

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
    assert fig.canvas.manager is not None
    fig.canvas.manager.set_window_title(title)
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
            xy_event = (float(latlon_scale * xy_events[i][0]), float(latlon_scale * xy_events[i][1]))
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


# %% Functions - plot_map_3d
def plot_map_3d(
    map_data: dict[str, _M] | None = None,
    lat: _N | None = None,
    lon: _N | None = None,
    alt: _N | None = None,
    *,
    title: str = "Orbit",
    units: str = "deg",
    latlon_units: str = "rad",
    earth_units: str = "m",
    opts: Opts | None = None,
    lla_events: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] | None = None,
    lla_annotations: tuple[str, ...] | list[str] | None = None,
    lla_colors: tuple[str, ...] | list[str] | None = None,
    land_colors: str = "same",
    color_by: tuple[str, _N] | None = None,
    cbar_colormap: str | None | ColorMap = None,
    map_labels: dict[str, tuple[float, float]] | None = None,
    map_colors: dict[str, int | str] | None = None,
    dir_skip: int | None = None,
    extra_plotter: ExtraPlotter | None = None,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_PlotMapKwargs],
) -> Figure:
    """
    Plots the given 3D map.

    Parameters
    ----------
    map_data : dict by name of Lat/Lon data
        Map data to plot
    lat : (N,)
        Additional Latitude data to plot, optional [rad]
    lon : (N,)
        Additional Longitude data to plot, optional [rad]
    alt : (N,)
        Additional altitude data to plot, optional [m]
    title : str, optional, default is "Orbit"
        Title to put on the plot
    units : str, optional, default is "deg", from {"deg", "rad"}
        Units for map data
    latlon_units : str, optional, default is "rad", from {"deg", "rad"}
        Units for lattitude and longitude inputs
    earth_units : str, optional, default is "m"
        Units for altitude
    opts : class Opts
        Plotting options
    lla_events : iterable of tuple[float, float, float], optional
        Series of lat/lon/alt event locations
    lla_annotations : iterable of str, optional
        Series of event labels
    lla_colors : iterable of str, optional
        Series of event color strings
    land_colors : str, optional, default is "same"
        Whether to color all the countries the same or no colors or multicolored
    color_by : tuple[str, _N], optional
        What data to color the orbit by, usually altitude
    cbar_colormap: str| ColorMap, optional
        Colormap used to color the orbit
    map_labels : dict[str, tuple[float, float]], optional
        Dict of label and lat/lon positions
    map_colors : dict[str, int | str], optional
        Colors or index to colors for each country location
    dir_skip : int, optional
        How many points to skip when subsampling the orbit
    extra_plotter : protocol ExtraPlotter
        A function to pass into plotting routines to plot vertical lines at the given events
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots call
    **kwargs : dict
        Additional arguments for saving, classification, and map options

    Returns
    -------
    fig : matplot.figure.Figure
        Map figure

    Notes
    -----
    #.  This 3D map has some major limitations with clipping and such. Consider it experimental.

    Examples
    --------
    >>> from dstauffman.plotting import plot_map_3d
    >>> fig = plot_map_3d()  # doctest: +SKIP

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
    assert earth_units in {"m", "ft", "EU", "km"}  # TODO: implement these
    map_scale = DEG2RAD if units == "deg" else 1.0
    latlon_scale = DEG2RAD if latlon_units == "deg" else 1.0
    if earth_units == "m":
        earth_radius = EARTH["a"]
    elif earth_units == "ft":
        earth_radius = M2FT * EARTH["a"]
    elif earth_units == "EU":
        earth_radius = 1.0
    elif earth_units == "km":
        earth_radius = 1e-3 * EARTH["a"]
    else:
        raise ValueError(f"Unexpected option for earth_units: {earth_units}")

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
    map_lat = [map_scale * map_data[name][0, ...] for name in land_names]
    map_lon = [map_scale * map_data[name][1, ...] for name in land_names]

    fig = plt.figure()
    assert fig.canvas.manager is not None
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
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
    # Plot globe
    az, el = np.mgrid[0.0 : 2 * np.pi : 24j, -np.pi : np.pi : 12j]  # type: ignore[misc]
    x, y, z = sph2cart(az, el, 0.9999 * earth_radius)
    for i in range(x.shape[0] - 1):  # type: ignore[attr-defined]
        for j in range(x.shape[1] - 1):  # type: ignore[attr-defined]
            indices = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1), (i, j)]
            verts = np.array([[x[k], y[k], z[k]] for k in indices])  # type: ignore[index]
            poly = Poly3DCollection(verts[np.newaxis, :, :], facecolor=background, edgecolor="none", zorder=1)
            ax.add_collection3d(poly)

    # Plot land borders
    colors = tuple(land_cmap.get_color(x) for x in range(len(land_names)))
    for la, lo, c in zip(map_lat, map_lon, colors):
        x, y, z = geod2ecf(la, lo, np.zeros_like(la), output="split")  # type: ignore[assignment]
        verts = [list(zip(x, y, z))]  # type: ignore[assignment, call-overload]
        poly = Poly3DCollection(verts, color=c, edgecolor=border, alpha=0.9, linewidth=linewidth, zorder=2)
        ax.add_collection3d(poly)
    ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel(f"X [{earth_units}]")
    ax.set_ylabel(f"Y [{earth_units}]")
    ax.set_zlabel(f"Z [{earth_units}]")

    # Plot ground/orbit track
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
        if alt is None:
            alt = np.zeros_like(lat)
        x, y, z = geod2ecf(lat, lon, alt, output="split")  # type: ignore[assignment]
        ax.scatter(x, y, z, s=10, c=colors, label="Trace", zorder=6)
        if dir_skip is not None:
            ang = unit(np.vstack([np.diff(lon), np.diff(lat), np.diff(alt) / earth_radius]), axis=0)
            ax.quiver(
                latlon_scale * lon[0:-1:dir_skip],
                latlon_scale * lat[0:-1:dir_skip],
                alt[0:-1:dir_skip],
                ang[0, ::dir_skip],
                ang[1, ::dir_skip],
                ang[2, ::dir_skip],
                zorder=8,
                pivot="tail",
            )
        if color_by is not None:
            cbar = fig.colorbar(cbar_cmap.get_smap(), ax=ax, orientation="horizontal")
            cbar.ax.set_ylabel(cbar_title)

    # Plot specific events
    if lla_events is not None:
        for i in range(len(lla_events)):  # pylint: disable=consider-using-enumerate
            event_x, event_y, event_z = geod2ecf(np.asanyarray(lla_events[i]), output="split")
            lla_color = lla_colors[i] if lla_colors is not None else "xkcd:green"
            lla_annot = lla_annotations[i] if lla_annotations is not None else None
            ax.scatter(event_x, event_y, event_z, color=lla_color, sizes=[50], zorder=9)
            if lla_annot is not None:
                ax.text(event_x, event_y, event_z, lla_annot, zdir="z", fontweight="bold", fontsize=12, zorder=9)
    if map_labels is not None:
        for map_label, map_latlon in map_labels.items():
            map_x, map_y, map_z = geod2ecf(DEG2RAD * map_latlon[0], DEG2RAD * map_latlon[1], 0.0, output="split")
            ax.text(map_x, map_y, map_z, map_label, zdir="x", fontsize=8, horizontalalignment="center", zorder=3)

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
