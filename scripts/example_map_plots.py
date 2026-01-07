"""Earth Plotting Examples."""

# %% Imports
from collections import defaultdict

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

import dstauffman as dcs
import dstauffman.aerospace as space
import dstauffman.plotting as plot


# %% Functions - _extra_plotter_factory
def _extra_plotter_factory() -> plot.ExtraPlotter:
    """Create a plotter to highlight the zero latitude and longitudes."""

    def _extra_plotter(fig: Figure, ax: list[Axes]) -> None:  # pylint: disable=unused-argument  # noqa: ARG001
        for this_axes in ax:
            this_axes.axvline(0, color="xkcd:brick red", linewidth=2, label="Prime Meridian")
            this_axes.axhline(0, color="xkcd:dark red", linewidth=2, label="Equator")

    return _extra_plotter


# %% Script
if __name__ == "__main__":
    # simple Earth plot
    fig1 = plot.plot_map()

    # create Opts
    opts = plot.Opts()

    # read ISS two line elements
    line1 = "1 25544U 98067A   24339.76032953  .00017305  00000-0  30797-3 0  9991"
    line2 = "2 25544  51.6387 190.3157 0006984 301.4763 161.1609 15.50217009485000"
    oe = space.two_line_elements(line1, line2)
    dt = 15 * dcs.NP_ONE_SECOND
    time = np.datetime64("2024-12-05T00:20:59", dcs.NP_DATETIME_UNITS) + np.arange(0 * dcs.NP_ONE_SECOND, oe.T * dcs.NP_ONE_SECOND + dt, dt)  # type: ignore[arg-type]
    mu = 3.9863e14

    delta_time_sec = (time - time[0]) / dcs.NP_ONE_SECOND  # TODO: should be oe.t?
    new_oe = space.advance_elements(oe, mu, delta_time_sec)
    pos_eci, _ = space.oe_2_rv(new_oe)

    time_jd = space.numpy_to_jd(time)
    I2F = space.quat_eci_2_ecf(time_jd)
    pos_ecf = space.quat_times_vector(I2F, pos_eci)

    lat, lon, alt = space.ecf2geod(pos_ecf, output="split")

    # no colors, with ground track
    fig2 = plot.plot_map(None, lat, lon, land_colors="none")

    # ground track colored by altitude, multicolor countries
    map_data, map_labels, map_colors = plot.get_map_data()
    color_by = ("Altitude [km]", 1e-3 * alt)
    fig3 = plot.plot_map(map_data, lat, lon, land_colors="multi", color_by=color_by, dir_skip=12, title="ISS orbit")

    # ground track with event
    xy_events: tuple[tuple[float, float], ...] = ((lon[10], lat[10]),)
    xy_annotations: tuple[str, ...] = ("ISS",)
    fig4 = plot.plot_map(None, lat, lon, land_colors="same", color_by=color_by, cbar_colormap="spring",
        title="Event", xy_events=xy_events, xy_annotations=xy_annotations)  # fmt: skip

    # ground track with multiple events and custom colors
    m = len(lat) // 2
    xy_events = ((lon[0], lat[0]), (lon[m], lat[m]))
    xy_annotations = ("Rev Start", "Rev Middle")
    xy_colors = ("xkcd:green", "xkcd:red")
    map_colors2: dict[str, str | int] = defaultdict(lambda: "xkcd:green")
    map_colors2["United States of America"] = "xkcd:white"
    map_colors2["Mexico"] = 0
    map_colors2["Canada"] = 5
    fig5 = plot.plot_map(None, lat, lon, land_colors="multi", color_by=color_by, map_colors=map_colors2,
        title="Custom Colors", xy_events=xy_events, xy_annotations=xy_annotations, xy_colors=xy_colors,
        background="xkcd:cyan", border="xkcd:yellow", land_colormap="tab10")  # fmt: skip

    # map with labels and extra plotter
    extra_plotter = _extra_plotter_factory()
    fig6 = plot.plot_map(map_data, land_colors="multi", map_labels=map_labels, extra_plotter=extra_plotter)
