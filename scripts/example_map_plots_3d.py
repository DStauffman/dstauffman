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
        az = np.linspace(0, 2 * np.pi, 100)
        el = np.zeros_like(az)
        rad = np.full(az.shape, space.EARTH["a"])
        x1, y1, z1 = space.sph2cart(az, el, rad)
        x2, y2, z2 = space.sph2cart(el, az, rad)
        for this_axes in ax:
            this_axes.plot(x1, y1, z1, color="xkcd:brick red", linewidth=2, label="Prime Meridian")
            this_axes.plot(x2, y2, z2, color="xkcd:dark red", linewidth=2, label="Equator")

    return _extra_plotter


# %% Script
if __name__ == "__main__":
    # simple Earth plot
    fig1 = plot.plot_map_3d()

    # create Opts
    opts = plot.Opts()

    # read ISS two line elements
    line1 = "1 25544U 98067A   24339.76032953  .00017305  00000-0  30797-3 0  9991"
    line2 = "2 25544  51.6387 190.3157 0006984 301.4763 161.1609 15.50217009485000"
    oe = space.two_line_elements(line1, line2)
    t = np.datetime64("2024-12-05T00:20:59", dcs.NP_DATETIME_UNITS)
    dt = 15 * dcs.NP_ONE_SECOND
    time = t + np.arange(0, oe.T * dcs.NP_ONE_SECOND + dt, dt)
    mu = 3.9863e14

    delta_time_sec = (time - time[0]) / dcs.NP_ONE_SECOND  # TODO: should be oe.t?
    new_oe = space.advance_elements(oe, mu, delta_time_sec)
    pos_eci, _ = space.oe_2_rv(new_oe)

    lat, lon, alt = space.ecf2geod(pos_eci, output="split")

    # no colors, with orbit
    fig2 = plot.plot_map_3d(None, lat, lon, land_colors="none")

    # orbit colored by altitude, multicolor countries
    map_data, map_labels, map_colors = plot.get_map_data()
    color_by = ("Altitude [km]", 1e-3 * alt)
    fig3 = plot.plot_map_3d(map_data, lat, lon, alt, land_colors="multi", color_by=color_by, dir_skip=12, title="ISS orbit")

    # orbit with event (satellite location)
    lla_events: tuple[tuple[float, float, float], ...] = ((lon[10], lat[10], alt[10]),)
    lla_annotations: tuple[str, ...] = ("ISS",)
    fig4 = plot.plot_map_3d(None, lat, lon, alt, land_colors="same", color_by=color_by, cbar_colormap="spring",
        title="Event", lla_events=lla_events, lla_annotations=lla_annotations)  # fmt: skip

    # orbit with multiple events and custom colors
    m = len(lat) // 2
    lla_events = ((lon[0], lat[0], alt[0]), (lon[m], lat[m], alt[m]))
    lla_annotations = ("Rev Start", "Rev Middle")
    lla_colors = ("xkcd:green", "xkcd:red")
    map_colors2: dict[str, str | int] = defaultdict(lambda: "xkcd:green")
    map_colors2["United States of America"] = "xkcd:white"
    map_colors2["Mexico"] = 0
    map_colors2["Canada"] = 5
    fig5 = plot.plot_map_3d(None, lat, lon, alt, land_colors="multi", color_by=color_by, map_colors=map_colors2,
        title="Custom Colors", lla_events=lla_events, lla_annotations=lla_annotations, lla_colors=lla_colors,
        background="xkcd:cyan", border="xkcd:yellow", land_colormap="tab10")  # fmt: skip

    # map with labels and extra plotter
    extra_plotter = _extra_plotter_factory()
    fig6 = plot.plot_map_3d(map_data, land_colors="multi", map_labels=map_labels, extra_plotter=extra_plotter)
