"""Plot some quaternion differences."""

# %% Imports
import numpy as np

from dstauffman.aerospace import quat_from_euler, quat_mult, quat_norm
from dstauffman.plotting import COLOR_LISTS, ColorMap, Opts, plot_quaternion, plot_time_difference

# %% Script
if __name__ == "__main__":
    # build data
    q1 = quat_norm(np.array([0.1, -0.2, 0.3, 0.4]))
    dq = quat_from_euler(1e-6 * np.array([-300, 100, 200]), [3, 1, 2])
    q2 = quat_mult(dq, q1)

    time_one = np.arange(11)
    quat_one = np.tile(q1[:, np.newaxis], (1, time_one.size))

    time_two = np.arange(2, 13)
    quat_two = np.tile(q2[:, np.newaxis], (1, time_two.size))
    quat_two[3, 4] += 50e-6
    quat_two = quat_norm(quat_two)

    # plotting options
    opts = Opts()
    opts.case_name = "test_plot"
    opts.quat_comp = False
    opts.sub_plots = False
    opts.sing_line = True
    opts.names = ["KF1", "KF2"]

    # make plots
    figs1 = plot_quaternion("Quaternion", time_one, time_two, quat_one, quat_two, opts=opts)  # type: ignore[call-overload]

    figs2 = plot_time_difference(  # type: ignore[call-overload]
        "State Differences",
        time_one,
        quat_one,
        time_two,
        quat_two,
        opts=opts,
        elements=("X", "Y", "Z", "S"),
        colormap=COLOR_LISTS["quat_comp"],
        units="rad",
        second_units="micro",
    )
