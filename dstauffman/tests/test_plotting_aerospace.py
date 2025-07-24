r"""
Test file for the `aerospace` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.

"""

# %% Imports
import unittest
from unittest.mock import patch
import warnings

from slog import LogLevel

from dstauffman import HAVE_MPL, HAVE_NUMPY
from dstauffman.aerospace import Kf
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np


# %% plotting.minimize_names
class Test_minimize_names(unittest.TestCase):
    r"""
    Tests the plotting.minimize_names function with the following cases:
        Nominal with numbers
    """

    def test_nominal(self) -> None:
        out = plot.minimize_names(("1", "2", "3"))
        self.assertEqual(out, "1,2,3")

    def test_single(self) -> None:
        out = plot.minimize_names(("Single Name",))
        self.assertEqual(out, "Single Name")

    def test_same_names(self) -> None:
        names = ("Gyro Bias", "Gyro Bias", "Gyro Bias")
        exp = "Gyro Bias,,"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_similar_names(self) -> None:
        names = ("Gyro Bias", "Gyro Bias ARW", "Gyro Bias RRW")
        exp = "Gyro Bias, ARW, RRW"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_almost_same_names(self) -> None:
        names = ("Attitude X", "Attitude Y", "Attitude Z")
        exp = "Attitude X,Y,Z"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_bad_numbers(self) -> None:
        out = plot.minimize_names(("11", "12", "13"))
        self.assertEqual(out, "11,12,13")

    def test_combine(self) -> None:
        names = ("Gyro Bias 1", "Gyro Bias 2", "Gyro Bias 444")
        exp = "Gyro Bias 1,2,444"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_bad_combine(self) -> None:
        names = ("Gyro Bias 21", "Gyro Bias 22", "Gyro Bias 24")
        exp = "Gyro Bias 21,22,24"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_mixed(self) -> None:
        names = ("Gyro Bias 1", "Gyro Bias 2", "Gyro Bias 3", "Gyro Drift 1", "Gyro Drift 5")
        exp = "Gyro Bias 1,Bias 2,Bias 3,Drift 1,Drift 5"
        out = plot.minimize_names(names)
        self.assertEqual(out, exp)

    def test_seperator(self) -> None:
        names = ("Gyro Bias 1", "Gyro Bias 2", "Gyro Bias 4", "Gyro Bias 5")
        exp = "Gyro Bias 1||2||4||5"
        out = plot.minimize_names(names, sep="||")
        self.assertEqual(out, exp)

    def test_no_matches(self) -> None:
        names = ("Bias 1", "Scale Factor 2", "Misalignment 3")
        exp = "Bias 1, Scale Factor 2, Misalignment 3"
        out = plot.minimize_names(names, sep=", ")
        self.assertEqual(out, exp)


# %% plotting.plot_quaternion
class Test_plotting_plot_quaternion(unittest.TestCase):
    r"""
    Tests the plotting.plot_quaternion function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_attitude
class Test_plotting_plot_attitude(unittest.TestCase):
    r"""
    Tests the plotting.plot_attitude function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_los
class Test_plotting_plot_los(unittest.TestCase):
    r"""
    Tests the plotting.plot_los function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_position
class Test_plotting_plot_position(unittest.TestCase):
    r"""
    Tests the plotting.plot_position function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_velocity
class Test_plotting_plot_velocity(unittest.TestCase):
    r"""
    Tests the plotting.plot_velocity function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_innovations
class Test_plotting_plot_innovations(unittest.TestCase):
    r"""
    Tests the plotting.plot_innovations function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_innov_fplocs
class Test_plotting_plot_innov_fplocs(unittest.TestCase):
    r"""
    Tests the plotting.plot_innov_fplocs function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_innov_hist
class Test_plotting_plot_innov_hist(unittest.TestCase):
    r"""
    Tests the plotting.plot_innov_hist function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_covariance
class Test_plotting_plot_covariance(unittest.TestCase):
    r"""
    Tests the plotting.plot_covariance function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.plot_states
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_states(unittest.TestCase):
    r"""
    Tests the plotting.plot_states function with the following cases:
        Single Kf structure
        Comparison
        Comparison with mismatched states
        Error output
    """

    def setUp(self) -> None:
        self.gnd1 = Kf(name="GARSE", num_points=30, num_states=9, active_states=np.arange(9))
        self.gnd2 = Kf(name="GBARS", num_points=60, num_states=8, active_states=np.array([0, 1, 2, 3, 5, 6, 7, 8]))
        self.gnd1.time[:] = np.arange(30.0)  # type: ignore[index]
        self.gnd2.time[:] = np.arange(60.0) - 10.0  # type: ignore[index]
        self.gnd1.state.fill(1.0)  # type: ignore[union-attr]
        self.gnd2.state.fill(0.99)  # type: ignore[union-attr]
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.figs: list[Figure] = []

    def test_single(self) -> None:
        with patch("dstauffman.plotting.aerospace.logger") as mock_logger:
            self.figs += plot.plot_states(self.gnd1, opts=self.opts)  # type: ignore[call-overload]
        mock_logger.log.assert_any_call(LogLevel.L4, "Plotting %s plots ...", "State Estimates")
        mock_logger.log.assert_called_with(LogLevel.L4, "... done.")

    def test_comp(self) -> None:
        with patch("dstauffman.plotting.aerospace.logger") as mock_logger:
            self.figs += plot.plot_states(self.gnd1, self.gnd2, opts=self.opts)  # type: ignore[call-overload]
        mock_logger.log.assert_any_call(LogLevel.L4, "Plotting %s plots ...", "State Estimates")
        mock_logger.log.assert_called_with(LogLevel.L4, "... done.")

    def test_errs(self) -> None:
        with patch("dstauffman.plotting.aerospace.logger") as mock_logger:
            (figs, err) = plot.plot_states(self.gnd1, self.gnd2, opts=self.opts, return_err=True)  # type: ignore[call-overload]
        mock_logger.log.assert_any_call(LogLevel.L4, "Plotting %s plots ...", "State Estimates")
        mock_logger.log.assert_called_with(LogLevel.L4, "... done.")
        self.figs += figs
        self.assertEqual(err.keys(), {"state"})

    def test_groups(self) -> None:
        groups: list[tuple[int, ...]] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            with patch("dstauffman.plotting.aerospace.logger") as mock_logger:
                self.figs += plot.plot_states(self.gnd1, self.gnd2, groups=groups, opts=self.opts)  # type: ignore[call-overload]
        mock_logger.log.assert_any_call(LogLevel.L4, "Plotting %s plots ...", "State Estimates")
        mock_logger.log.assert_called_with(LogLevel.L4, "... done.")

    def tearDown(self) -> None:
        plot.close_all(self.figs)


# %% plotting.plot_tci
class Test_plotting_plot_tci(unittest.TestCase):
    r"""
    Tests the plotting.plot_tci function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
