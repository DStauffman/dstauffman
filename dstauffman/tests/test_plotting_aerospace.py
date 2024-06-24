r"""
Test file for the `aerospace` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

# %% Imports
import datetime
import unittest
from unittest.mock import Mock, patch
import warnings

from slog import LogLevel

from dstauffman import HAVE_MPL, HAVE_NUMPY
from dstauffman.aerospace import Kf, quat_norm
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np


# %% plotting.make_quaternion_plot
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_quaternion_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_quaternion_plot function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.description      = "example"
        self.time_one         = np.arange(11)
        self.time_two         = np.arange(2, 13)
        self.quat_one         = quat_norm(np.random.rand(4, 11))
        self.quat_two         = quat_norm(np.random.rand(4, 11))
        self.name_one         = "test1"
        self.name_two         = "test2"
        self.time_units       = "sec"
        self.start_date       = str(datetime.datetime.now())
        self.plot_components  = True
        self.rms_xmin         = 0
        self.rms_xmax         = 10
        self.disp_xmin        = -2
        self.disp_xmax        = np.inf
        self.make_subplots    = True
        self.single_lines     = False
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = "best"
        self.show_extra       = True
        self.second_units     = ("µrad", 1e6)
        self.data_as_rows     = True
        self.tolerance        = 0
        self.return_err       = False
        self.use_zoh          = False
        self.label_vert_lines = True
        self.figs: list[Figure] | None = None
        # fmt: on

    def test_nominal(self) -> None:
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        for i in range(3):
            self.assertLess(abs(err["diff"][i]), 3.15)

    def test_no_subplots(self) -> None:
        self.make_subplots = False
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        for i in range(3):
            self.assertLess(abs(err["diff"][i]), 3.15)

    def test_no_components(self) -> None:
        self.plot_components = False
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        for i in range(3):
            self.assertLess(abs(err["diff"][i]), 3.15)
        self.assertLess(abs(err["mag"]), 3.15)

    def test_no_start_date(self) -> None:
        self.start_date = ""
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        for i in range(3):
            self.assertLess(abs(err["diff"][i]), 3.15)

    def test_only_quat_one(self) -> None:
        self.quat_two.fill(np.nan)
        self.name_two = ""
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        self.assertTrue(np.all(np.isnan(err["diff"])))

    def test_only_quat_two(self) -> None:
        self.name_one = ""
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            None,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        self.assertTrue(np.all(np.isnan(err["diff"])))

    def test_rms_bounds(self) -> None:
        self.rms_xmin = 5
        self.rms_xmax = 7
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.quat_one,
            self.quat_two,
            name_one=self.name_one,
            name_two=self.name_two,
            time_units=self.time_units,
            start_date=self.start_date,
            plot_components=self.plot_components,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        for i in range(3):
            self.assertLess(abs(err["diff"][i]), 3.15)

    def test_use_mean(self) -> None:
        self.figs = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description, self.time_one, self.time_two, self.quat_one, self.quat_two, use_mean=True
        )

    def test_no_rms_in_legend(self) -> None:
        self.figs = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description, self.time_one, self.time_two, self.quat_one, self.quat_two, use_mean=True, show_rms=False
        )

    def test_plot_zero(self) -> None:
        self.figs = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description, self.time_one, self.time_two, self.quat_one, self.quat_two, plot_zero=True
        )

    def test_disp_bounds(self) -> None:
        self.figs = plot.make_quaternion_plot(  # type: ignore[call-overload]
            self.description, self.time_one, self.time_two, self.quat_one, self.quat_two, disp_xmin=2, disp_xmax=5
        )

    def test_no_overlap(self) -> None:
        time_one = np.arange(11.0).astype(float)
        time_two = np.arange(2.0, 13.0) + 0.5
        self.figs = plot.make_quaternion_plot(self.description, time_one, time_two, self.quat_one, self.quat_two)  # type: ignore[call-overload]

    def test_none1(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, None, self.quat_one, None)  # type: ignore[call-overload]

    def test_none2(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, None, self.time_two, None, self.quat_two)  # type: ignore[call-overload]

    @patch("dstauffman.plotting.generic.logger")
    def test_none3(self, mock_logger: Mock) -> None:
        self.figs = plot.make_quaternion_plot("", None, None, None, None)  # type: ignore[call-overload]
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(
            LogLevel.L5, 'No %s data was provided, so no plot was generated for "%s".', "quat", ""
        )

    def tearDown(self) -> None:
        if self.figs:
            plot.close_all(self.figs)


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
