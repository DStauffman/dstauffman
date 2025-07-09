r"""
Test file for the `generic` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in May 2020.

"""

# %% Imports
import datetime
import unittest
from unittest.mock import Mock, patch

from slog import LogLevel

from dstauffman import HAVE_DS, HAVE_MPL, HAVE_NUMPY, NP_DATETIME_UNITS, NP_INT64_PER_SEC, NP_TIMEDELTA_FORM
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np


# %% plotting.make_time_plot
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_time_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_time_plot function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.description      = "Values vs Time"
        self.time             = np.arange(-10.0, 10.1, 0.1)
        self.data             = self.time + np.cos(self.time)
        self.name             = ""
        self.elements         = None
        self.units            = ""
        self.time_units       = "sec"
        self.start_date       = ""
        self.rms_xmin         = -np.inf
        self.rms_xmax         = np.inf
        self.disp_xmin        = -np.inf
        self.disp_xmax        = np.inf
        self.single_lines     = False
        self.colormap         = "Paired"
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = "best"
        self.second_units     = None
        self.ylabel           = None
        self.data_as_rows     = True
        self.extra_plotter    = None
        self.use_zoh          = False
        self.label_vert_lines = True
        self.fig: Figure | None = None
        # fmt: on

    def test_simple(self) -> None:
        self.fig = plot.make_time_plot(self.description, self.time, self.data)
        self.assertIsNotNone(self.fig)

    def test_nominal(self) -> None:
        self.fig = plot.make_time_plot(
            self.description,
            self.time,
            self.data,
            name=self.name,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            extra_plotter=self.extra_plotter,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        self.assertIsNotNone(self.fig)

    def test_scalars(self) -> None:
        self.fig = plot.make_time_plot("", 0, 0)
        self.assertIsNotNone(self.fig)

    def test_bad_description(self) -> None:
        with self.assertRaises(AssertionError):
            plot.make_time_plot(None, 0, 0)  # type: ignore[arg-type]

    def test_0d(self) -> None:
        self.fig = plot.make_time_plot("", np.array(5), np.array(10.0))
        self.assertIsNotNone(self.fig)

    def test_list1(self) -> None:
        data = [self.data, self.data + 0.5, self.data + 1.0]
        self.fig = plot.make_time_plot(self.description, self.time, data)
        self.assertIsNotNone(self.fig)

    def test_list2(self) -> None:
        time = [self.time, self.time[:-1]]
        data = [self.data, 2 * self.data[:-1]]
        self.fig = plot.make_time_plot(self.description, time, data)
        self.assertIsNotNone(self.fig)

    def test_row_vectors(self) -> None:
        data = np.vstack((self.data, np.sin(self.time)))
        self.fig = plot.make_time_plot(self.description, self.time, data)
        self.assertIsNotNone(self.fig)

    def test_col_vectors(self) -> None:
        data = np.column_stack((self.data, np.sin(self.time)))
        self.fig = plot.make_time_plot(self.description, self.time, data, data_as_rows=False)
        self.assertIsNotNone(self.fig)

    def test_datetimes(self) -> None:
        time = np.datetime64("2021-06-01T00:00:00", NP_DATETIME_UNITS) + np.round(self.time * NP_INT64_PER_SEC).astype(
            np.int64
        ).astype(NP_TIMEDELTA_FORM)
        self.fig = plot.make_time_plot(
            self.description,
            time,
            self.data,
            name=self.name,
            elements=self.elements,
            units=self.units,
            time_units="numpy",
            start_date="",
            rms_xmin=time[5],
            rms_xmax=time[25],
            disp_xmin=time[1],
            disp_xmax=time[-2],
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            extra_plotter=self.extra_plotter,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        self.assertIsNotNone(self.fig)

    def test_strings(self) -> None:
        time = np.arange(100.0)
        data = np.full(100, "open", dtype="S6")
        data[10:20] = "closed"
        self.fig = plot.make_time_plot(self.description, time, data, show_rms=False)
        self.assertIsNotNone(self.fig)

    @unittest.skipIf(not HAVE_DS, "Skipping due to missing datashader dependency.")
    def test_datashader(self) -> None:
        time = np.linspace(0.0, 1000.0, 10**6)
        data = np.random.default_rng().random(10**6)
        self.fig = plot.make_time_plot(self.description, time, data, use_datashader=True)
        self.assertIsNotNone(self.fig)

    @unittest.skipIf(not HAVE_DS, "Skipping due to missing datashader dependency.")
    def test_datashader_dates(self) -> None:
        temp = np.linspace(0.0, 1000.0, 10**6)
        time = np.datetime64("2021-06-01T00:00:00", NP_DATETIME_UNITS) + np.round(temp * NP_INT64_PER_SEC).astype(
            np.int64
        ).astype(NP_TIMEDELTA_FORM)
        data = np.random.default_rng().random(10**6)
        self.fig = plot.make_time_plot(self.description, time, data, time_units="numpy", use_datashader=True)
        self.assertIsNotNone(self.fig)

    @unittest.skipIf(not HAVE_DS, "Skipping due to missing datashader dependency.")
    def test_datashader_strings(self) -> None:
        time = np.linspace(0.0, 1000.0, 10**4)
        data = np.full(10**4, "open", dtype="S6")
        data[1000:2000] = "closed"
        self.fig = plot.make_time_plot(self.description, time, data, show_rms=False, use_datashader=True)
        self.assertIsNotNone(self.fig)

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


# %% plotting.make_error_bar_plot
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_error_bar_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_error_bar_plot function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        prng = np.random.default_rng()
        # fmt: off
        self.description      = "Random Data Error Bars"
        self.time             = np.arange(11)
        self.data             = np.array([[3.0], [-2.0], [5]]) + prng.random((3, 11))
        self.mins             = self.data - 0.5 * prng.random((3, 11))
        self.maxs             = self.data + 1.5 * prng.random((3, 11))
        self.elements         = ["x", "y", "z"]
        self.units            = "rad"
        self.time_units       = "sec"
        self.start_date       = "  t0 = " + str(datetime.datetime.now())
        self.rms_xmin         = 1
        self.rms_xmax         = 10
        self.disp_xmin        = -2
        self.disp_xmax        = np.inf
        self.single_lines     = False
        self.colormap         = "tab10"
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = "best"
        self.second_units     = "milli"
        self.ylabel           = None
        self.data_as_rows     = True
        self.label_vert_lines = True
        self.fig: Figure | None = None
        # fmt: on

    def test_nominal(self) -> None:
        self.fig = plot.make_error_bar_plot(
            self.description,
            self.time,
            self.data,
            self.mins,
            self.maxs,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            label_vert_lines=self.label_vert_lines,
        )

    def test_bad_data(self) -> None:
        with self.assertRaises(AssertionError):
            plot.make_error_bar_plot(None, self.time, self.data, self.mins, self.maxs)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        if self.fig:
            plt.close(self.fig)


# %% plotting.make_difference_plot
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_difference_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_difference_plot function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.description      = "example"
        self.time_one         = np.arange(11)
        self.time_two         = np.arange(2, 13)
        self.data_one         = 1e-6 * np.random.default_rng().random((2, 11))
        self.data_two         = self.data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6
        self.name_one         = "test1"
        self.name_two         = "test2"
        self.elements         = ["x", "y"]
        self.units            = "rad"
        self.time_units       = "sec"
        self.start_date       = str(datetime.datetime.now())
        self.rms_xmin         = 0
        self.rms_xmax         = 10
        self.disp_xmin        = -2
        self.disp_xmax        = np.inf
        self.make_subplots    = True
        self.single_lines     = False
        self.colormap         = plot.COLOR_LISTS["dbl_comp"]
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = "best"
        self.show_extra       = True
        self.second_units     = "micro"
        self.ylabel           = None
        self.data_as_rows     = True
        self.tolerance        = 0
        self.return_err       = True
        self.use_zoh          = False
        self.label_vert_lines = True
        self.figs: list[Figure] | None = None
        # fmt: on

    def test_nominal(self) -> None:
        self.return_err = False
        self.figs = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_no_subplots(self) -> None:
        self.make_subplots = False
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_no_start_date(self) -> None:
        self.start_date = ""
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_only_data_one(self) -> None:
        self.data_two.fill(np.nan)
        self.name_two = ""
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )
        self.assertTrue(np.all(np.isnan(err["diff"])))

    def test_only_data_two(self) -> None:
        self.data_one = None  # type: ignore[assignment]
        self.name_one = ""
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
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
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_use_mean(self) -> None:
        self.use_mean = True
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_no_rms_in_legend(self) -> None:
        self.show_rms = False
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_plot_zero(self) -> None:
        self.plot_zero = True
        (self.figs, err) = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_disp_bounds(self) -> None:
        self.figs = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            self.time_one,
            self.time_two,
            self.data_one,
            self.data_two,
            elements=self.elements,
            units=self.units,
            disp_xmin=2,
            disp_xmax=5,
        )

    def test_no_overlap(self) -> None:
        time_one = np.arange(11.0)
        time_two = np.arange(2.0, 13.0) + 0.5
        self.return_err = False
        self.figs = plot.make_difference_plot(  # type: ignore[call-overload]
            self.description,
            time_one,
            time_two,
            self.data_one,
            self.data_two,
            name_one=self.name_one,
            name_two=self.name_two,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            show_extra=self.show_extra,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            tolerance=self.tolerance,
            return_err=self.return_err,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_none1(self) -> None:
        self.figs = plot.make_difference_plot(self.description, self.time_one, None, self.data_one, None)  # type: ignore[call-overload]

    def test_none2(self) -> None:
        self.figs = plot.make_difference_plot(self.description, None, self.time_two, None, self.data_two)  # type: ignore[call-overload]

    @patch("dstauffman.plotting.generic.logger")
    def test_none3(self, mock_logger: Mock) -> None:
        self.figs = plot.make_difference_plot("", None, None, None, None)  # type: ignore[call-overload]
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(
            LogLevel.L5, 'No %s data was provided, so no plot was generated for "%s".', "diff", ""
        )

    def test_bad_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            plot.make_difference_plot(None, None, None, None, None)  # type: ignore[call-overload]

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)


# %% plotting.make_categories_plot
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_categories_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_categories_plot function with the following cases:
        Nominal
        Minimal
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.description      = "Values vs Time"
        self.time             = np.arange(-10.0, 10.1, 0.1)
        self.data             = self.time + np.cos(self.time)
        self.MeasStatus       = type("MeasStatus", (object,), {"rejected": 0, "accepted": 1})
        self.cats             = np.full(self.time.shape, self.MeasStatus.accepted, dtype=int)  # type: ignore[attr-defined, type-var]
        self.cats[50:100]     = self.MeasStatus.rejected  # type: ignore[attr-defined]
        self.cat_names        = {0: "rejected", 1: "accepted"}
        self.name             = ""
        self.elements         = None
        self.units            = ""
        self.time_units       = "sec"
        self.start_date       = ""
        self.rms_xmin         = -np.inf
        self.rms_xmax         = np.inf
        self.disp_xmin        = -np.inf
        self.disp_xmax        = np.inf
        self.make_subplots    = True
        self.single_lines     = False
        self.colormap         = "Paired"
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = "best"
        self.second_units     = "unity"
        self.ylabel           = None
        self.data_as_rows     = True
        self.use_zoh          = False
        self.label_vert_lines = True
        self.figs: list[Figure] | None = None
        # fmt: on

    def test_nominal(self) -> None:
        self.figs = plot.make_categories_plot(
            self.description,
            self.time,
            self.data,
            self.cats,
            cat_names=self.cat_names,
            name=self.name,
            elements=self.elements,
            units=self.units,
            time_units=self.time_units,
            start_date=self.start_date,
            rms_xmin=self.rms_xmin,
            rms_xmax=self.rms_xmax,
            disp_xmin=self.disp_xmin,
            disp_xmax=self.disp_xmax,
            make_subplots=self.make_subplots,
            single_lines=self.single_lines,
            colormap=self.colormap,
            use_mean=self.use_mean,
            plot_zero=self.plot_zero,
            show_rms=self.show_rms,
            legend_loc=self.legend_loc,
            second_units=self.second_units,
            ylabel=self.ylabel,
            data_as_rows=self.data_as_rows,
            use_zoh=self.use_zoh,
            label_vert_lines=self.label_vert_lines,
        )

    def test_minimal(self) -> None:
        self.figs = plot.make_categories_plot(self.description, self.time, self.data, self.cats)

    def test_bad_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            plot.make_categories_plot(None, self.time, self.data, self.cats)  # type: ignore[arg-type]

    @unittest.skipIf(not HAVE_DS, "Skipping due to missing datashader dependency.")
    def test_datashader_cats(self) -> None:
        time = np.arange(10000.0)
        data = time + np.sin(time / np.pi / 100.0)
        cats = np.full(time.shape, self.MeasStatus.accepted, dtype=int)  # type: ignore[attr-defined, type-var]
        cats[500:1000] = self.MeasStatus.rejected  # type: ignore[attr-defined]
        self.figs = plot.make_categories_plot(self.description, time, data, cats, use_datashader=True)

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)


# %% plotting.make_bar_plot
class Test_plotting_make_bar_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_bar_plot function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% plotting.make_connected_sets
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_make_connected_sets(unittest.TestCase):
    r"""
    Tests the plotting.make_connected_sets function with the following cases:
        Nominal
        Color by Quad
        Color by Magnitude
        Center Origin
    """

    def setUp(self) -> None:
        prng = np.random.default_rng()
        self.description = "Focal Plane Sightings"
        self.points = 2 * prng.uniform(-1.0, 0, (2, 100))
        self.innovs = 0.1 * prng.normal(size=self.points.shape)
        self.fig: Figure | None = None

    def test_nominal(self) -> None:
        self.fig = plot.make_connected_sets(self.description, self.points, self.innovs)

    def test_color_by_direction(self) -> None:
        self.fig = plot.make_connected_sets(self.description, self.points, self.innovs, color_by="direction")

    def test_color_by_magnitude(self) -> None:
        self.fig = plot.make_connected_sets(self.description, self.points, self.innovs, color_by="magnitude")

    def test_center_origin(self) -> None:
        self.fig = plot.make_connected_sets(self.description, self.points, self.innovs, center_origin=True)

    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            plot.make_connected_sets(self.description, self.points, self.innovs, color_by="bad_option")

    def tearDown(self) -> None:
        if self.fig:
            plt.close(self.fig)


# %% Unit test execution
if __name__ == "__main__":
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
