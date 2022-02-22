r"""
Test file for the `health` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Separated into plot_health.py from plotting.py by David C. Stauffer in May 2020.
"""

#%% Imports
from typing import List, Optional
import unittest

from dstauffman import capture_output, HAVE_MPL, HAVE_NUMPY, HAVE_SCIPY
import dstauffman.plotting as plot

if HAVE_MPL:
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

#%% plotting.TruthPlotter
@unittest.skipIf(not HAVE_MPL or not HAVE_NUMPY, 'Skipping due to missing matplotlib/numpy dependency.')
class Test_plotting_TruthPlotter(unittest.TestCase):
    r"""
    Tests the plotting.TruthPlotter class with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig: Optional[plt.Figure] = None
        self.x = np.arange(0, 10, 0.1)
        self.y = np.sin(self.x)
        self.data = np.vstack((self.y, self.y + 0.01, self.y + 0.03)).T

    def test_nominal(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        np.testing.assert_array_almost_equal(self.y + 0.01, truth.data)  # type: ignore[arg-type]
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)  # type: ignore[arg-type]
        np.testing.assert_array_almost_equal(self.y + 0.03, truth.data_hi)  # type: ignore[arg-type]

    def test_matrix1(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data[:, 1])
        np.testing.assert_array_almost_equal(self.y + 0.01, truth.data)  # type: ignore[arg-type]
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_matrix2(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data)
        np.testing.assert_array_almost_equal(self.y + 0.01, truth.data)  # type: ignore[arg-type]
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)  # type: ignore[arg-type]
        np.testing.assert_array_almost_equal(self.y + 0.03, truth.data_hi)  # type: ignore[arg-type]

    def test_matrix3(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data[:, np.array([1])])
        np.testing.assert_array_almost_equal(self.y + 0.01, truth.data)  # type: ignore[arg-type]
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_bad_matrix(self) -> None:
        with self.assertRaises(ValueError):
            plot.TruthPlotter(self.x, np.random.rand(self.x.size, 4))

    def test_plotting0(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter()
        truth.plot_truth(ax)

    def test_plotting1(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        truth.plot_truth(ax)

    def test_plotting2(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting3(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, None, lo=self.y, hi=self.y + 0.03)
        truth.plot_truth(ax)

    def test_plotting4(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=None, hi=self.y + 0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting5(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03, type_='errorbar')
        # fake out data (can't be done through __init__ right now, this might need work)
        truth.data = self.data
        truth.data_lo = self.data - 0.01
        truth.data_hi = self.data + 0.01
        truth.plot_truth(ax, ix=1)

    def test_dont_hold_limits(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x - 10, self.y, lo=self.y - 1000, hi=self.y + 1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=False, hold_ylim=False)

    def test_hold_limits(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x - 10, self.y, lo=self.y - 1000, hi=self.y + 1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=True, hold_ylim=True)

    def test_bad_type(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03, type_='bad type')
        with self.assertRaises(ValueError):
            truth.plot_truth(ax)

    def test_pprint(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        with capture_output() as out:
            truth.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'TruthPlotter')
        self.assertTrue(lines[1].startswith(' time    = ['))
        self.assertEqual(lines[-1], ' name    = Observed')

    def test_is_null(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        self.assertFalse(truth.is_null)

    def test_get_data1(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        data = truth.get_data(self.data, scale=2)
        np.testing.assert_array_almost_equal(data, 2 * self.data)

    def test_get_data2(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y + 0.01, lo=self.y, hi=self.y + 0.03)
        data = truth.get_data(self.data, scale=3, ix=0)
        np.testing.assert_array_almost_equal(data, 3 * self.data[:, 0])

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


#%% plotting.plot_health_time_history
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_health_time_history(unittest.TestCase):
    r"""
    Tests the plotting.plot_health_time_history function with the following cases:
        Nominal
        Defaults
        With label
        With type
        With Opts
        With legend
        No data
        Ignore all zeros
        Bad legend
        Show zero
    """

    def setUp(self) -> None:
        self.time = np.arange(0, 10, 0.1) + 2000
        num_channels = 5
        self.data = np.random.rand(len(self.time), num_channels)
        mag = np.sum(self.data, axis=1)
        self.data = 10 * self.data / np.expand_dims(mag, axis=1)
        self.label = 'Plot description'
        self.units = 'percentage'
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.legend = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.second_units = 1000000
        self.figs: List[plt.Figure] = []

    def test_nominal(self) -> None:
        self.figs.append(
            plot.plot_health_time_history(
                self.time, self.data, label=self.label, units=self.units, opts=self.opts, legend=self.legend
            )
        )

    def test_defaults(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label))

    def test_with_units(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, self.units))

    def test_with_opts(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, opts=self.opts))

    def test_with_legend(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, legend=self.legend))

    def test_no_data(self) -> None:
        with capture_output() as out:
            plot.plot_health_time_history(self.time, None, '')
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'plot skipped due to missing data.')

    def test_ignore_zeros(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, ignore_empties=True))

    def test_ignore_zeros2(self) -> None:
        self.data[:, 1] = 0
        self.data[:, 3] = 0
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, ignore_empties=True))

    def test_ignore_zeros3(self) -> None:
        self.data = np.zeros(self.data.shape)
        with capture_output() as out:
            not_a_fig = plot.plot_health_time_history(self.time, self.data, label='All Zeros', ignore_empties=True)
        output = out.getvalue().strip()
        out.close()
        self.assertIs(not_a_fig, None)
        self.assertEqual(output, 'All Zeros plot skipped due to missing data.')

    def test_colormap(self) -> None:
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(
            plot.plot_health_time_history(self.time, self.data, self.label, ignore_empties=True, colormap=colormap)
        )

    def test_bad_legend(self) -> None:
        with self.assertRaises(AssertionError):
            plot.plot_health_time_history(self.time, self.data, self.label, legend=self.legend[:-1])

    def test_second_units1(self) -> None:
        self.figs.append(
            plot.plot_health_time_history(self.time, self.data, self.label, units='population', second_units=self.second_units)
        )

    def test_second_units2(self) -> None:
        second_units = ('New ylabel [units]', 100)
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, second_units=second_units))

    def test_single_point(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time[1:], self.data[1:, :], self.label))

    def test_show_zero(self) -> None:
        self.data += 1000
        self.opts.show_zero = True
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, opts=self.opts))

    def test_data_lo_and_hi(self) -> None:
        self.figs.append(
            plot.plot_health_time_history(self.time, self.data, self.label, data_lo=self.data - 1, data_hi=self.data + 1)
        )

    def test_not_ndarray(self) -> None:
        self.figs.append(plot.plot_health_time_history(0, 0, 'Zero'))

    def test_0d(self) -> None:
        self.figs.append(plot.plot_health_time_history(np.array(0), np.array(0), 'Zero'))

    def test_1d(self) -> None:
        self.figs.append(plot.plot_health_time_history(np.arange(5), np.arange(5), 'Line'))

    def test_3d(self) -> None:
        data3 = np.empty((self.data.shape[0], 3, self.data.shape[1]), dtype=float)
        data3[:, 0, :] = self.data
        data3[:, 1, :] = self.data + 0.1
        data3[:, 2, :] = self.data + 0.2
        self.opts.names = ['Run 1', 'Run 2', 'Run 3']
        self.figs.append(plot.plot_health_time_history(self.time, data3, self.label, opts=self.opts))

    def test_bad_4d(self) -> None:
        bad_data = np.random.rand(self.time.shape[0], 4, 5, 1)
        with self.assertRaises(AssertionError):
            plot.plot_health_time_history(self.time, bad_data, self.label, opts=self.opts)

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)


#%% plotting.plot_health_monte_carlo
@unittest.skipIf(not HAVE_MPL or not HAVE_NUMPY, 'Skipping due to missing matplotlib/numpy dependency.')
class Test_plotting_plot_health_monte_carlo(unittest.TestCase):
    r"""
    Tests the plotting.plot_health_monte_carlo function with the following cases:
        Nominal usage
        Truth data
        Opts
        plotting as diffs
        plotting as diffs + Opts
        plotting as a group
        using a different colormap
        plotting array data as individual
        plotting array data as group
        plotting with second Y axis (x2)
        plotting single scalars
        plotting empty lists
        no RMS in legend
        Show zero
    """

    def setUp(self) -> None:
        self.time = np.arange(0, 10, 0.1) if HAVE_NUMPY else list(range(10))
        self.data = np.sin(self.time) if HAVE_NUMPY else [x + 1.0 for x in self.time]  # type: ignore[attr-defined, call-overload]
        self.label = 'Sin'
        self.units = 'population'
        self.opts = plot.Opts()
        self.opts.names = ['Name 1']
        self.truth = plot.TruthPlotter(self.time, np.cos(self.time))  # type: ignore[arg-type, call-overload]
        self.data_matrix = np.column_stack((self.data, self.truth.data))  # type: ignore[arg-type]
        self.second_units = 1000000
        self.fig: Optional[List[plt.Figure]] = None

    def test_normal(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units)

    def test_truth1(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, truth=self.truth)

    def test_truth2(self) -> None:
        assert self.truth.data is not None
        self.truth.data_lo = self.truth.data - 0.1
        self.truth.data_hi = self.truth.data + 0.1
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, truth=self.truth)

    def test_bad_truth_size(self) -> None:
        assert self.truth.data is not None
        self.truth.data = self.truth.data[:-1]
        with self.assertRaises(ValueError):
            plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, truth=self.truth)
        # close uncompleted plot window
        plt.close(plt.gcf())

    def test_opts(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_diffs(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data_matrix, self.label, self.units, plot_as_diffs=True)

    def test_diffs_and_opts(self) -> None:
        self.fig = plot.plot_health_monte_carlo(
            self.time, self.data_matrix, self.label, self.units, opts=self.opts, plot_as_diffs=True
        )

    def test_group(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts, plot_indiv=False)

    def test_colormap(self) -> None:
        self.opts.colormap = 'Dark2'
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts)

    def test_colormap2(self) -> None:
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts, colormap=colormap)

    def test_array_data1(self) -> None:
        data = np.column_stack((self.data, self.data))
        self.fig = plot.plot_health_monte_carlo(self.time, data, self.label, self.units)

    def test_array_data2(self) -> None:
        data = np.column_stack((self.data, self.data))
        self.fig = plot.plot_health_monte_carlo(self.time, data, self.label, self.units, plot_as_diffs=True)

    def test_second_units1(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, second_units=self.second_units)

    def test_second_units2(self) -> None:
        second_units = ('New ylabel [units]', 100)
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, units='percentage', second_units=second_units)

    def test_simple(self) -> None:
        self.fig = plot.plot_health_monte_carlo(0, 0, 'Text')

    def test_plot_empty(self) -> None:
        self.fig = plot.plot_health_monte_carlo([], [], '')

    def test_plot_all_nans(self) -> None:
        self.fig = plot.plot_health_monte_carlo(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), self.label)

    def test_no_rms_in_legend1(self) -> None:
        self.opts.show_rms = False
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts)

    def test_no_rms_in_legend2(self) -> None:
        self.opts.show_rms = False
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts, plot_as_diffs=True)

    def test_show_zero(self) -> None:
        self.data += 1000
        self.opts.show_zero = True
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_skip_plot_sigmas(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, plot_sigmas=0)

    @unittest.skipIf(not HAVE_SCIPY, 'Skipping due to missing scipy dependency.')
    def test_plot_confidence(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data_matrix, self.label, self.units, plot_confidence=0.95)

    def test_not_ndarray(self) -> None:
        self.fig = plot.plot_health_monte_carlo(0, 0, 'Zero')

    def test_0d(self) -> None:
        self.fig = plot.plot_health_monte_carlo(np.array(0), np.array(0), 'Zero')

    def test_1d(self) -> None:
        self.fig = plot.plot_health_monte_carlo(np.arange(5), np.arange(5), 'Line')

    def test_bad_3d(self) -> None:
        bad_data = np.random.rand(self.time.shape[0], 4, 5)  # type: ignore[attr-defined]
        with self.assertRaises(ValueError):
            plot.plot_health_monte_carlo(self.time, bad_data, self.label, opts=self.opts)

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


#%% plotting.plot_icer
class Test_plotting_plot_icer(unittest.TestCase):
    r"""
    Tests the plotting.plot_icer function with the following cases:
        TBD
    """
    pass  # TODO: write this


#%% plotting.plot_population_pyramid
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_population_pyramid(unittest.TestCase):
    r"""
    Tests the plotting.plot_population_pyramid function with the following cases:
        Nominal
        Default arguments
    """

    def setUp(self) -> None:
        self.age_bins = np.array([0, 5, 10, 15, 20, 1000], dtype=int) if HAVE_NUMPY else [0, 5, 1000]
        self.male_per = np.array([100, 200, 300, 400, 500], dtype=int) if HAVE_NUMPY else [100, 200, 500]
        self.fmal_per = np.array([125, 225, 325, 375, 450], dtype=int) if HAVE_NUMPY else [125, 225, 450]
        self.title    = 'Test Title'
        self.opts     = plot.Opts()
        self.name1    = 'M'
        self.name2    = 'W'
        self.color1   = 'k'
        self.color2   = 'w'
        self.fig: Optional[List[plt.Figure]]

    def test_nominal(self) -> None:
        self.fig = plot.plot_population_pyramid(
            self.age_bins,
            self.male_per,
            self.fmal_per,
            self.title,
            opts=self.opts,
            name1=self.name1,
            name2=self.name2,
            color1=self.color1,
            color2=self.color2,
        )

    def test_defaults(self) -> None:
        self.fig = plot.plot_population_pyramid(self.age_bins, self.male_per, self.fmal_per, self.title)

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


#%% Unit test execution
if __name__ == '__main__':
    plot.suppress_plots()
    unittest.main(exit=False)
