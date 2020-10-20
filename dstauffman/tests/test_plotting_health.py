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

import matplotlib.pyplot as plt
import numpy as np

from dstauffman import capture_output
import dstauffman.plotting as plot

#%% Plotter for testing
plotter = plot.Plotter(False)

#%% plotting.plot_health_time_history
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
        self.time     = np.arange(0, 10, 0.1) + 2000
        num_channels  = 5
        self.data     = np.random.rand(len(self.time), num_channels)
        mag           = np.sum(self.data, axis=1)
        self.data     = 10 * self.data / np.expand_dims(mag, axis=1)
        self.label    = 'Plot description'
        self.units    = 'percentage'
        self.opts     = plot.Opts()
        self.opts.show_plot = False
        self.legend   = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.figs: List[plt.Figure] = []
        self.second_yscale = 1000000

    def test_nominal(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, label=self.label, \
            units=self.units, opts=self.opts, legend=self.legend))

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
        self.data[:,1] = 0
        self.data[:,3] = 0
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, ignore_empties=True))

    def test_ignore_zeros3(self) -> None:
        self.data = np.zeros(self.data.shape)
        with capture_output() as out:
            not_a_fig = plot.plot_health_time_history(self.time, self.data, label='All Zeros', ignore_empties=True)
        output = out.getvalue().strip()
        out.close()
        self.assertIs(not_a_fig, None)
        self.assertEqual(output,'All Zeros plot skipped due to missing data.')

    def test_colormap(self) -> None:
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, \
            ignore_empties=True, colormap=colormap))

    def test_bad_legend(self) -> None:
        with self.assertRaises(AssertionError):
            plot.plot_health_time_history(self.time, self.data, self.label, legend=self.legend[:-1])

    def test_second_yscale1(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, units='population', \
            second_yscale=self.second_yscale))

    def test_second_yscale2(self) -> None:
        second_yscale = {'New ylabel [units]': 100}
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, \
            second_yscale=second_yscale))

    def test_single_point(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time[1:], self.data[1:,:], self.label))

    def test_show_zero(self) -> None:
        self.data += 1000
        self.opts.show_zero = True
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, opts=self.opts))

    def test_data_lo_and_hi(self) -> None:
        self.figs.append(plot.plot_health_time_history(self.time, self.data, self.label, \
            data_lo=self.data-1, data_hi=self.data+1))

    def test_not_ndarray(self) -> None:
        self.figs.append(plot.plot_health_time_history(0, 0, 'Zero'))

    def test_0d(self) -> None:
        self.figs.append(plot.plot_health_time_history(np.array(0), np.array(0), 'Zero'))

    def test_1d(self) -> None:
        self.figs.append(plot.plot_health_time_history(np.arange(5), np.arange(5), 'Line'))

    def test_3d(self) -> None:
        data3 = np.empty((self.data.shape[0], 3, self.data.shape[1]), dtype=float)
        data3[:,0,:] = self.data
        data3[:,1,:] = self.data + 0.1
        data3[:,2,:] = self.data + 0.2
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
        self.time = np.arange(0, 10, 0.1)
        self.data = np.sin(self.time)
        self.label = 'Sin'
        self.units = 'population'
        self.opts = plot.Opts()
        self.opts.names = ['Name 1']
        self.truth = plot.TruthPlotter(self.time, np.cos(self.time))
        self.data_matrix = np.column_stack((self.data, self.truth.data))
        self.second_yscale = 1000000
        self.fig: Optional[List[plt.Figure]] = None

    def test_normal(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units)

    def test_truth1(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, \
            truth=self.truth)

    def test_truth2(self) -> None:
        self.truth.data_lo = self.truth.data - 0.1
        self.truth.data_hi = self.truth.data + 0.1
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, \
            truth=self.truth)

    def test_bad_truth_size(self) -> None:
        self.truth.data = self.truth.data[:-1]
        with self.assertRaises(ValueError):
            plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, \
                truth=self.truth)
        # close uncompleted plot window
        plt.close(plt.gcf())

    def test_opts(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_diffs(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data_matrix, self.label, self.units, \
            plot_as_diffs=True)

    def test_diffs_and_opts(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data_matrix, self.label, self.units, \
            opts=self.opts, plot_as_diffs=True)

    def test_group(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, \
            opts=self.opts, plot_indiv=False)

    def test_colormap(self) -> None:
        self.opts.colormap = 'Dark2'
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts)

    def test_colormap2(self) -> None:
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts, \
            colormap=colormap)

    def test_array_data1(self) -> None:
        data = np.column_stack((self.data, self.data))
        self.fig = plot.plot_health_monte_carlo(self.time, data, self.label, self.units)

    def test_array_data2(self) -> None:
        data = np.column_stack((self.data, self.data))
        self.fig = plot.plot_health_monte_carlo(self.time, data, self.label, self.units, plot_as_diffs=True)

    def test_second_yscale1(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, \
            second_yscale=self.second_yscale)

    def test_second_yscale2(self) -> None:
        second_yscale = {'New ylabel [units]': 100}
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, units='percentage', \
            second_yscale=second_yscale)

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
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, self.opts, \
            plot_as_diffs=True)

    def test_show_zero(self) -> None:
        self.data += 1000
        self.opts.show_zero = True
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_skip_plot_sigmas(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data, self.label, self.units, plot_sigmas=0)

    def test_plot_confidence(self) -> None:
        self.fig = plot.plot_health_monte_carlo(self.time, self.data_matrix, self.label, self.units, plot_confidence=0.95)

    def test_not_ndarray(self) -> None:
        self.fig = plot.plot_health_monte_carlo(0, 0, 'Zero')

    def test_0d(self) -> None:
        self.fig = plot.plot_health_monte_carlo(np.array(0), np.array(0), 'Zero')

    def test_1d(self) -> None:
        self.fig = plot.plot_health_monte_carlo(np.arange(5), np.arange(5), 'Line')

    def test_bad_3d(self) -> None:
        bad_data = np.random.rand(self.time.shape[0], 4, 5)
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
    pass # TODO: write this

#%% plotting.plot_population_pyramid
class Test_plotting_plot_population_pyramid(unittest.TestCase):
    r"""
    Tests the plotting.plot_population_pyramid function with the following cases:
        Nominal
        Default arguments
    """
    def setUp(self) -> None:
        self.age_bins = np.array([0, 5, 10, 15, 20, 1000], dtype=int)
        self.male_per = np.array([100, 200, 300, 400, 500], dtype=int)
        self.fmal_per = np.array([125, 225, 325, 375, 450], dtype=int)
        self.title    = 'Test Title'
        self.opts     = plot.Opts()
        self.name1    = 'M'
        self.name2    = 'W'
        self.color1   = 'k'
        self.color2   = 'w'
        self.fig: Optional[List[plt.Figure]]

    def test_nominal(self) -> None:
        self.fig = plot.plot_population_pyramid(self.age_bins, self.male_per, self.fmal_per, \
            self.title, opts=self.opts, name1=self.name1, name2=self.name2, color1=self.color1, \
            color2=self.color2)

    def test_defaults(self) -> None:
        self.fig = plot.plot_population_pyramid(self.age_bins, self.male_per, self.fmal_per, \
            self.title)

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)
