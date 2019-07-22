# -*- coding: utf-8 -*-
r"""
Test file for the `plotting` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from datetime import datetime
import unittest

import matplotlib.pyplot as plt
import numpy as np

import dstauffman as dcs

#%% Plotter for testing
plotter = dcs.Plotter(False)

#%% Classes - Opts
class Test_Opts(unittest.TestCase):
    r"""
    Test Opts class, and by extension the frozen function and Frozen class using cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self):
        self.opts_fields = ['case_name']

    def test_calling(self):
        opts = dcs.Opts()
        for field in self.opts_fields:
            self.assertTrue(hasattr(opts, field))

    def test_new_attr(self):
        opts = dcs.Opts()
        with self.assertRaises(AttributeError):
            opts.new_field_that_does_not_exist = 1

    def test_get_names_successful(self):
        opts = dcs.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(0)
        self.assertEqual(name, 'Name 1')

    def test_get_names_unsuccessful(self):
        opts = dcs.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(2)
        self.assertEqual(name, '')

    def test_get_date_zero_str(self):
        opts = dcs.Opts()
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str, '')
        opts.date_zero = datetime(2019, 4, 1, 18, 0, 0)
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str,'  t(0) = 01-Apr-2019 18:00:00 Z')

    def test_pprint(self):
        opts = dcs.Opts()
        with dcs.capture_output() as out:
            opts.pprint(indent=2)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Opts')
        self.assertEqual(lines[1], '  case_name = ')
        self.assertEqual(lines[3], '  save_plot = False')
        self.assertEqual(lines[-1], '  names     = []')

#%% Functions - plot_time_history
class Test_plot_time_history(unittest.TestCase):
    r"""
    Tests the plot_time_history function with the following cases:
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
    def setUp(self):
        self.time     = np.arange(0, 10, 0.1) + 2000
        num_channels  = 5
        self.data     = np.random.rand(len(self.time), num_channels)
        mag           = self.data.cumsum(axis=1)[:,-1]
        self.data     = 10 * self.data / np.expand_dims(mag, axis=1)
        self.label    = 'Plot description'
        self.units    = 'percentage'
        self.opts     = dcs.Opts()
        self.opts.show_plot = False
        self.legend   = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.figs     = []
        self.second_y_scale = 1000000

    def test_nominal(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, label=self.label, \
            units=self.units, opts=self.opts, legend=self.legend))

    def test_defaults(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label))

    def test_with_units(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, self.units))

    def test_with_opts(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, opts=self.opts))

    def test_with_legend(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, legend=self.legend))

    def test_no_data(self):
        with dcs.capture_output() as out:
            dcs.plot_time_history(self.time, None, '')
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'plot skipped due to missing data.')

    def test_ignore_zeros(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, ignore_empties=True))

    def test_ignore_zeros2(self):
        self.data[:,1] = 0
        self.data[:,3] = 0
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, ignore_empties=True))

    def test_ignore_zeros3(self):
        self.data = np.zeros(self.data.shape)
        with dcs.capture_output() as out:
            not_a_fig = dcs.plot_time_history(self.time, self.data, label='All Zeros', ignore_empties=True)
        output = out.getvalue().strip()
        out.close()
        self.assertIs(not_a_fig, None)
        self.assertEqual(output,'All Zeros plot skipped due to missing data.')

    def test_colormap(self):
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, \
            ignore_empties=True, colormap=colormap))

    def test_bad_legend(self):
        with self.assertRaises(AssertionError):
            dcs.plot_time_history(self.time, self.data, self.label, legend=self.legend[:-1])

    def test_second_y_scale1(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, units='population', \
            second_y_scale=self.second_y_scale))

    def test_second_y_scale2(self):
        second_y_scale = {'New ylabel [units]': 100}
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, \
            second_y_scale=second_y_scale))

    def test_single_point(self):
        self.figs.append(dcs.plot_time_history(self.time[1:], self.data[1:,:], self.label))

    def test_show_zero(self):
        self.data += 1000
        self.opts.show_zero = True
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, opts=self.opts))

    def test_data_lo_and_hi(self):
        self.figs.append(dcs.plot_time_history(self.time, self.data, self.label, \
            data_lo=self.data-1, data_hi=self.data+1))

    def test_not_ndarray(self):
        self.figs.append(dcs.plot_time_history(0, 0, 'Zero'))

    def test_0d(self):
        self.figs.append(dcs.plot_time_history(np.array(0), np.array(0), 'Zero'))

    def test_1d(self):
        self.figs.append(dcs.plot_time_history(np.arange(5), np.arange(5), 'Line'))

    def test_3d(self):
        data3 = np.empty((self.data.shape[0], 3, self.data.shape[1]), dtype=float)
        data3[:,0,:] = self.data
        data3[:,1,:] = self.data + 0.1
        data3[:,2,:] = self.data + 0.2
        self.opts.names = ['Run 1', 'Run 2', 'Run 3']
        self.figs.append(dcs.plot_time_history(self.time, data3, self.label, opts=self.opts))

    def test_bad_4d(self):
        bad_data = np.random.rand(self.time.shape[0], 4, 5, 1)
        with self.assertRaises(AssertionError):
            dcs.plot_time_history(self.time, bad_data, self.label, opts=self.opts)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Functions - plot_monte_carlo
class Test_plot_monte_carlo(unittest.TestCase):
    r"""
    Tests plot_monte_carlo function with the following cases:
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
    def setUp(self):
        self.time = np.arange(0, 10, 0.1)
        self.data = np.sin(self.time)
        self.label = 'Sin'
        self.units = 'population'
        self.opts = dcs.Opts()
        self.opts.names = ['Name 1']
        self.truth = dcs.TruthPlotter(self.time, np.cos(self.time))
        self.data_matrix = np.column_stack((self.data, self.truth.data))
        self.second_y_scale = 1000000
        self.fig = None

    def test_normal(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units)

    def test_truth1(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, \
            truth=self.truth)

    def test_truth2(self):
        self.truth.data_lo = self.truth.data - 0.1
        self.truth.data_hi = self.truth.data + 0.1
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, \
            truth=self.truth)

    def test_bad_truth_size(self):
        self.truth.data = self.truth.data[:-1]
        with self.assertRaises(ValueError):
            dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, \
                truth=self.truth)
        # close uncompleted plot window
        plt.close(plt.gcf())

    def test_opts(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_diffs(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data_matrix, self.label, self.units, \
            plot_as_diffs=True)

    def test_diffs_and_opts(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data_matrix, self.label, self.units, \
            opts=self.opts, plot_as_diffs=True)

    def test_group(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, \
            opts=self.opts, plot_indiv=False)

    def test_colormap(self):
        self.opts.colormap = 'Dark2'
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, self.opts)

    def test_colormap2(self):
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, self.opts, \
            colormap=colormap)

    def test_array_data1(self):
        data = np.column_stack((self.data, self.data))
        self.fig = dcs.plot_monte_carlo(self.time, data, self.label, self.units)

    def test_array_data2(self):
        data = np.column_stack((self.data, self.data))
        self.fig = dcs.plot_monte_carlo(self.time, data, self.label, self.units, plot_as_diffs=True)

    def test_second_y_scale1(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, \
            second_y_scale=self.second_y_scale)

    def test_second_y_scale2(self):
        second_y_scale = {'New ylabel [units]': 100}
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, units='percentage', \
            second_y_scale=second_y_scale)

    def test_simple(self):
        self.fig = dcs.plot_monte_carlo(0, 0, 'Text')

    def test_plot_empty(self):
        self.fig = dcs.plot_monte_carlo([], [], '')

    def test_plot_all_nans(self):
        self.fig = dcs.plot_monte_carlo(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), self.label)

    def test_no_rms_in_legend1(self):
        self.opts.show_rms = False
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, self.opts)

    def test_no_rms_in_legend2(self):
        self.opts.show_rms = False
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, self.opts, \
            plot_as_diffs=True)

    def test_show_zero(self):
        self.data += 1000
        self.opts.show_zero = True
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, opts=self.opts)

    def test_skip_plot_sigmas(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data, self.label, self.units, plot_sigmas=0)

    def test_plot_confidence(self):
        self.fig = dcs.plot_monte_carlo(self.time, self.data_matrix, self.label, self.units, plot_confidence=0.95)

    def test_not_ndarray(self):
        self.fig = dcs.plot_monte_carlo(0, 0, 'Zero')

    def test_0d(self):
        self.fig = dcs.plot_monte_carlo(np.array(0), np.array(0), 'Zero')

    def test_1d(self):
        self.fig = dcs.plot_monte_carlo(np.arange(5), np.arange(5), 'Line')

    def test_bad_3d(self):
        bad_data = np.random.rand(self.time.shape[0], 4, 5)
        with self.assertRaises(ValueError):
            dcs.plot_monte_carlo(self.time, bad_data, self.label, opts=self.opts)

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

#%% Functions - plot_correlation_matrix
class Test_plot_correlation_matrix(unittest.TestCase):
    r"""
    Tests plot_correlation_matrix function with the following cases:
        normal mode
        non-square inputs
        default labels
        all arguments passed in
        symmetric matrix
        coloring with values above 1
        coloring with values below -1
        coloring with values in -1 to 1 instead of 0 to 1
        x label rotation
        bad labels (should raise error)
    """
    def setUp(self):
        num = 10
        self.figs   = []
        self.data   = dcs.unit(np.random.rand(num, num), axis=0)
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.units  = 'percentage'
        self.opts   = dcs.Opts()
        self.opts.case_name = 'Testing Correlation'
        self.matrix_name    = 'Not a Correlation Matrix'
        self.sym = self.data.copy()
        for j in range(num):
            for i in range(num):
                if i == j:
                    self.sym[i, j] = 1
                elif i > j:
                    self.sym[i, j] = self.data[j, i]

    def test_normal(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels))

    def test_nonsquare(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data[:5, :3], [self.labels[:3], \
            self.labels[:5]]))

    def test_default_labels(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data[:5, :3]))

    def test_type(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, units=self.units))

    def test_all_args(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels, self.units, self.opts, \
            matrix_name=self.matrix_name, cmin=0, cmax=1, xlabel='', ylabel='', \
            plot_lower_only=False, label_values=True, x_lab_rot=180, colormap='Paired'))

    def test_symmetric(self):
        self.figs.append(dcs.plot_correlation_matrix(self.sym))

    def test_symmetric_all(self):
        self.figs.append(dcs.plot_correlation_matrix(self.sym, plot_lower_only=False))

    def test_above_one(self):
        large_data = self.data * 1000
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_above_one_part2(self):
        large_data = self.data * 1000
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels, cmax=2000))

    def test_below_one(self):
        large_data = 1000*(self.data - 0.5)
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_below_one_part2(self):
        large_data = 1000*(self.data - 0.5)
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels, cmin=-2))

    def test_within_minus_one(self):
        large_data = self.data - 0.5
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_within_minus_one_part2(self):
        large_data = self.data - 0.5
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels, cmin=-1, cmax=1))

    def test_xlabel(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, xlabel='Testing Label'))

    def test_ylabel(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, ylabel='Testing Label'))

    def test_x_label_rotation(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels, x_lab_rot=0))

    def test_nans(self):
        self.data[0, 0] = np.nan
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels))

    def test_bad_labels(self):
        with self.assertRaises(ValueError):
            self.figs.append(dcs.plot_correlation_matrix(self.data, ['a']))

    def test_label_values(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, label_values=True))

    def tearDown(self):
        for i in range(len(self.figs)):
            plt.close(self.figs.pop())

#%% Functions - plot_bar_breakdown
class Test_plot_bar_breakdown(unittest.TestCase):
    r"""
    Tests the plot_bar_breakdown function with the following cases:
        Nominal
        Defaults
        With label
        With opts
        With legend
        Null data
        Bad legend
        With Colormap
    """
    def setUp(self):
        self.time = np.arange(0, 5, 1./12) + 2000
        num_bins = 5
        self.data = np.random.rand(len(self.time), num_bins)
        mag = self.data.cumsum(axis=1)[:,-1]
        self.data = self.data / np.expand_dims(mag, axis=1)
        self.label = 'Plot bar testing'
        self.legend = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.opts = dcs.Opts()
        self.opts.show_plot = False
        self.figs = []

    def test_nominal(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts, \
            legend=self.legend))

    def test_defaults(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label))

    def test_opts(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def test_legend(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend))

    def test_ignore_zeros(self):
        self.data[:, 1] = 0
        self.data[:, 3] = np.nan
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, ignore_empties=True))

    def test_null_data(self):
        with dcs.capture_output() as out:
            dcs.plot_bar_breakdown(self.time, None, '')
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'plot skipped due to missing data.')

    def test_colormap(self):
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, \
            opts=self.opts, colormap=colormap))

    def test_bad_legend(self):
        with self.assertRaises(AssertionError):
            dcs.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend[:-1])

    def test_single_point(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time[:1], self.data[:1,:], label=self.label))

    def test_new_colormap(self):
        self.opts.colormap = 'seismic'
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Functions - general_quaternion_plot
class Test_general_quaternion_plot(unittest.TestCase):
    r"""
    Tests the general_quaternion_plot function with the following cases:
        TBD
    """
    def setUp(self):
        self.description     = 'example'
        self.time_one        = np.arange(11)
        self.time_two        = np.arange(2, 13)
        self.quat_one        = dcs.quat_norm(np.random.rand(4, 11))
        self.quat_two        = dcs.quat_norm(np.random.rand(4, 11))
        self.name_one        = 'test1'
        self.name_two        = 'test2'
        self.start_date      = str(datetime.now())
        self.rms_xmin        = 0
        self.rms_xmax        = 10
        self.disp_xmin       = -2
        self.disp_xmax       = np.inf
        self.fig_visible     = True
        self.make_subplots   = True
        self.plot_components = True
        self.use_mean        = False
        self.plot_zero       = False
        self.show_rms        = True
        self.figs            = None

    def test_nominal(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_not_visible(self):
        self.fig_visible = False
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_subplots(self):
        self.make_subplots = False
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_components(self):
        self.plot_components = False
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)
        self.assertLess(abs(err['mag']), 3.15)

    def test_no_start_date(self):
        self.start_date = ''
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_only_quat_one(self):
        self.quat_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_quat_two(self):
        self.quat_one = None
        self.name_one = ''
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self):
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, fig_visible=self.fig_visible, \
             make_subplots=self.make_subplots, plot_components=self.plot_components)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_use_mean(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_rms_in_legend(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True, show_rms=False)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_plot_zero(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, plot_zero=True)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_plot_truth(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, truth_time=self.time_one, truth_data=self.quat_two)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_disp_bounds(self):
        (self.figs, err) = dcs.general_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, disp_xmin=2, disp_xmax=5)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Functions - general_difference_plot
class Test_general_difference_plot(unittest.TestCase):
    r"""
    Tests the general_defference_plot function with the following cases:
        TBD
    """
    def setUp(self):
        self.description     = 'example'
        self.time_one        = np.arange(11)
        self.time_two        = np.arange(2, 13)
        self.data_one        = 1e-6 * np.random.rand(4, 11)
        self.data_two        = self.data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6
        self.name_one        = 'test1'
        self.name_two        = 'test2'
        self.elements        = ['x', 'y']
        self.units           = 'rad'
        self.leg_scale       = 'micro'
        self.start_date      = str(datetime.now())
        self.rms_xmin        = 0
        self.rms_xmax        = 10
        self.disp_xmin       = -2
        self.disp_xmax       = np.inf
        self.fig_visible     = True
        self.make_subplots   = True
        color_lists          = dcs.get_color_lists()
        self.colormap        = color_lists['dbl_diff'] # + color_lists['double']
        self.use_mean        = False
        self.plot_zero       = False
        self.show_rms        = True
        self.legend_loc      = 'best'
        self.second_y_scale  = {u'µrad': 1e6}
        self.figs            = None

    def test_nominal(self):
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_not_visible(self):
        self.fig_visible = False
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_no_subplots(self):
        self.make_subplots = False
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_no_start_date(self):
        self.start_date = ''
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_only_data_one(self):
        self.data_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_data_two(self):
        self.data_one = None
        self.name_one = ''
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self):
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_use_mean(self):
        self.use_mean = True
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_no_rms_in_legend(self):
        self.show_rms = False
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_plot_zero(self):
        self.plot_zero = True
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            fig_visible=self.fig_visible, make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_y_scale=self.second_y_scale)

    def test_plot_truth(self):
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, elements=self.elements, units=self.units, \
            truth_time=self.time_one, truth_data=self.data_two)

    def test_disp_bounds(self):
        (self.figs, err) = dcs.general_difference_plot(self.description, self.time_one, self.time_two, \
             self.data_one, self.data_two, elements=self.elements, units=self.units, \
             disp_xmin=2, disp_xmax=5)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Functions - plot_phases
class Test_plot_phases(unittest.TestCase):
    r"""
    Tests the plot_phases function with the following cases:
        Single time
        End times
        No labels
    """
    def setUp(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Sine Wave')
        self.ax = self.fig.add_subplot(111)
        time = np.arange(101)
        data = np.cos(time / 10)
        self.times = np.array([5, 20, 60, 90])
        self.times2 = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
        self.ax.plot(time, data, '.-')
        self.colormap = 'tab10'
        self.labels = ['Part 1', 'Phase 2', 'Watch Out', 'Final']

    def test_single_time(self):
        dcs.plot_phases(self.ax, self.times, self.colormap, self.labels)

    def test_with_end_times(self):
        dcs.plot_phases(self.ax, self.times2, self.colormap, self.labels)

    def test_no_labels(self):
        dcs.plot_phases(self.ax, self.times, colormap=self.colormap)

    def test_no_colormap(self):
        dcs.plot_phases(self.ax, self.times, labels=self.labels)

    def tearDown(self):
        if self.fig:
            plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)
