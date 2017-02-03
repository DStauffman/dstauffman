# -*- coding: utf-8 -*-
r"""
Test file for the `plotting` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
import os
import unittest
import dstauffman as dcs
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

#%% Plotter for testing
plotter = dcs.Plotter(False)

#%% Classes for testing
# Plotter
class Test_Plotter(unittest.TestCase):
    r"""
    Tests the Plotter class with the following cases:
        Get level
        Set level
        Bad level (raises ValueError)
        printing
    """
    def setUp(self):
        self.flag    = True
        self.plotter = dcs.Plotter(self.flag)
        self.print   = 'Plotter(True)'

    def test_get_plotter(self):
        flag = self.plotter.get_plotter()
        self.assertTrue(flag)

    def test_set_plotter(self):
        flag = self.plotter.get_plotter()
        self.assertTrue(flag)
        self.plotter.set_plotter(False)
        self.assertFalse(plotter.get_plotter())

    def test_printing(self):
        with dcs.capture_output() as out:
            print(self.plotter)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.print)

    def test_no_show(self):
        self.plotter = dcs.Plotter()

    def tearDown(self):
        self.plotter.set_plotter(False)

# Opts
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

# TruthPlotter
class Test_TruthPlotter(unittest.TestCase):
    r"""
    Tests the TruthPlotter class with the following cases:
        TBD
    """
    def setUp(self):
        self.fig  = None
        self.x    = np.arange(0, 10, 0.1)
        self.y    = np.sin(self.x)
        self.data = np.vstack((self.y, self.y+0.01, self.y+0.03)).T

    def test_nominal(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_matrix1(self):
        truth = dcs.TruthPlotter(self.x, self.data[:, np.array([1])])
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_matrix2(self):
        truth = dcs.TruthPlotter(self.x, self.data)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_plotting0(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter()
        truth.plot_truth(ax)

    def test_plotting1(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting2(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting3(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, None, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting4(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=None, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_bad_type(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='bad type')
        with self.assertRaises(ValueError):
            truth.plot_truth(ax)

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

# MyCustomToolbar
class Test_MyCustomToolbar(unittest.TestCase):
    r"""
    Tests the MyCustomToolbar class with the following cases:
        No nothing
        Next plot
        Prev plot
        Close all
        Multiple nexts
        Multiple prevs
    """
    def setUp(self):
        plt.ioff()
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig1.toolbar_custom_ = dcs.MyCustomToolbar(self.fig1)
        self.fig2.toolbar_custom_ = dcs.MyCustomToolbar(self.fig2)

    def test_do_nothing(self):
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_next_plot(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_prev_plot(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_close_all(self):
        self.assertTrue(plt.fignum_exists(self.fig1.number))
        self.assertTrue(plt.fignum_exists(self.fig2.number))
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_close_all, Qt.LeftButton)
        self.assertFalse(plt.fignum_exists(self.fig1.number))
        self.assertFalse(plt.fignum_exists(self.fig2.number))

    def test_multiple_nexts(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_multiple_prevs(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_enter_and_exit_events(self):
        # TODO: this is not causing what I want to happen, the style changes aren't happening.
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_next_plot, delay=100)
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_prev_plot, delay=100)

    def tearDown(self):
        plt.close(self.fig1)
        plt.close(self.fig2)

# ColorMap
class Test_ColorMap(unittest.TestCase):
    r"""
    Tests ColorMap class with the following cases:
        Nominal mode
        Using num_colors specifier
        No inputs
        get_color method
        get_smap method
        set_colors method
    """
    def setUp(self):
        self.colormap   = 'Paired'
        self.low        = 0
        self.high       = 1
        self.num_colors = 5

    def test_nominal(self):
        cm = dcs.ColorMap(self.colormap, self.low, self.high)
        self.assertTrue(cm.get_color(self.low))

    def test_num_colors(self):
        cm = dcs.ColorMap(self.colormap, num_colors=self.num_colors)
        self.assertTrue(cm.get_color(self.low))

    def test_no_inputs(self):
        cm = dcs.ColorMap()
        self.assertTrue(cm.get_color(self.low))

    def test_get_color(self):
        cm = dcs.ColorMap(num_colors=self.num_colors)
        for i in range(self.num_colors):
            self.assertTrue(cm.get_color(i))

    def test_get_smap(self):
        cm = dcs.ColorMap()
        smap = cm.get_smap()
        self.assertTrue(isinstance(smap, cmx.ScalarMappable))

    def test_set_colors(self):
        cm = dcs.ColorMap(num_colors=5)
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm.set_colors(ax)
        ax.plot(0, 0)
        plt.close()
        self.assertTrue(True)

    def test_set_color_failure(self):
        cm = dcs.ColorMap()
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        with self.assertRaises(ValueError):
            cm.set_colors(ax)
        plt.close()

#%% Functions for testing
# close_all
class Test_close_all(unittest.TestCase):
    r"""
    Tests the close_all function with the following cases:
        Nominal
        Specified list
    """
    def test_nominal(self):
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        dcs.close_all()
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

    def test_list(self):
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        dcs.close_all([fig1])
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        plt.close(fig2)
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

# get_axes_scales
class Test_get_axes_scales(unittest.TestCase):
    r"""
    Tests get_axes_scales function with the following cases:
        Nominal usage
        Bad type (should raise ValueError)
    """
    def setUp(self):
        self.types  = ['unity', 'population', 'percentage', 'per 1K', 'per 100K', 'cost']
        self.scales = [1, 1, 100, 1000, 100000, 1e-3]
        self.units  = ['', '#', '%', 'per 1,000', 'per 100,000', "$K's"]
        self.bad_type = 'nonexistant'

    def test_nominal_usage(self):
        for (ix, this_type) in enumerate(self.types):
            (scale, units) = dcs.get_axes_scales(this_type)
            self.assertEqual(scale, self.scales[ix])
            self.assertEqual(units, self.units[ix])

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            dcs.get_axes_scales(self.bad_type)

# plot_time_history
class Test_plot_time_history(unittest.TestCase):
    r"""
    Tests plot_time_history function with the following cases:
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
    """
    def setUp(self):
        self.time = np.arange(0, 10, 0.1)
        self.data = np.sin(self.time)
        self.description = 'Sin'
        self.type_ = 'population'
        self.opts = dcs.Opts()
        self.opts.names = ['Name 1']
        self.truth = dcs.TruthPlotter(self.time, np.cos(self.time))
        self.data_matrix = np.column_stack((self.data, self.truth.data))
        self.second_y_scale = 1000000

    def test_normal(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_)

    def test_truth1(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
            truth=self.truth)

    def test_truth2(self):
        self.truth.data_lo = self.truth.data - 0.1
        self.truth.data_hi = self.truth.data + 0.1
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
            truth=self.truth)

    def test_bad_truth_size(self):
        self.truth.data = self.truth.data[:-1]
        with self.assertRaises(ValueError):
            dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
                truth=self.truth)

    def test_opts(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, opts=self.opts)

    def test_diffs(self):
        self.fig = dcs.plot_time_history(self.time, self.data_matrix, self.description, self.type_, \
            plot_as_diffs=True)

    def test_diffs_and_opts(self):
        self.fig = dcs.plot_time_history(self.time, self.data_matrix, self.description, self.type_, \
            opts=self.opts, plot_as_diffs=True)

    def test_group(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
            opts=self.opts, plot_indiv=False)

    def test_colormap(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, colormap='Dark2')

    def test_array_data1(self):
        data = np.column_stack((self.data, self.data))
        self.fig = dcs.plot_time_history(self.time, data, self.description, self.type_)

    def test_array_data2(self):
        data = np.column_stack((self.data, self.data))
        self.fig = dcs.plot_time_history(self.time, data, self.description, self.type_, plot_as_diffs=True)

    def test_second_y_scale1(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
            second_y_scale=self.second_y_scale)

    def test_second_y_scale2(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, type_='percentage', \
            second_y_scale=self.second_y_scale)

    def test_simple(self):
        self.fig = dcs.plot_time_history(0, 0)

    def test_plot_empty(self):
        self.fig = dcs.plot_time_history([], [])

    def test_plot_all_nans(self):
        self.fig = dcs.plot_time_history(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

    def test_no_rms_in_legend1(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, rms_in_legend=False)

    def test_no_rms_in_legend2(self):
        self.fig = dcs.plot_time_history(self.time, self.data, self.description, self.type_, \
            rms_in_legend=False, plot_as_diffs=True)

    def tearDown(self):
        if hasattr(self,'fig'):
            plt.close(self.fig)

# plot_correlation_matrix
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
        self.type_  = 'percentage'
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
        self.figs.append(dcs.plot_correlation_matrix(self.data, type_=self.type_))

    def test_all_args(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels, self.type_, self.opts, \
            matrix_name=self.matrix_name, cmin=0, cmax=1, colormap='viridis', xlabel='', ylabel='', \
            plot_lower_only=False, label_values=True, x_lab_rot=180))

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

    def test_colormap(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, colormap='seismic_r'))

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

# plot_multiline_history
class Test_plot_multiline_history(unittest.TestCase):
    r"""
    Tests the plot_multipline_history function with the following cases:
        Nominal
        Defaults
        With label
        With type
        With Opts
        With legend
        With Colormap
        No data
        Ignore all zeros
        Bad legend
    """
    def setUp(self):
        self.time     = np.arange(0, 10, 0.1) + 2000
        num_channels  = 5
        self.data     = np.random.rand(len(self.time), num_channels)
        mag           = self.data.cumsum(axis=1)[:,-1]
        self.data     = 10 * self.data / np.expand_dims(mag, axis=1)
        self.type_    = 'percentage'
        self.label    = 'Plot description'
        self.opts     = dcs.Opts()
        self.opts.show_plot = False
        self.legend   = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.colormap = 'seismic'
        self.figs     = []
        self.second_y_scale = 1000000

    def test_nominal(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, type_=self.type_, \
            label = self.label, opts=self.opts, legend=self.legend, colormap=self.colormap))

    def test_defaults(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data))

    def test_with_label(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, label=self.label))

    def test_with_type_(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, self.type_))

    def test_with_opts(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, opts=self.opts))

    def test_with_legend(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, legend=self.legend))

    def test_with_colormap(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, colormap=self.colormap))

    def test_no_data(self):
        with dcs.capture_output() as out:
            dcs.plot_multiline_history(self.time, None)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'plot skipped due to missing data.')

    def test_ignore_zeros(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, ignore_empties=True))

    def test_ignore_zeros2(self):
        self.data[:,1] = 0
        self.data[:,3] = 0
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, ignore_empties=True))

    def test_ignore_zeros3(self):
        self.data = np.zeros(self.data.shape)
        with dcs.capture_output() as out:
            not_a_fig = dcs.plot_multiline_history(self.time, self.data, label='All Zeros', ignore_empties=True)
        output = out.getvalue().strip()
        out.close()
        self.assertIs(not_a_fig, None)
        self.assertEqual(output,'All Zeros plot skipped due to missing data.')

    def test_bad_legend(self):
        with self.assertRaises(AssertionError):
            self.figs.append(dcs.plot_multiline_history(self.time, self.data, legend=self.legend[:-1]))

    def test_second_y_scale1(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, type_='population', \
            second_y_scale=self.second_y_scale))

    def test_second_y_scale2(self):
        self.figs.append(dcs.plot_multiline_history(self.time, self.data, \
            second_y_scale=self.second_y_scale))

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

# plot_bar_breakdown
class Test_plot_bar_breakdown(unittest.TestCase):
    r"""
    Tests the plot_bar_breakdown function with the following cases:
        Nominal
        Defaults
        With label
        With opts
        With legend
        With colormap
        Null data
        Bad legend
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
        self.colormap = 'seismic'
        self.figs = []

    def test_nominal(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts, \
            legend=self.legend, colormap=self.colormap))

    def test_defaults(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data))

    def test_label(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, label=self.label))

    def test_opts(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, opts=self.opts))

    def test_legend(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, legend=self.legend))

    def test_colormap(self):
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, colormap=self.colormap))

    def test_ignore_zeros(self):
        self.data[:, 1] = 0
        self.data[:, 3] = np.nan
        self.figs.append(dcs.plot_bar_breakdown(self.time, self.data, ignore_empties=True))

    def test_null_data(self):
        with dcs.capture_output() as out:
            dcs.plot_bar_breakdown(self.time, None)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'plot skipped due to missing data.')

    def test_bad_legend(self):
        with self.assertRaises(AssertionError):
            dcs.plot_bar_breakdown(self.time, self.data, legend=self.legend[:-1])

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

# plot_bpe_convergence
class Test_plot_bpe_convergence(unittest.TestCase):
    r"""
    Tests the plot_bpe_convergence function with the following cases:
        Nominal
        Only two costs
        No Opts
        No Costs
    """
    def setUp(self):
        self.costs = np.array([1, 0.1, 0.05, 0.01])
        self.opts = dcs.Opts()
        self.opts.show_plot = False
        self.figs = []

    def test_nominal(self):
        self.figs.append(dcs.plot_bpe_convergence(self.costs, self.opts))

    def test_only_two_costs(self):
        self.figs.append(dcs.plot_bpe_convergence(self.costs[np.array([0, 3])], self.opts))

    def test_no_opts(self):
        self.figs.append(dcs.plot_bpe_convergence(self.costs))

    def test_no_costs(self):
        self.figs.append(dcs.plot_bpe_convergence([], self.opts))

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

# storefig
class Test_storefig(unittest.TestCase):
    r"""
    Tests the storefig function with the following cases:
        saving one plot to disk
        saving one plot to multiple plot types
        saving multiple plots to one plot type
        saving to a bad folder location (should raise error)
        specifying a bad plot type (should raise error)
    """
    @classmethod
    def setUpClass(cls):
        # create data
        cls.time = np.arange(0, 10, 0.1)
        cls.data = np.sin(cls.time)
        cls.title = 'Test Plot'
        cls.folder = dcs.get_tests_dir()
        cls.plot_type = ['png', 'jpg']
        # turn interaction off to make the plots draw all at once on a show() command
        plt.ioff()
        # create the figure and set the title
        cls.fig = plt.figure()
        cls.fig.canvas.set_window_title(cls.title)
        # add an axis and plot the data
        ax = cls.fig.add_subplot(111)
        ax.plot(cls.time, cls.data)
        # add labels and legends
        plt.xlabel('Time [year]')
        plt.ylabel('Value [units]')
        plt.title(cls.title)
        # show a grid
        plt.grid(True)

    def test_saving(self):
        dcs.storefig(self.fig, self.folder, self.plot_type[0])
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type[0])
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_multiple_plot_types(self):
        dcs.storefig(self.fig, self.folder, self.plot_type)
        # assert that files exist
        for this_type in self.plot_type:
            this_filename = os.path.join(self.folder, self.title + '.' + this_type)
            self.assertTrue(os.path.isfile(this_filename))
            # remove file
            os.remove(this_filename)

    def test_multiple_figures(self):
        dcs.storefig([self.fig, self.fig], self.folder, self.plot_type[0])
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type[0])
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_bad_folder(self):
        with self.assertRaises(ValueError):
            dcs.storefig(self.fig, 'ZZ:\\non_existant_path')
        # TODO:
        pass

    def test_bad_plot_type(self):
        # TODO:
        pass

    @classmethod
    def tearDownClass(cls):
        plt.close(cls.fig)

# titleprefix
class Test_titleprefix(unittest.TestCase):
    r"""
    Tests the titleprefix function with the following cases:
        normal use
        null prefix
        multiple figures
    """
    def setUp(self):
        plt.ioff()
        self.fig = plt.figure()
        self.title = 'Figure Title'
        self.prefix = 'Prefix'
        self.fig.canvas.set_window_title(self.title)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title('X vs Y')

    def test_normal(self):
        dcs.titleprefix(self.fig, self.prefix)

    def test_null_prefix(self):
        dcs.titleprefix(self.fig)

    def test_multiple_figs(self):
        dcs.titleprefix([self.fig, self.fig], self.prefix)
        plt.close()

    def tearDown(self):
        plt.close()

# disp_xlimits
class Test_disp_xlimits(unittest.TestCase):
    r"""
    Tests the disp_xlimits function with the following cases:
        Normal use
        Null action
        Only xmin
        Only xmax
        Multiple figures
    """
    def setUp(self):
        plt.ioff()
        self.fig = plt.figure()
        self.xmin = 2
        self.xmax = 5
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        plt.plot(x, y)

    def test_normal(self):
        dcs.disp_xlimits(self.fig, self.xmin, self.xmax)

    def test_null_action(self):
        dcs.disp_xlimits(self.fig)

    def test_just_xmin(self):
        dcs.disp_xlimits(self.fig, xmin=self.xmin)

    def test_just_xmax(self):
        dcs.disp_xlimits(self.fig, xmax=self.xmax)

    def test_multiple_figs(self):
        dcs.disp_xlimits([self.fig, self.fig], self.xmin, self.xmax)

    def tearDown(self):
        plt.close()

# setup_plots
class Test_setup_plots(unittest.TestCase):
    r"""
    Tests the setup_plots function with the following cases:
        Prepend a title
        Don't prepend a title
        Don't show the plot
        Multiple figures
        Save the plot
        Show the plot link
    """
    def setUp(self):
        plt.ioff()
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title('X vs Y')
        plt.xlabel('time [years]')
        plt.ylabel('value [radians]')
        self.opts = dcs.Opts()
        self.opts.case_name = 'Testing'
        self.opts.show_plot = True
        self.opts.save_plot = False
        self.opts.save_path = dcs.get_tests_dir()

    def test_title(self):
        dcs.setup_plots(self.fig, self.opts)

    def test_no_title(self):
        self.opts.case_name = ''
        dcs.setup_plots(self.fig, self.opts)

    def test_not_showing_plot(self):
        self.opts.show_plot = False
        dcs.setup_plots(self.fig, self.opts)

    def test_multiple_figs(self):
        self.fig = [self.fig]
        plt.ioff()
        new_fig = plt.figure()
        plt.plot(0, 0)
        self.fig.append(new_fig)
        dcs.setup_plots(self.fig, self.opts)

    def test_saving_plot(self):
        this_filename = os.path.join(dcs.get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        dcs.setup_plots(self.fig, self.opts)
        # remove file
        os.remove(this_filename)

    def test_show_link(self):
        this_filename = os.path.join(dcs.get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        self.opts.show_link = True
        with dcs.capture_output() as out:
            dcs.setup_plots(self.fig, self.opts)
        output = out.getvalue().strip()
        out.close()
        # remove file
        os.remove(this_filename)
        self.assertTrue(output.startswith('Plots saved to <a href="'))

    def tearDown(self):
        plt.close()

# figmenu
class Test_figmenu(unittest.TestCase):
    r"""
    Tests the figmenu function with the following cases:
        One input
        List input
    """
    def setUp(self):
        plt.ioff()
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()

    def test_one_input(self):
        dcs.figmenu(self.fig1)

    def test_list_input(self):
        dcs.figmenu([self.fig1, self.fig2])

    def tearDown(self):
        plt.close(self.fig1)
        plt.close(self.fig2)

# rgb_ints_to_hex
class Test_rgb_ints_to_hex(unittest.TestCase):
    r"""
    Tests the rgb_ints_to_hex function with the following cases:
        Nominal
        Out of range
    """
    def setUp(self):
        self.colors    = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (79, 129, 189), (0, 0, 0)]
        self.hex_codes = ['#ff0000', '#00ff00', '#0000ff', '#4f81bd', '#000000']

    def test_nominal(self):
        for (ix, this_color) in enumerate(self.colors):
            hex_codes = dcs.rgb_ints_to_hex(this_color)
            self.assertEqual(hex_codes, self.hex_codes[ix])

    def test_bad_range(self):
        hex_code = dcs.rgb_ints_to_hex((-100, 500, 9))
        self.assertEqual(hex_code, '#00ff09')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
