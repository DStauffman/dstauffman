r"""
Test file for the `plotting` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import contextlib
import datetime
import os
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from dstauffman import capture_output, get_tests_dir, LogLevel, unit

import dstauffman.plotting as plot

#%% Plotter for testing
plotter = plot.Plotter(False)

#%% plotting.Opts
class Test_plotting_Opts(unittest.TestCase):
    r"""
    Tests the plotting.Opts class with the following cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self):
        self.opts_fields = ['case_name']

    def test_calling(self):
        opts = plot.Opts()
        for field in self.opts_fields:
            self.assertTrue(hasattr(opts, field))

    def test_new_attr(self):
        opts = plot.Opts()
        with self.assertRaises(AttributeError):
            opts.new_field_that_does_not_exist = 1

    def test_get_names_successful(self):
        opts = plot.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(0)
        self.assertEqual(name, 'Name 1')

    def test_get_names_unsuccessful(self):
        opts = plot.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(2)
        self.assertEqual(name, '')

    def test_get_date_zero_str(self):
        opts = plot.Opts()
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str, '')
        opts.date_zero = datetime.datetime(2019, 4, 1, 18, 0, 0)
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str,'  t(0) = 01-Apr-2019 18:00:00 Z')

    def get_time_limits(self):
        opts = plot.Opts()
        opts.disp_xmin = 60
        opts.disp_xmax = np.inf
        opts.rms_xmin = -np.inf
        opts.rms_xmax = None
        opts.time_base = 'sec'
        opts.time_unit = 'min'
        (d1, d2, r1, r2) = opts.get_time_limits()
        self.assertEqual(d1, 1)
        self.assertEqual(d2, np.inf)
        self.assertEqual(r1, -np.inf)
        self.assertIsNone(r2)

    def get_time_limits2(self):
        opts = plot.Opts().convert_dates('datetime')
        opts.disp_xmin = datetime.datetime(2020, 6, 1, 0, 0, 0)
        opts.disp_xmax = datetime.datetime(2020, 6, 1, 12, 0, 0)
        (d1, d2, r1, r2) = opts.get_time_limits()
        self.assertEqual(d1, datetime.datetime(2020, 6, 1, 0, 0, 0))
        self.assertEqual(d2, datetime.datetime(2020, 6, 1, 12, 0, 0))
        self.assertIsNone(r1)
        self.assertIsNone(r2)

    def test_pprint(self):
        opts = plot.Opts()
        with capture_output() as out:
            opts.pprint(indent=2)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Opts')
        self.assertEqual(lines[1], '  case_name = ')
        self.assertEqual(lines[3], '  save_plot = False')
        self.assertEqual(lines[-1], '  names     = []')

    def test_convert_dates(self):
        opts = plot.Opts()
        self.assertEqual(opts.disp_xmin, -np.inf)
        self.assertEqual(opts.time_base, 'sec')
        opts.convert_dates('datetime')
        self.assertIsNone(opts.disp_xmin)
        self.assertEqual(opts.time_base, 'datetime')

    def test_convert_dates2(self):
        opts = plot.Opts(date_zero=datetime.datetime(2020, 6, 1))
        opts.rms_xmin = -10
        opts.rms_xmax = 10
        opts.disp_xmin = 5
        opts.disp_xmax = 150
        opts.convert_dates('datetime')
        self.assertEqual(opts.time_base, 'datetime')
        self.assertEqual(opts.rms_xmin,  datetime.datetime(2020, 5, 31, 23, 59, 50))
        self.assertEqual(opts.rms_xmax,  datetime.datetime(2020, 6, 1, 0, 0, 10))
        self.assertEqual(opts.disp_xmin, datetime.datetime(2020, 6, 1, 0, 0, 5))
        self.assertEqual(opts.disp_xmax, datetime.datetime(2020, 6, 1, 0, 2, 30))

#%% plotting.Plotter
class Test_plotting_Plotter(unittest.TestCase):
    r"""
    Tests the plotting.Plotter class with the following cases:
        Get level
        Set level
        Bad level (raises ValueError)
        printing
    """
    def setUp(self):
        self.flag    = True
        self.plotter = plot.Plotter(self.flag)
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
        with capture_output() as out:
            print(self.plotter)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.print)

    def test_no_show(self):
        self.plotter = plot.Plotter()

    def tearDown(self):
        self.plotter.set_plotter(False)

#%% plotting.plot_time_history
class Test_plotting_plot_time_history(unittest.TestCase):
    r"""
    Tests the plotting.plot_time_history function with the following cases:
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
        self.description = 'Plot description'
        self.time        = np.arange(0, 10, 0.1) + 2000
        num_channels     = 5
        self.data        = np.random.rand(len(self.time), num_channels)
        mag              = np.sum(self.data, axis=1)
        self.data        = 10 * self.data / np.expand_dims(mag, axis=1)
        self.units       = 'percentage'
        self.opts        = plot.Opts()
        self.opts.show_plot = False
        self.elements    = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.figs        = []
        self.second_yscale = 1000000

    def test_nominal(self):
        self.figs.append(plot.plot_time_history(self.description, self.time, self.data, \
            opts=self.opts, data_as_rows=False))

#    def test_defaults(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label))
#
#    def test_with_units(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, self.units))
#
#    def test_with_opts(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, opts=self.opts))
#
#    def test_with_legend(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, legend=self.legend))
#
#    @patch('dstauffman.plotting.plotting.logger')
#    def test_no_data(self, mock_logger):
#        plot.plot_time_history(self.time, None, '')
#        self.assertEqual(mock_logger.log.call_count, 1)
#        mock_logger.log.assert_called_with(LogLevel.L5, '  plot skipped due to missing data.')
#
#    def test_ignore_zeros(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, ignore_empties=True))
#
#    def test_ignore_zeros2(self):
#        self.data[:,1] = 0
#        self.data[:,3] = 0
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, ignore_empties=True))
#
#    @patch('dstauffman.plotting.plotting.logger')
#    def test_ignore_zeros3(self, mock_logger):
#        self.data = np.zeros(self.data.shape)
#        not_a_fig = plot.plot_time_history(self.time, self.data, label='All Zeros', ignore_empties=True)
#        self.assertIs(not_a_fig, None)
#        self.assertEqual(mock_logger.log.call_count, 1)
#        mock_logger.log.assert_called_with(LogLevel.L5, 'All Zeros plot skipped due to missing data.')
#
#    def test_colormap(self):
#        self.opts.colormap = 'Dark2'
#        colormap = 'Paired'
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, \
#            ignore_empties=True, colormap=colormap))
#
#    def test_bad_legend(self):
#        with self.assertRaises(AssertionError):
#            plot.plot_time_history(self.time, self.data, self.label, legend=self.legend[:-1])
#
#    def test_second_yscale1(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, units='population', \
#            second_yscale=self.second_yscale))
#
#    def test_second_yscale2(self):
#        second_yscale = {'New ylabel [units]': 100}
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, \
#            second_yscale=second_yscale))
#
#    def test_single_point(self):
#        self.figs.append(plot.plot_time_history(self.time[1:], self.data[1:,:], self.label))
#
#    def test_show_zero(self):
#        self.data += 1000
#        self.opts.show_zero = True
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, opts=self.opts))
#
#    def test_data_lo_and_hi(self):
#        self.figs.append(plot.plot_time_history(self.time, self.data, self.label, \
#            data_lo=self.data-1, data_hi=self.data+1))
#
#    def test_not_ndarray(self):
#        self.figs.append(plot.plot_time_history(0, 0, 'Zero'))
#
#    def test_0d(self):
#        self.figs.append(plot.plot_time_history(np.array(0), np.array(0), 'Zero'))
#
#    def test_1d(self):
#        self.figs.append(plot.plot_time_history(np.arange(5), np.arange(5), 'Line'))
#
#    def test_3d(self):
#        data3 = np.empty((self.data.shape[0], 3, self.data.shape[1]), dtype=float)
#        data3[:,0,:] = self.data
#        data3[:,1,:] = self.data + 0.1
#        data3[:,2,:] = self.data + 0.2
#        self.opts.names = ['Run 1', 'Run 2', 'Run 3']
#        self.figs.append(plot.plot_time_history(self.time, data3, self.label, opts=self.opts))
#
#    def test_bad_4d(self):
#        bad_data = np.random.rand(self.time.shape[0], 4, 5, 1)
#        with self.assertRaises(AssertionError):
#            plot.plot_time_history(self.time, bad_data, self.label, opts=self.opts)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% plotting.plot_correlation_matrix
class Test_plotting_plot_correlation_matrix(unittest.TestCase):
    r"""
    Tests the plotting.plot_correlation_matrix function with the following cases:
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
        self.data   = unit(np.random.rand(num, num), axis=0)
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.units  = 'percentage'
        self.opts   = plot.Opts()
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
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels))

    def test_nonsquare(self):
        self.figs.append(plot.plot_correlation_matrix(self.data[:5, :3], [self.labels[:3], \
            self.labels[:5]]))

    def test_default_labels(self):
        self.figs.append(plot.plot_correlation_matrix(self.data[:5, :3]))

    def test_type(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, units=self.units))

    def test_all_args(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels, self.units, self.opts, \
            matrix_name=self.matrix_name, cmin=0, cmax=1, xlabel='', ylabel='', \
            plot_lower_only=False, label_values=True, x_lab_rot=180, colormap='Paired'))

    def test_symmetric(self):
        self.figs.append(plot.plot_correlation_matrix(self.sym))

    def test_symmetric_all(self):
        self.figs.append(plot.plot_correlation_matrix(self.sym, plot_lower_only=False))

    def test_above_one(self):
        large_data = self.data * 1000
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_above_one_part2(self):
        large_data = self.data * 1000
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmax=2000))

    def test_below_one(self):
        large_data = 1000*(self.data - 0.5)
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_below_one_part2(self):
        large_data = 1000*(self.data - 0.5)
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmin=-2))

    def test_within_minus_one(self):
        large_data = self.data - 0.5
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_within_minus_one_part2(self):
        large_data = self.data - 0.5
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmin=-1, cmax=1))

    def test_xlabel(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, xlabel='Testing Label'))

    def test_ylabel(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, ylabel='Testing Label'))

    def test_x_label_rotation(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels, x_lab_rot=0))

    def test_nans(self):
        self.data[0, 0] = np.nan
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels))

    def test_bad_labels(self):
        with self.assertRaises(ValueError):
            self.figs.append(plot.plot_correlation_matrix(self.data, ['a']))

    def test_label_values(self):
        self.figs.append(plot.plot_correlation_matrix(self.data, label_values=True))

    def tearDown(self):
        for i in range(len(self.figs)):
            plt.close(self.figs.pop())

#%% plotting.plot_bar_breakdown
class Test_plotting_plot_bar_breakdown(unittest.TestCase):
    r"""
    Tests the plotting.plot_bar_breakdown function with the following cases:
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
        mag = np.sum(self.data, axis=1)
        self.data = self.data / np.expand_dims(mag, axis=1)
        self.label = 'Plot bar testing'
        self.legend = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.figs = []

    def test_nominal(self):
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts, \
            legend=self.legend))

    def test_defaults(self):
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label))

    def test_opts(self):
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def test_legend(self):
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend))

    def test_ignore_zeros(self):
        self.data[:, 1] = 0
        self.data[:, 3] = np.nan
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, ignore_empties=True))

    @patch('dstauffman.plotting.plotting.logger')
    def test_null_data(self, mock_logger):
        plot.plot_bar_breakdown(self.time, None, '')
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(LogLevel.L5, '  plot skipped due to missing data.')

    def test_colormap(self):
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, \
            opts=self.opts, colormap=colormap))

    def test_bad_legend(self):
        with self.assertRaises(AssertionError):
            plot.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend[:-1])

    def test_single_point(self):
        self.figs.append(plot.plot_bar_breakdown(self.time[:1], self.data[:1,:], label=self.label))

    def test_new_colormap(self):
        self.opts.colormap = 'seismic'
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% plotting.setup_plots
class Test_plotting_setup_plots(unittest.TestCase):
    r"""
    Tests the plotting.setup_plots function with the following cases:
        Prepend a title
        Don't prepend a title
        Don't show the plot
        Multiple figures
        Save the plot
        Show the plot link
    """
    def setUp(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('X vs Y')
        ax.set_xlabel('time [years]')
        ax.set_ylabel('value [radians]')
        self.opts = plot.Opts()
        self.opts.case_name = 'Testing'
        self.opts.show_plot = True
        self.opts.save_plot = False
        self.opts.save_path = get_tests_dir()

    def test_title(self):
        plot.setup_plots(self.fig, self.opts)

    def test_no_title(self):
        self.opts.case_name = ''
        plot.setup_plots(self.fig, self.opts)

    def test_not_showing_plot(self):
        self.opts.show_plot = False
        plot.setup_plots(self.fig, self.opts)

    def test_multiple_figs(self):
        fig_list = [self.fig]
        (new_fig, ax) = plt.subplots()
        ax.plot(0, 0)
        fig_list.append(new_fig)
        plot.setup_plots(fig_list, self.opts)
        plt.close(new_fig)

    def test_saving_plot(self):
        this_filename = os.path.join(get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        plot.setup_plots(self.fig, self.opts)
        # remove file
        with contextlib.suppress(FileNotFoundError):
            os.remove(this_filename)

    def test_show_link(self):
        this_filename = os.path.join(get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        self.opts.show_link = True
        with capture_output() as out:
            plot.setup_plots(self.fig, self.opts)
        output = out.getvalue().strip()
        out.close()
        # remove file
        with contextlib.suppress(FileNotFoundError):
            os.remove(this_filename)
        self.assertTrue(output.startswith('Plots saved to <a href="'))

    def tearDown(self):
        plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)
