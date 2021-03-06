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
from typing import List
import unittest
from unittest.mock import patch

from dstauffman import capture_output, get_tests_dir, HAVE_MPL, HAVE_NUMPY, LogLevel, unit
import dstauffman.plotting as plot

if HAVE_MPL:
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np
    inf = np.inf
else:
    from math import inf

#%% plotting.Opts
class Test_plotting_Opts(unittest.TestCase):
    r"""
    Tests the plotting.Opts class with the following cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self) -> None:
        self.opts_fields = ['case_name']

    def test_calling(self) -> None:
        opts = plot.Opts()
        for field in self.opts_fields:
            self.assertTrue(hasattr(opts, field))

    def test_new_attr(self) -> None:
        opts = plot.Opts()
        with self.assertRaises(AttributeError):
            opts.new_field_that_does_not_exist = 1  # type: ignore[attr-defined]

    def test_get_names_successful(self) -> None:
        opts = plot.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(0)
        self.assertEqual(name, 'Name 1')

    def test_get_names_unsuccessful(self) -> None:
        opts = plot.Opts()
        opts.names = ['Name 1', 'Name 2']
        name = opts.get_names(2)
        self.assertEqual(name, '')

    def test_get_date_zero_str(self) -> None:
        opts = plot.Opts()
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str, '')
        opts.date_zero = datetime.datetime(2019, 4, 1, 18, 0, 0)
        date_str = opts.get_date_zero_str()
        self.assertEqual(date_str,'  t(0) = 01-Apr-2019 18:00:00 Z')

    def get_time_limits(self) -> None:
        opts = plot.Opts()
        opts.disp_xmin = 60
        opts.disp_xmax = inf
        opts.rms_xmin = -inf
        opts.rms_xmax = None
        opts.time_base = 'sec'
        opts.time_unit = 'min'
        (d1, d2, r1, r2) = opts.get_time_limits()
        self.assertEqual(d1, 1)
        self.assertEqual(d2, inf)
        self.assertEqual(r1, -inf)
        self.assertIsNone(r2)

    def get_time_limits2(self) -> None:
        opts = plot.Opts().convert_dates('datetime')
        opts.disp_xmin = datetime.datetime(2020, 6, 1, 0, 0, 0)
        opts.disp_xmax = datetime.datetime(2020, 6, 1, 12, 0, 0)
        (d1, d2, r1, r2) = opts.get_time_limits()
        self.assertEqual(d1, datetime.datetime(2020, 6, 1, 0, 0, 0))
        self.assertEqual(d2, datetime.datetime(2020, 6, 1, 12, 0, 0))
        self.assertIsNone(r1)
        self.assertIsNone(r2)

    def test_pprint(self) -> None:
        opts = plot.Opts()
        with capture_output() as out:
            opts.pprint(indent=2)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Opts')
        self.assertEqual(lines[1], '  case_name = ')
        self.assertEqual(lines[3], '  save_plot = False')
        self.assertEqual(lines[-1], '  names     = []')

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_convert_dates(self) -> None:
        opts = plot.Opts()
        self.assertEqual(opts.disp_xmin, -inf)
        self.assertEqual(opts.time_base, 'sec')
        opts.convert_dates('datetime')
        self.assertIsNone(opts.disp_xmin)
        self.assertEqual(opts.time_base, 'datetime')

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_convert_dates2(self) -> None:
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

#%% plotting.suppress_plots and plotting.unsupress_plots
class Test_plotting_Plotter(unittest.TestCase):
    r"""
    Tests the plotting.Plotter class with the following cases:
        Suppress and Unsuppress
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_flag = plot.plotting._Plotter  # type: ignore[attr-defined]

    def test_suppress_and_unsupress(self) -> None:
        plot.suppress_plots()
        self.assertFalse(plot.plotting._Plotter)
        plot.unsuppress_plots()
        self.assertTrue(plot.plotting._Plotter)

    def tearDown(self) -> None:
        if self.orig_flag:  # type: ignore[attr-defined]
            plot.unsuppress_plots()
        else:
            plot.suppress_plots()

#%% plotting.plot_time_history
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
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
    def setUp(self) -> None:
        self.description = 'Plot description'
        self.time        = np.arange(0, 10, 0.1) + 2000
        num_channels     = 5
        self.row_data    = np.random.rand(len(self.time), num_channels)
        mag              = np.sum(self.row_data, axis=1)
        self.row_data    = 10 * self.row_data / np.expand_dims(mag, axis=1)
        self.col_data    = self.row_data.T.copy()
        self.units       = 'percentage'
        self.opts        = plot.Opts()
        self.opts.show_plot = False
        self.elements    = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.figs: List[plt.Figure] = []
        self.second_yscale = 1000000

    def test_nominal(self) -> None:
        self.figs.append(plot.plot_time_history(self.description, self.time, self.row_data, \
            opts=self.opts, data_as_rows=False))

    def test_defaults(self) -> None:
        self.figs.append(plot.plot_time_history('', self.time, self.col_data))

    def test_with_units(self) -> None:
        self.figs.append(plot.plot_time_history(self.description, self.time, self.col_data, units=self.units))

    def test_with_opts(self) -> None:
        self.figs.append(plot.plot_time_history(self.description, self.time, self.col_data, opts=self.opts))

    @patch('dstauffman.plotting.plotting.logger')
    def test_no_data(self, mock_logger):
        plot.plot_time_history('', self.time, None)
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(LogLevel.L5, '  plot skipped due to missing data.')

    def test_ignore_zeros(self) -> None:
        self.figs.append(plot.plot_time_history(self.description, self.time, self.col_data, ignore_empties=True))

    def test_ignore_zeros2(self) -> None:
        self.col_data[1, :] = 0
        self.col_data[3, :] = 0
        self.figs.append(plot.plot_time_history(self.description, self.time, self.col_data, ignore_empties=True))

    @patch('dstauffman.plotting.plotting.logger')
    def test_ignore_zeros3(self, mock_logger):
        self.col_data = np.zeros(self.col_data.shape)
        not_a_fig = plot.plot_time_history('All Zeros', self.time, self.col_data, ignore_empties=True)
        self.assertIs(not_a_fig, None)
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(LogLevel.L5, ' All Zeros plot skipped due to missing data.')

    def test_not_ndarray(self) -> None:
        self.figs.append(plot.plot_time_history('Zero', 0, 0))

    def test_0d(self) -> None:
        self.figs.append(plot.plot_time_history('Zero', np.array(0), np.array(0)))

    def test_1d(self) -> None:
        self.figs.append(plot.plot_time_history('Line', np.arange(5), np.arange(5)))

    def test_bad_3d(self) -> None:
        bad_data = np.random.rand(self.time.shape[0], 4, 5)
        with self.assertRaises(AssertionError):
            plot.plot_time_history(self.description, self.time, bad_data, opts=self.opts)

    def test_datetime(self) -> None:
        dates = np.datetime64('2020-01-11 12:00:00') + np.arange(0, 1000, 10).astype('timedelta64[ms]')
        self.figs.append(plot.plot_time_history(self.description, dates, self.col_data, opts=self.opts, \
            time_units='numpy'))

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% plotting.plot_correlation_matrix
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
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
    def setUp(self) -> None:
        num = 10
        self.figs: List[plt.Figure] = []
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

    def test_normal(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels))

    def test_nonsquare(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data[:5, :3], [self.labels[:3], \
            self.labels[:5]]))

    def test_default_labels(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data[:5, :3]))

    def test_type(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, units=self.units))

    def test_all_args(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels, self.units, self.opts, \
            matrix_name=self.matrix_name, cmin=0, cmax=1, xlabel='', ylabel='', \
            plot_lower_only=False, label_values=True, x_lab_rot=180, colormap='Paired'))

    def test_symmetric(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.sym))

    def test_symmetric_all(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.sym, plot_lower_only=False))

    def test_above_one(self) -> None:
        large_data = self.data * 1000
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_above_one_part2(self) -> None:
        large_data = self.data * 1000
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmax=2000))

    def test_below_one(self) -> None:
        large_data = 1000*(self.data - 0.5)
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_below_one_part2(self) -> None:
        large_data = 1000*(self.data - 0.5)
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmin=-2))

    def test_within_minus_one(self) -> None:
        large_data = self.data - 0.5
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels))

    def test_within_minus_one_part2(self) -> None:
        large_data = self.data - 0.5
        self.figs.append(plot.plot_correlation_matrix(large_data, self.labels, cmin=-1, cmax=1))

    def test_xlabel(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, xlabel='Testing Label'))

    def test_ylabel(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, ylabel='Testing Label'))

    def test_x_label_rotation(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels, x_lab_rot=0))

    def test_nans(self) -> None:
        self.data[0, 0] = np.nan
        self.figs.append(plot.plot_correlation_matrix(self.data, self.labels))

    def test_bad_labels(self) -> None:
        with self.assertRaises(ValueError):
            self.figs.append(plot.plot_correlation_matrix(self.data, ['a']))

    def test_label_values(self) -> None:
        self.figs.append(plot.plot_correlation_matrix(self.data, label_values=True))

    def tearDown(self) -> None:
        for i in range(len(self.figs)):
            plt.close(self.figs.pop())

#%% plotting.plot_bar_breakdown
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
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
    def setUp(self) -> None:
        self.time = np.arange(0, 5, 1./12) + 2000
        num_bins = 5
        self.data = np.random.rand(len(self.time), num_bins)
        mag = np.sum(self.data, axis=1)
        self.data = self.data / np.expand_dims(mag, axis=1)
        self.label = 'Plot bar testing'
        self.legend = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.figs: List[plt.Figure] = []

    def test_nominal(self) -> None:
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts, \
            legend=self.legend))

    def test_defaults(self) -> None:
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label))

    def test_opts(self) -> None:
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def test_legend(self) -> None:
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend))

    def test_ignore_zeros(self) -> None:
        self.data[:, 1] = 0
        self.data[:, 3] = np.nan
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, ignore_empties=True))

    @patch('dstauffman.plotting.plotting.logger')
    def test_null_data(self, mock_logger):
        plot.plot_bar_breakdown(self.time, None, '')
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(LogLevel.L5, '  plot skipped due to missing data.')

    def test_colormap(self) -> None:
        self.opts.colormap = 'Dark2'
        colormap = 'Paired'
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, \
            opts=self.opts, colormap=colormap))

    def test_bad_legend(self) -> None:
        with self.assertRaises(AssertionError):
            plot.plot_bar_breakdown(self.time, self.data, label=self.label, legend=self.legend[:-1])

    def test_single_point(self) -> None:
        self.figs.append(plot.plot_bar_breakdown(self.time[:1], self.data[:1,:], label=self.label))

    def test_new_colormap(self) -> None:
        self.opts.colormap = 'seismic'
        self.figs.append(plot.plot_bar_breakdown(self.time, self.data, label=self.label, opts=self.opts))

    def test_datetime(self) -> None:
        dates = np.datetime64('2020-01-11 12:00:00') + np.arange(0, 7200, 120).astype('timedelta64[s]')
        self.figs.append(plot.plot_bar_breakdown(dates, self.data, label=self.label, opts=self.opts, \
        time_units='numpy'))

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% plotting.setup_plots
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
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
    def setUp(self) -> None:
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

    def test_title(self) -> None:
        plot.setup_plots(self.fig, self.opts)

    def test_no_title(self) -> None:
        self.opts.case_name = ''
        plot.setup_plots(self.fig, self.opts)

    def test_not_showing_plot(self) -> None:
        self.opts.show_plot = False
        plot.setup_plots(self.fig, self.opts)

    def test_multiple_figs(self) -> None:
        fig_list = [self.fig]
        (new_fig, ax) = plt.subplots()
        ax.plot(0, 0)
        fig_list.append(new_fig)
        plot.setup_plots(fig_list, self.opts)
        plt.close(new_fig)

    def test_saving_plot(self) -> None:
        this_filename = os.path.join(get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        plot.setup_plots(self.fig, self.opts)
        # remove file
        with contextlib.suppress(FileNotFoundError):
            os.remove(this_filename)

    def test_show_link(self) -> None:
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

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plot.suppress_plots()
    unittest.main(exit=False)
