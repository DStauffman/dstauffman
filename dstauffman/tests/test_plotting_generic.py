r"""
Test file for the `generic` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

#%% Imports
import datetime
from typing import List, Optional
import unittest
from unittest.mock import patch

from dstauffman import HAVE_MPL, HAVE_NUMPY, LogLevel
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

#%% plotting.make_time_plot
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_make_time_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_time_plot function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.description = 'Values vs Time'
        self.time          = np.arange(-10., 10.1, 0.1)
        self.data          = self.time + np.cos(self.time)
        self.name          = ''
        self.elements      = None
        self.units         = ''
        self.time_units    = 'sec'
        self.leg_scale     = 'unity'
        self.start_date    = ''
        self.rms_xmin      = -np.inf
        self.rms_xmax      = np.inf
        self.disp_xmin     = -np.inf
        self.disp_xmax     = np.inf
        self.single_lines  = False
        self.colormap      = 'Paired'
        self.use_mean      = False
        self.plot_zero     = False
        self.show_rms      = True
        self.legend_loc    = 'best'
        self.second_yscale = None
        self.ylabel        = None
        self.data_as_rows  = True
        self.figs: Optional[List[plt.Figure]] = None

    def test_simple(self) -> None:
        self.fig = plot.make_time_plot(self.description, self.time, self.data)

    def test_nominal(self) -> None:
        self.fig = plot.make_time_plot(self.description, self.time, self.data, name=self.name, elements=self.elements, \
            units=self.units, time_units=self.time_units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            single_lines=self.single_lines, colormap=self.colormap, use_mean=self.use_mean, \
            plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, ylabel=self.ylabel, data_as_rows=self.data_as_rows)

    def test_list1(self) -> None:
        data = [self.data, self.data+0.5, self.data + 1.0]
        self.fig = plot.make_time_plot(self.description, self.time, data)

    def test_list2(self) -> None:
        time = [self.time, self.time[:-1]]
        data = [self.data, 2*self.data[:-1]]
        self.fig = plot.make_time_plot(self.description, time, data)

    def test_vectors(self) -> None:
        data = np.vstack((self.data, np.sin(self.time)))
        self.fig = plot.make_time_plot(self.description, self.time, data)

    def tearDown(self) -> None:
        if self.fig:
            plt.close(self.fig)

#%% plotting.make_error_bar_plot
class Test_plotting_make_error_bar_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_error_bar_plot function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.make_difference_plot
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_make_difference_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_difference_plot function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.description   = 'example'
        self.time_one      = np.arange(11)
        self.time_two      = np.arange(2, 13)
        self.data_one      = 1e-6 * np.random.rand(2, 11)
        self.data_two      = self.data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6
        self.name_one      = 'test1'
        self.name_two      = 'test2'
        self.elements      = ['x', 'y']
        self.units         = 'rad'
        self.leg_scale     = 'micro'
        self.start_date    = str(datetime.datetime.now())
        self.rms_xmin      = 0
        self.rms_xmax      = 10
        self.disp_xmin     = -2
        self.disp_xmax     = np.inf
        self.make_subplots = True
        color_lists        = plot.get_color_lists()
        self.colormap      = ListedColormap(color_lists['dbl_diff'].colors + color_lists['double'].colors)  # type: ignore[attr-defined]
        self.use_mean      = False
        self.plot_zero     = False
        self.show_rms      = True
        self.legend_loc    = 'best'
        self.second_yscale = {u'µrad': 1e6}
        self.return_err    = True
        self.figs: Optional[List[plt.Figure]] = None

    def test_nominal(self) -> None:
        self.return_err = False
        self.figs = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_subplots(self) -> None:
        self.make_subplots = False
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_start_date(self) -> None:
        self.start_date = ''
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_only_data_one(self) -> None:
        self.data_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_data_two(self) -> None:
        self.data_one = None
        self.name_one = ''
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self) -> None:
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_use_mean(self) -> None:
        self.use_mean = True
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_rms_in_legend(self) -> None:
        self.show_rms = False
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_plot_zero(self) -> None:
        self.plot_zero = True
        (self.figs, err) = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_plot_truth(self) -> None:
        self.figs = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, elements=self.elements, units=self.units, \
            truth_time=self.time_one, truth_data=self.data_two)

    def test_disp_bounds(self) -> None:
        self.figs = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
             self.data_one, self.data_two, elements=self.elements, units=self.units, \
             disp_xmin=2, disp_xmax=5)

    def test_no_overlap(self) -> None:
        self.time_one = np.arange(11).astype(float)
        self.time_two = np.arange(2, 13) + 0.5
        self.return_err = False
        self.figs = plot.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_none1(self) -> None:
        self.figs = plot.make_difference_plot(self.description, self.time_one, None, self.data_one, None)

    def test_none2(self) -> None:
        self.figs = plot.make_difference_plot(self.description, None, self.time_two, None, self.data_two)

    @patch('dstauffman.plotting.generic.logger')
    def test_none3(self, mock_logger):
        self.figs = plot.make_difference_plot('', None, None, None, None)
        self.assertEqual(mock_logger.log.call_count, 1)
        mock_logger.log.assert_called_with(LogLevel.L5, 'No difference data was provided, so no plot was generated for "".')

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% plotting.make_categories_plot
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_make_categories_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_categories_plot with the following cases:
        Nominal
        Minimal
        TBD
    """
    def setUp(self) -> None:
        self.description = 'Values vs Time'
        self.time = np.arange(-10., 10.1, 0.1)
        self.data = self.time + np.cos(self.time)
        MeasStatus = type('MeasStatus', (object,), {'rejected': 0, 'accepted': 1})
        self.cats = np.full(self.time.shape, MeasStatus.accepted, dtype=int)  # type: ignore[attr-defined]
        self.cats[50:100] = MeasStatus.rejected  # type: ignore[attr-defined]
        self.cat_names = {0: 'rejected', 1: 'accepted'}
        self.name = ''
        self.elements = None
        self.units = ''
        self.time_units = 'sec'
        self.leg_scale = 'unity'
        self.start_date = ''
        self.rms_xmin = -np.inf
        self.rms_xmax = np.inf
        self.disp_xmin = -np.inf
        self.disp_xmax = np.inf
        self.single_plots = False
        self.colormap = 'Paired'
        self.use_mean = False
        self.plot_zero = False
        self.show_rms = True
        self.legend_loc = 'best'
        self.second_yscale = None
        self.ylabel = None
        self.data_as_rows = True
        self.figs: Optional[List[plt.Figure]] = None

    def test_nominal(self) -> None:
        self.figs = plot.make_categories_plot(self.description, self.time, self.data, self.cats, \
            cat_names=self.cat_names, name=self.name, elements=self.elements, units=self.units, \
            time_units=self.time_units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, \
            disp_xmax=self.disp_xmax, single_plots=self.single_plots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, second_yscale=self.second_yscale, ylabel=self.ylabel, \
            data_as_rows=self.data_as_rows)

    def test_minimal(self) -> None:
        self.figs = plot.make_categories_plot(self.description, self.time, self.data, self.cats)

    def tearDown(self) -> None:
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Unit test execution
if __name__ == '__main__':
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
