# -*- coding: utf-8 -*-
r"""
Test file for the `plot_generic` module module of the "dstauffman" library.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

#%% Imports
from datetime import datetime
import unittest

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import dstauffman as dcs

#%% Functions - make_time_plot
class Test_make_time_plot(unittest.TestCase):
    r"""
    Tests the make_time_plot function with the following cases:
        TBD
    """
    def setUp(self):
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
        self.fig           = None

    def test_simple(self):
        self.fig = dcs.make_time_plot(self.description, self.time, self.data)

    def test_nominal(self):
        self.fig = dcs.make_time_plot(self.description, self.time, self.data, name=self.name, elements=self.elements, \
            units=self.units, time_units=self.time_units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            single_lines=self.single_lines, colormap=self.colormap, use_mean=self.use_mean, \
            plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, ylabel=self.ylabel, data_as_rows=self.data_as_rows)

    def test_list1(self):
        data = [self.data, self.data+0.5, self.data + 1.0]
        self.fig = dcs.make_time_plot(self.description, self.time, data)

    def test_list2(self):
        time = [self.time, self.time[:-1]]
        data = [self.data, 2*self.data[:-1]]
        self.fig = dcs.make_time_plot(self.description, time, data)

    def test_vectors(self):
        data = np.vstack((self.data, np.sin(self.time)))
        self.fig = dcs.make_time_plot(self.description, self.time, data)

    def tearDown(self):
        if self.fig:
            plt.close(self.fig)

#%% Functions - make_error_bar_plot
pass # TODO: write this

#%% Functions - make_quaternion_plot
class Test_make_quaternion_plot(unittest.TestCase):
    r"""
    Tests the make_quaternion_plot function with the following cases:
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
        self.make_subplots   = True
        self.plot_components = True
        self.use_mean        = False
        self.plot_zero       = False
        self.show_rms        = True
        self.return_err      = True
        self.figs            = None

    def test_nominal(self):
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_subplots(self):
        self.make_subplots = False
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_components(self):
        self.plot_components = False
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)
        self.assertLess(abs(err['mag']), 3.15)

    def test_no_start_date(self):
        self.start_date = ''
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_only_quat_one(self):
        self.quat_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_quat_two(self):
        self.quat_one = None
        self.name_one = ''
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self):
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_use_mean(self):
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True)

    def test_no_rms_in_legend(self):
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True, show_rms=False)

    def test_plot_zero(self):
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, plot_zero=True)

    def test_plot_truth(self):
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, truth_time=self.time_one, truth_data=self.quat_two)

    def test_disp_bounds(self):
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, disp_xmin=2, disp_xmax=5)

    def test_no_overlap(self):
        self.time_one = np.arange(11).astype(float)
        self.time_two = np.arange(2, 13) + 0.5
        self.figs = dcs.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Functions - make_difference_plot
class Test_make_difference_plot(unittest.TestCase):
    r"""
    Tests the make_difference_plot function with the following cases:
        TBD
    """
    def setUp(self):
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
        self.start_date    = str(datetime.now())
        self.rms_xmin      = 0
        self.rms_xmax      = 10
        self.disp_xmin     = -2
        self.disp_xmax     = np.inf
        self.make_subplots = True
        color_lists        = dcs.get_color_lists()
        self.colormap      = ListedColormap(color_lists['dbl_diff'].colors + color_lists['double'].colors)
        self.use_mean      = False
        self.plot_zero     = False
        self.show_rms      = True
        self.legend_loc    = 'best'
        self.second_yscale = {u'µrad': 1e6}
        self.return_err    = True
        self.figs          = None

    def test_nominal(self):
        self.return_err = False
        self.figs = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_subplots(self):
        self.make_subplots = False
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_start_date(self):
        self.start_date = ''
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_only_data_one(self):
        self.data_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_data_two(self):
        self.data_one = None
        self.name_one = ''
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self):
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_use_mean(self):
        self.use_mean = True
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_no_rms_in_legend(self):
        self.show_rms = False
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_plot_zero(self):
        self.plot_zero = True
        (self.figs, err) = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def test_plot_truth(self):
        self.figs = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, elements=self.elements, units=self.units, \
            truth_time=self.time_one, truth_data=self.data_two)

    def test_disp_bounds(self):
        self.figs = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
             self.data_one, self.data_two, elements=self.elements, units=self.units, \
             disp_xmin=2, disp_xmax=5)

    def test_no_overlap(self):
        self.time_one = np.arange(11).astype(float)
        self.time_two = np.arange(2, 13) + 0.5
        self.return_err = False
        self.figs = dcs.make_difference_plot(self.description, self.time_one, self.time_two, \
            self.data_one, self.data_two, name_one=self.name_one, name_two=self.name_two, \
            elements=self.elements, units=self.units, leg_scale=self.leg_scale, start_date=self.start_date, \
            rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, \
            make_subplots=self.make_subplots, colormap=self.colormap, \
            use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, legend_loc=self.legend_loc, \
            second_yscale=self.second_yscale, return_err=self.return_err)

    def tearDown(self):
        if self.figs:
            for this_fig in self.figs:
                plt.close(this_fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)