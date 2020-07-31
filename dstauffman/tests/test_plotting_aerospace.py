r"""
Test file for the `aerospace` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import datetime
import unittest
from unittest.mock import patch

import numpy as np

from dstauffman import LogLevel
from dstauffman.aerospace import quat_norm

import dstauffman.plotting as plot

#%% plotting.make_quaternion_plot
class Test_plotting_make_quaternion_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_quaternion_plot function with the following cases:
        TBD
    """
    def setUp(self):
        self.description     = 'example'
        self.time_one        = np.arange(11)
        self.time_two        = np.arange(2, 13)
        self.quat_one        = quat_norm(np.random.rand(4, 11))
        self.quat_two        = quat_norm(np.random.rand(4, 11))
        self.name_one        = 'test1'
        self.name_two        = 'test2'
        self.start_date      = str(datetime.datetime.now())
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
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_subplots(self):
        self.make_subplots = False
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_components(self):
        self.plot_components = False
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)
        self.assertLess(abs(err['mag']), 3.15)

    def test_no_start_date(self):
        self.start_date = ''
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_only_quat_one(self):
        self.quat_two.fill(np.nan)
        self.name_two = ''
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_quat_two(self):
        self.quat_one = None
        self.name_one = ''
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self):
        self.rms_xmin = 5
        self.rms_xmax = 7
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, name_one=self.name_one, name_two=self.name_two, \
             rms_xmin=self.rms_xmin, rms_xmax=self.rms_xmax, start_date=self.start_date, \
             make_subplots=self.make_subplots, plot_components=self.plot_components, \
             return_err=self.return_err)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_use_mean(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True)

    def test_no_rms_in_legend(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True, show_rms=False)

    def test_plot_zero(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, plot_zero=True)

    def test_plot_truth(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, truth_time=self.time_one, truth_data=self.quat_two)

    def test_disp_bounds(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, disp_xmin=2, disp_xmax=5)

    def test_no_overlap(self):
        self.time_one = np.arange(11).astype(float)
        self.time_two = np.arange(2, 13) + 0.5
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two)

    def test_none1(self):
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, None, self.quat_one, None)

    def test_none2(self):
        self.figs = plot.make_quaternion_plot(self.description, None, self.time_two, None, self.quat_two)

    @patch('dstauffman.plotting.aerospace.logger')
    def test_none3(self, mock_logger):
        self.figs = plot.make_quaternion_plot('', None, None, None, None)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L5, 'No quaternion data was provided, so no plot was generated for "".')

    def tearDown(self):
        if self.figs:
            plot.close_all(self.figs)

#%% plotting.plot_attitude
class Test_plotting_plot_attitude(unittest.TestCase):
    r"""
    Tests the plotting.plot_attitude function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_los
class Test_plotting_plot_los(unittest.TestCase):
    r"""
    Tests the plotting.plot_los function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_position
class Test_plotting_plot_position(unittest.TestCase):
    r"""
    Tests the plotting.plot_position function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_velocity
class Test_plotting_plot_velocity(unittest.TestCase):
    r"""
    Tests the plotting.plot_velocity function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_innovations
class Test_plotting_plot_innovations(unittest.TestCase):
    r"""
    Tests the plotting.plot_innovations function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_covariance
class Test_plotting_plot_covariance(unittest.TestCase):
    r"""
    Tests the plotting.plot_covariance function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.plot_states
class Test_plotting_plot_states(unittest.TestCase):
    r"""
    Tests the plotting.plot_states function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
