r"""
Test file for the `aerospace` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import datetime
from typing import List, Optional
import unittest
from unittest.mock import patch

from dstauffman import HAVE_MPL, HAVE_NUMPY, LogLevel
from dstauffman.aerospace import quat_norm
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np

#%% plotting.make_quaternion_plot
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_make_quaternion_plot(unittest.TestCase):
    r"""
    Tests the plotting.make_quaternion_plot function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.description     = 'example'
        self.time_one        = np.arange(11)
        self.time_two        = np.arange(2, 13)
        self.quat_one        = quat_norm(np.random.rand(4, 11))
        self.quat_two        = quat_norm(np.random.rand(4, 11))
        self.name_one        = 'test1'
        self.name_two        = 'test2'
        self.time_units       = 'sec'
        self.start_date      = str(datetime.datetime.now())
        self.plot_components  = True
        self.rms_xmin        = 0
        self.rms_xmax        = 10
        self.disp_xmin       = -2
        self.disp_xmax       = np.inf
        self.make_subplots    = True
        self.single_lines     = False
        self.use_mean         = False
        self.plot_zero        = False
        self.show_rms         = True
        self.legend_loc       = 'best'
        self.show_extra       = True
        self.second_units     = (u'µrad', 1e6)
        self.truth_name       = 'Truth'
        self.truth_time       = None
        self.truth_data       = None
        self.data_as_rows     = True
        self.tolerance        = 0
        self.return_err       = False
        self.use_zoh          = False
        self.label_vert_lines = True
        self.figs: Optional[List[Figure]] = None

    def test_nominal(self) -> None:
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_subplots(self) -> None:
        self.make_subplots = False
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_no_components(self) -> None:
        self.plot_components = False
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)
        self.assertLess(abs(err['mag']), 3.15)

    def test_no_start_date(self) -> None:
        self.start_date = ''
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_only_quat_one(self) -> None:
        self.quat_two.fill(np.nan)
        self.name_two = ''
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_only_quat_two(self) -> None:
        self.quat_one = None
        self.name_one = ''
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        self.assertTrue(np.all(np.isnan(err['diff'])))

    def test_rms_bounds(self) -> None:
        self.rms_xmin = 5
        self.rms_xmax = 7
        self.return_err = True
        (self.figs, err) = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, self.quat_one, self.quat_two,
            name_one=self.name_one, name_two=self.name_two, time_units=self.time_units, \
            start_date=self.start_date, plot_components=self.plot_components, rms_xmin=self.rms_xmin, \
            rms_xmax=self.rms_xmax, disp_xmin=self.disp_xmin, disp_xmax=self.disp_xmax, make_subplots=self.make_subplots, \
            single_lines=self.single_lines, use_mean=self.use_mean, plot_zero=self.plot_zero, show_rms=self.show_rms, \
            legend_loc=self.legend_loc, show_extra=self.show_extra, truth_name=self.truth_name, truth_time=self.truth_time, \
            truth_data=self.truth_data, data_as_rows=self.data_as_rows, tolerance=self.tolerance, \
            return_err=self.return_err, use_zoh=self.use_zoh, label_vert_lines=self.label_vert_lines)
        for i in range(3):
            self.assertLess(abs(err['diff'][i]), 3.15)

    def test_use_mean(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True)

    def test_no_rms_in_legend(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, use_mean=True, show_rms=False)

    def test_plot_zero(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, plot_zero=True)

    def test_plot_truth(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, truth_time=self.time_one, truth_data=self.quat_two)

    def test_disp_bounds(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two, disp_xmin=2, disp_xmax=5)

    def test_no_overlap(self) -> None:
        self.time_one = np.arange(11).astype(float)
        self.time_two = np.arange(2, 13) + 0.5
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, self.time_two, \
             self.quat_one, self.quat_two)

    def test_none1(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, self.time_one, None, self.quat_one, None)

    def test_none2(self) -> None:
        self.figs = plot.make_quaternion_plot(self.description, None, self.time_two, None, self.quat_two)

    @patch('dstauffman.plotting.aerospace.logger')
    def test_none3(self, mock_logger):
        self.figs = plot.make_quaternion_plot('', None, None, None, None)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L5, 'No quaternion data was provided, so no plot was generated for "".')

    def tearDown(self) -> None:
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

#%% plotting.plot_innov_fplocs
class Test_plotting_plot_innov_fplocs(unittest.TestCase):
    r"""
    Tests the plotting.plot_innov_fplocs function with the following cases:
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
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
