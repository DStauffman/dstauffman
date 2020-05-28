# -*- coding: utf-8 -*-
r"""
Test file for the `time` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

#%% Imports
import unittest
import datetime

import numpy as np
import pandas as pd # TODO: this is only a dependency for testing

import dstauffman as dcs

#%% get_np_time_units
class Test_get_np_time_units(unittest.TestCase):
    r"""
    Tests the get_np_time_units with the following cases
        Default
        nanosecond
    """
    def test_nominal(self):
        units = dcs.get_np_time_units(np.datetime64(datetime.datetime.now()))
        self.assertEqual(units, 'us')

    def test_nanoseconds(self):
        units = dcs.get_np_time_units(np.datetime64(datetime.datetime(2020, 1, 1), 'ns'))
        self.assertEqual(units, 'ns')

#%% round_datetime
class Test_round_datetime(unittest.TestCase):
    r"""
    Tests the round_datetime function with these cases:
        normal use (round to one minute)
        extended use (round to a different specified time)
        get current time
    """
    def test_normal_use(self):
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 10))
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 4, 0))

    def test_extended_use(self):
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 10), round_to_sec=300)
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 5, 0))

    def test_current_time(self):
        dcs.round_datetime()
        self.assertTrue(True)

    def test_flooring(self):
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 45), round_to_sec=60, floor=True)
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 4, 0))

#%% Functions - round_np_datetime
class Test_round_np_datetime(unittest.TestCase):
    r"""
    Tests the round_np_datetime function with these cases:
        Nominal
        Flooring
    """
    def setUp(self):
        date_zero       = np.datetime64(datetime.date(2020, 1, 1))
        dt_sec          = np.array([0, 0.2, 0.35, 0.45, 0.59, 0.61])
        self.date_in    = date_zero + np.round(1000*dt_sec).astype('timedelta64[ms]')
        self.time_delta = np.timedelta64(200, 'ms')
        self.expected   = date_zero + np.array([0, 200, 400, 400, 600, 600]).astype('timedelta64[ms]')

    def test_nominal(self):
        date_out   = dcs.round_np_datetime(self.date_in, self.time_delta)
        np.testing.assert_array_equal(date_out, self.expected)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
