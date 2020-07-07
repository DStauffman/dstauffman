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

import matplotlib.dates as dates
import numpy as np

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

#%% Functions - round_num_datetime
class Test_round_num_datetime(unittest.TestCase):
    r"""
    Tests the round_num_datetime function with the following cases:
        TBD
    """
    def setUp(self):
        self.date_exact = np.arange(0, 10.1, 0.1)
        self.date_in    = self.date_exact + 0.001 * np.random.rand(101)
        self.time_delta = 0.1

    def test_nominal(self):
        date_out   = dcs.round_num_datetime(self.date_in, self.time_delta)
        np.testing.assert_array_almost_equal(date_out, self.date_exact, 12)

    def test_small(self):
        with self.assertWarns(Warning):
            dcs.round_num_datetime(1e6 * self.date_in, 1e-12)

    def test_floor(self):
        date_out = dcs.round_num_datetime(np.array([0., 1.1, 1.9, 3.05, 4.9]), 1., floor=True)
        np.testing.assert_array_almost_equal(date_out, np.array([0., 1., 1., 3., 4.]), 12)

#%% Functions - convert_date
class Test_convert_date(unittest.TestCase):
    r"""
    Tests the convert_date function with the following cases:
        seconds
        datetimes
        numpys
        matplotlibs
        infs and nans
        nats
        no date_zero
        alternative numpy forms
    """
    def setUp(self):
        self.seconds    = 3725.5
        self.date_zero  = datetime.datetime(2020, 6, 1, 0, 0, 0)
        self.datetime   = datetime.datetime(2020, 6, 1, 1, 2, 5, 500000)
        self.numpy      = np.datetime64('2020-06-01 01:02:05.500000', 'ns')
        self.matplotlib = dates.date2num(self.datetime)
        self.nat        = np.datetime64('nat')

    def test_secs(self):
        out = dcs.convert_date(self.seconds, 'datetime', self.date_zero)
        self.assertEqual(out, self.datetime)
        out = dcs.convert_date(self.seconds, 'numpy', self.date_zero)
        self.assertEqual(out, self.numpy)
        out = dcs.convert_date(self.seconds, 'matplotlib', self.date_zero)
        self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.seconds, 'sec', self.date_zero)
        self.assertEqual(out, self.seconds)

    def test_datetimes(self):
        out = dcs.convert_date(self.datetime, 'datetime', old_form='datetime')
        self.assertEqual(out, self.datetime)
        out = dcs.convert_date(self.datetime, 'numpy', old_form='datetime')
        self.assertEqual(out, self.numpy)
        out = dcs.convert_date(self.datetime, 'matplotlib', old_form='datetime')
        self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.datetime, 'sec', self.date_zero, old_form='datetime')
        self.assertEqual(out, self.seconds)

    def test_numpys(self):
        out = dcs.convert_date(self.numpy, 'datetime', old_form='numpy')
        self.assertEqual(out, self.datetime)
        out = dcs.convert_date(self.numpy, 'numpy', old_form='numpy')
        self.assertEqual(out, self.numpy)
        out = dcs.convert_date(self.numpy, 'matplotlib', old_form='numpy')
        self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.numpy, 'sec', self.date_zero, old_form='numpy')
        self.assertEqual(out, self.seconds)

    def test_matplotlibs(self):
        out = dcs.convert_date(self.matplotlib, 'datetime', old_form='matplotlib')
        exp = self.datetime.replace(tzinfo=datetime.timezone.utc)
        self.assertEqual(out, exp)
        out = dcs.convert_date(self.matplotlib, 'numpy', old_form='matplotlib')
        self.assertEqual(out, self.numpy)
        out = dcs.convert_date(self.matplotlib, 'matplotlib', old_form='matplotlib')
        self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.matplotlib, 'sec', self.date_zero, old_form='matplotlib')
        self.assertAlmostEqual(out, self.seconds, 6)

    def test_infs_and_nans(self):
        out = dcs.convert_date(np.inf, 'datetime')
        self.assertIsNone(out)
        out = dcs.convert_date(np.inf, 'numpy')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(np.inf, 'matplotlib')
        self.assertEqual(out, np.inf)
        out = dcs.convert_date(np.inf, 'sec')
        self.assertEqual(out, np.inf)
        out = dcs.convert_date(-np.inf, 'datetime')
        self.assertIsNone(out)
        out = dcs.convert_date(-np.inf, 'numpy')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(-np.inf, 'matplotlib')
        self.assertEqual(out, -np.inf)
        out = dcs.convert_date(-np.inf, 'sec')
        self.assertEqual(out, -np.inf)
        out = dcs.convert_date(np.nan, 'datetime')
        self.assertIsNone(out)
        out = dcs.convert_date(np.nan, 'numpy')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(np.nan, 'matplotlib')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(np.nan, 'sec')
        self.assertTrue(np.isnan(out))

    def test_nats(self):
        out = dcs.convert_date(None, 'datetime', old_form='datetime')
        self.assertIsNone(out)
        out = dcs.convert_date(None, 'numpy', old_form='datetime')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(None, 'matplotlib', old_form='datetime')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(None, 'sec', self.date_zero, old_form='datetime')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(self.nat, 'datetime', old_form='numpy')
        self.assertIsNone(out)
        out = dcs.convert_date(self.nat, 'numpy', old_form='numpy')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(self.nat, 'matplotlib', old_form='numpy')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(self.nat, 'sec', self.date_zero, old_form='numpy')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(np.inf, 'datetime', old_form='matplotlib')
        self.assertIsNone(out)
        out = dcs.convert_date(np.inf, 'numpy', old_form='matplotlib')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(np.inf, 'matplotlib', old_form='matplotlib')
        self.assertEqual(out, np.inf)
        out = dcs.convert_date(np.inf, 'sec', self.date_zero, old_form='matplotlib')
        self.assertEqual(out, np.inf)
        out = dcs.convert_date(np.nan, 'datetime', old_form='matplotlib')
        self.assertIsNone(out)
        out = dcs.convert_date(np.nan, 'numpy', old_form='matplotlib')
        self.assertTrue(np.isnat(out))
        out = dcs.convert_date(np.nan, 'matplotlib', old_form='matplotlib')
        self.assertTrue(np.isnan(out))
        out = dcs.convert_date(np.nan, 'sec', self.date_zero, old_form='matplotlib')
        self.assertTrue(np.isnan(out))

    def test_no_date_zero_error(self):
        with self.assertRaises(AssertionError):
            dcs.convert_date(self.seconds, 'datetime')

    def test_numpy_form(self):
        out = dcs.convert_date(self.seconds, 'numpy', self.date_zero, numpy_form='datetime64[ms]')
        self.assertEqual(dcs.get_np_time_units(out), 'ms')

    def test_numpy_vectors(self):
        dates = np.array([self.numpy, self.nat], dtype='datetime64[ns]')
        out = dcs.convert_date(dates, 'sec', self.date_zero, old_form='numpy', numpy_form='datetime64[ms]')
        np.testing.assert_array_equal(out, np.array([self.seconds, np.nan]))
        out = dcs.convert_date(dates, 'matplotlib', self.date_zero, old_form='numpy', numpy_form='datetime64[ms]')
        np.testing.assert_array_equal(out, np.array([self.matplotlib, np.nan]))

    def test_seconds_vectors(self):
        dates = np.array([self.seconds, -np.inf, np.inf, np.nan], dtype=float)
        out = dcs.convert_date(dates, 'matplotlib', self.date_zero, old_form='sec')
        np.testing.assert_array_equal(out, np.array([self.matplotlib, -np.inf, np.inf, np.nan]))
        out = dcs.convert_date(dates, 'numpy', self.date_zero, old_form='sec')
        np.testing.assert_array_equal(out, np.array([self.numpy, self.nat, self.nat, self.nat], dtype='datetime64[ns]'))

    def test_mpl_vectors(self):
        dates = np.array([self.matplotlib, -np.inf, np.inf, np.nan], dtype=float)
        out = dcs.convert_date(dates, 'sec', self.date_zero, old_form='matplotlib')
        np.testing.assert_array_almost_equal(out, np.array([self.seconds, -np.inf, np.inf, np.nan]))
        out = dcs.convert_date(dates, 'numpy', self.date_zero, old_form='matplotlib')
        np.testing.assert_array_equal(out, np.array([self.numpy, self.nat, self.nat, self.nat], dtype='datetime64[ns]'))

#%% convert_time_units
class Test_convert_time_units(unittest.TestCase):
    r"""Tests the convert_time_units function with the following cases:
        Conversions
        Bad values
    """
    def test_sec(self):
        out = dcs.convert_time_units(3600, 'sec', 'sec')
        self.assertEqual(out, 3600.)
        out = dcs.convert_time_units(3600, 'sec', 'min')
        self.assertEqual(out, 60.)
        out = dcs.convert_time_units(3600, 'sec', 'hr')
        self.assertEqual(out, 1.)
        out = dcs.convert_time_units(3600, 'sec', 'day')
        self.assertEqual(out, 1/24)

    def test_min(self):
        out = dcs.convert_time_units(100, 'min', 'sec')
        self.assertEqual(out, 6000.)
        out = dcs.convert_time_units(100, 'min', 'min')
        self.assertEqual(out, 100.)
        out = dcs.convert_time_units(100, 'min', 'hr')
        self.assertEqual(out, 5/3)
        out = dcs.convert_time_units(100, 'min', 'day')
        self.assertEqual(out, 5/3/24)

    def test_hr(self):
        out = dcs.convert_time_units(2.5, 'hr', 'sec')
        self.assertEqual(out, 2.5*3600)
        out = dcs.convert_time_units(2.5, 'hr', 'min')
        self.assertEqual(out, 2.5*60)
        out = dcs.convert_time_units(2.5, 'hr', 'hr')
        self.assertEqual(out, 2.5)
        out = dcs.convert_time_units(2.5, 'hr', 'day')
        self.assertAlmostEqual(out, 2.5/24, 12)

    def test_day(self):
        out = dcs.convert_time_units(1.5, 'day', 'sec')
        self.assertEqual(out, 1.5*86400)
        out = dcs.convert_time_units(1.5, 'day', 'min')
        self.assertEqual(out, 2160.)
        out = dcs.convert_time_units(1.5, 'day', 'hr')
        self.assertEqual(out, 36)
        out = dcs.convert_time_units(1.5, 'day', 'day')
        self.assertEqual(out, 1.5)

    def test_bad(self):
        with self.assertRaises(ValueError):
            dcs.convert_time_units(1, 'sec', 'bad')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
