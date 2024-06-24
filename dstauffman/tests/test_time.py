r"""
Test file for the `time` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

# %% Imports
import datetime
import unittest
import warnings

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

    inf = np.inf
    nan = np.nan
    isnan = np.isnan
else:
    from math import inf, isnan, nan  # type: ignore[assignment]
if dcs.HAVE_MPL:
    import matplotlib.dates as dates


# %% get_np_time_units
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_get_np_time_units(unittest.TestCase):
    r"""
    Tests the get_np_time_units function with the following cases:
        Default
        nanosecond
    """

    def test_nominal(self) -> None:
        units = dcs.get_np_time_units(np.datetime64(datetime.datetime.now()))
        self.assertEqual(units, "us")

    def test_nanoseconds(self) -> None:
        units = dcs.get_np_time_units(np.datetime64(datetime.datetime(2020, 1, 1), "ns"))
        self.assertEqual(units, "ns")


# %% get_ymd_from_np
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_get_ymd_from_np(unittest.TestCase):
    r"""
    Tests the get_ymd_from_np function with the following cases:
        Single
        Vector
    """

    def setUp(self) -> None:
        self.date1 = np.datetime64("2022-07-15T12:34:56")
        self.date2 = np.datetime64("1905-03-30T23:15:00")
        self.date = np.array([self.date1, self.date2])
        self.exp1 = (2022, 7, 15)
        self.exp2 = (1905, 3, 30)

    def test_nominal(self) -> None:
        ymd = dcs.get_ymd_from_np(self.date1)
        self.assertEqual(ymd, self.exp1)
        ymd = dcs.get_ymd_from_np(self.date2)
        self.assertEqual(ymd, self.exp2)

    def test_vector(self) -> None:
        (y, m, d) = dcs.get_ymd_from_np(self.date)
        np.testing.assert_array_equal(y, np.array([self.exp1[0], self.exp2[0]]))
        np.testing.assert_array_equal(m, np.array([self.exp1[1], self.exp2[1]]))
        np.testing.assert_array_equal(d, np.array([self.exp1[2], self.exp2[2]]))


# %% round_datetime
class Test_round_datetime(unittest.TestCase):
    r"""
    Tests the round_datetime function with the following cases:
        normal use (round to one minute)
        extended use (round to a different specified time)
        get current time
    """

    def test_normal_use(self) -> None:
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 10))
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 4, 0))

    def test_extended_use(self) -> None:
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 10), round_to_sec=300)
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 5, 0))

    def test_current_time(self) -> None:
        dcs.round_datetime()
        self.assertTrue(True)

    def test_flooring(self) -> None:
        rounded_time = dcs.round_datetime(datetime.datetime(2015, 3, 13, 8, 4, 45), round_to_sec=60, floor=True)
        self.assertEqual(rounded_time, datetime.datetime(2015, 3, 13, 8, 4, 0))


# %% round_np_datetime
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_round_np_datetime(unittest.TestCase):
    r"""
    Tests the round_np_datetime function with the following cases:
        Nominal
        Flooring
    """

    def setUp(self) -> None:
        # fmt: off
        date_zero       = np.datetime64(datetime.date(2020, 1, 1))
        dt_sec          = np.array([0, 0.2, 0.35, 0.45, 0.59, 0.61])
        self.date_in    = date_zero + np.round(1000 * dt_sec).astype("timedelta64[ms]")
        self.time_delta = np.timedelta64(200, "ms")
        self.expected   = date_zero + np.array([0, 200, 400, 400, 600, 600]).astype("timedelta64[ms]")
        # fmt: on

    def test_nominal(self) -> None:
        date_out = dcs.round_np_datetime(self.date_in, self.time_delta)
        np.testing.assert_array_equal(date_out, self.expected)


# %% round_num_datetime
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_round_num_datetime(unittest.TestCase):
    r"""
    Tests the round_num_datetime function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.date_exact = np.arange(0, 10.1, 0.1)
        self.date_in = self.date_exact + 0.001 * np.random.rand(101)
        self.time_delta = 0.1

    def test_nominal(self) -> None:
        date_out = dcs.round_num_datetime(self.date_in, self.time_delta)
        np.testing.assert_array_almost_equal(date_out, self.date_exact, 12)

    def test_small(self) -> None:
        with self.assertWarns(Warning):
            dcs.round_num_datetime(1e6 * self.date_in, 1e-12)

    def test_floor(self) -> None:
        date_out = dcs.round_num_datetime(np.array([0.0, 1.1, 1.9, 3.05, 4.9]), 1.0, floor=True)
        np.testing.assert_array_almost_equal(date_out, np.array([0.0, 1.0, 1.0, 3.0, 4.0]), 12)


# %% round_time
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_round_time(unittest.TestCase):
    r"""
    Tests the round_time function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.date_zero = np.datetime64(datetime.date(2020, 1, 1)).astype(dcs.NP_DATETIME_FORM)
        self.x_sec = np.array([0, 0.2, 0.35, 0.45, 0.59, 0.61])
        self.x_np = self.date_zero + np.round(dcs.NP_INT64_PER_SEC * self.x_sec).astype(dcs.NP_TIMEDELTA_FORM)
        self.t_round = np.timedelta64(200, "ms").astype(dcs.NP_TIMEDELTA_FORM)

    def test_seconds(self) -> None:
        date_out = dcs.round_time(self.x_sec, self.t_round)
        expected = np.array([0.0, 0.2, 0.4, 0.4, 0.6, 0.6])
        np.testing.assert_array_almost_equal(date_out, expected, 14)  # type: ignore[arg-type]

    def test_numpy(self) -> None:
        date_out = dcs.round_time(self.x_np, self.t_round)
        expected = self.date_zero + np.array([0, 200, 400, 400, 600, 600]).astype("timedelta64[ms]").astype(
            dcs.NP_TIMEDELTA_FORM
        )
        np.testing.assert_array_equal(date_out, expected)


# %% convert_date
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

    def setUp(self) -> None:
        self.seconds = 3725.5
        self.date_zero = datetime.datetime(2020, 6, 1, 0, 0, 0, tzinfo=datetime.UTC)
        self.datetime = datetime.datetime(2020, 6, 1, 1, 2, 5, 500000, tzinfo=datetime.UTC)
        self.date = datetime.date(2022, 1, 17)
        if dcs.HAVE_NUMPY:
            self.numpy = np.datetime64("2020-06-01 01:02:05.500000", "ns")
            self.nat = np.datetime64("nat")
            self.np2 = np.datetime64("2022-01-17 00:00:00", "ns")
        if dcs.HAVE_MPL:
            self.matplotlib = dates.date2num(self.datetime)
            self.mpl2 = dates.date2num(self.date)

    def test_secs(self) -> None:
        out = dcs.convert_date(self.seconds, "datetime", self.date_zero)
        self.assertEqual(out, self.datetime)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.seconds, "numpy", self.date_zero)
            self.assertEqual(out, self.numpy)
        if dcs.HAVE_MPL:
            out = dcs.convert_date(self.seconds, "matplotlib", self.date_zero)
            self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.seconds, "sec", self.date_zero)
        self.assertEqual(out, self.seconds)

    def test_datetimes(self) -> None:
        out = dcs.convert_date(self.datetime, "datetime", old_form="datetime")
        self.assertEqual(out, self.datetime)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.datetime, "numpy", old_form="datetime")
            self.assertEqual(out, self.numpy)
            out = dcs.convert_date(self.date, "numpy", old_form="datetime")
            self.assertEqual(out, self.np2)
        if dcs.HAVE_MPL:
            out = dcs.convert_date(self.datetime, "matplotlib", old_form="datetime")
            self.assertEqual(out, self.matplotlib)
            out = dcs.convert_date(self.date, "matplotlib", old_form="datetime")
            self.assertEqual(out, self.mpl2)
        out = dcs.convert_date(self.datetime, "sec", self.date_zero, old_form="datetime")
        self.assertEqual(out, self.seconds)

    def test_numpys(self) -> None:
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.numpy, "datetime", old_form="numpy")
            self.assertEqual(out, self.datetime)
            out = dcs.convert_date(self.numpy, "numpy", old_form="numpy")
            self.assertEqual(out, self.numpy)
        if dcs.HAVE_MPL:
            out = dcs.convert_date(self.numpy, "matplotlib", old_form="numpy")
            self.assertEqual(out, self.matplotlib)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.numpy, "sec", self.date_zero, old_form="numpy")
            self.assertEqual(out, self.seconds)

    @unittest.skipIf(not dcs.HAVE_MPL, "Skipping due to missing matplotlib dependency.")
    def test_matplotlibs(self) -> None:
        out = dcs.convert_date(self.matplotlib, "datetime", old_form="matplotlib")
        exp = self.datetime.replace(tzinfo=datetime.timezone.utc)
        self.assertEqual(out, exp)
        out = dcs.convert_date(self.matplotlib, "numpy", old_form="matplotlib")
        self.assertEqual(out, self.numpy)
        out = dcs.convert_date(self.matplotlib, "matplotlib", old_form="matplotlib")
        self.assertEqual(out, self.matplotlib)
        out = dcs.convert_date(self.matplotlib, "sec", self.date_zero, old_form="matplotlib")
        self.assertAlmostEqual(out, self.seconds, 6)  # type: ignore[arg-type, misc]

    def test_mpl_missing_error(self) -> None:
        if not dcs.HAVE_MPL:
            with self.assertRaises(RuntimeError):
                dcs.convert_date(1000.0, "datetime", old_form="matplotlib")

    def test_infs_and_nans(self) -> None:
        out = dcs.convert_date(inf, "datetime")
        self.assertIsNone(out)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(inf, "numpy")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(inf, "matplotlib")
            self.assertEqual(out, inf)
        out = dcs.convert_date(inf, "sec")
        self.assertEqual(out, inf)
        out = dcs.convert_date(-inf, "datetime")
        self.assertIsNone(out)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(-inf, "numpy")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(-inf, "matplotlib")
            self.assertEqual(out, -inf)
        out = dcs.convert_date(-inf, "sec")
        self.assertEqual(out, -inf)
        out = dcs.convert_date(nan, "datetime")
        self.assertIsNone(out)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(nan, "numpy")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(nan, "matplotlib")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]
        out = dcs.convert_date(nan, "sec")
        self.assertTrue(isnan(out))  # type: ignore[arg-type]

    def test_nats(self) -> None:
        out = dcs.convert_date(None, "datetime", old_form="datetime")
        self.assertIsNone(out)
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(None, "numpy", old_form="datetime")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(None, "matplotlib", old_form="datetime")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]
        out = dcs.convert_date(None, "sec", self.date_zero, old_form="datetime")
        self.assertTrue(isnan(out))  # type: ignore[arg-type]
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.nat, "datetime", old_form="numpy")
            self.assertIsNone(out)
            out = dcs.convert_date(self.nat, "numpy", old_form="numpy")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(self.nat, "matplotlib", old_form="numpy")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]
        if dcs.HAVE_NUMPY:
            out = dcs.convert_date(self.nat, "sec", self.date_zero, old_form="numpy")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:
            out = dcs.convert_date(inf, "datetime", old_form="matplotlib")
            self.assertIsNone(out)
            out = dcs.convert_date(inf, "numpy", old_form="matplotlib")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
            out = dcs.convert_date(inf, "matplotlib", old_form="matplotlib")
            self.assertEqual(out, inf)
            out = dcs.convert_date(inf, "sec", self.date_zero, old_form="matplotlib")
            self.assertEqual(out, inf)
            out = dcs.convert_date(nan, "datetime", old_form="matplotlib")
            self.assertIsNone(out)
            out = dcs.convert_date(nan, "numpy", old_form="matplotlib")
            self.assertTrue(np.isnat(out))  # type: ignore[arg-type]
            out = dcs.convert_date(nan, "matplotlib", old_form="matplotlib")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]
            out = dcs.convert_date(nan, "sec", self.date_zero, old_form="matplotlib")
            self.assertTrue(isnan(out))  # type: ignore[arg-type]

    def test_no_date_zero_error(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.convert_date(self.seconds, "datetime")

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_numpy_form(self) -> None:
        out = dcs.convert_date(self.seconds, "numpy", self.date_zero, numpy_form="datetime64[ms]")
        self.assertEqual(dcs.get_np_time_units(out), "ms")  # type: ignore[arg-type]

    @unittest.skipIf(not dcs.HAVE_MPL, "Skipping due to missing matplotlib dependency.")
    def test_numpy_vectors(self) -> None:
        dates = np.array([self.numpy, self.nat], dtype="datetime64[ns]")
        out = dcs.convert_date(dates, "sec", self.date_zero, old_form="numpy", numpy_form="datetime64[ms]")
        np.testing.assert_array_equal(out, np.array([self.seconds, nan]))  # type: ignore[arg-type]
        out = dcs.convert_date(dates, "matplotlib", self.date_zero, old_form="numpy", numpy_form="datetime64[ms]")
        np.testing.assert_array_equal(out, np.array([self.matplotlib, nan]))  # type: ignore[arg-type]

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_datetime_lists(self) -> None:
        # DCS: note that this avoids a bug in matplotlib by controlling the order of elements in this list
        dates = [self.date, self.datetime]
        out = dcs.convert_date(dates, "numpy", old_form="datetime")
        np.testing.assert_array_equal(out, np.array([self.np2, self.numpy]))  # type: ignore[arg-type]
        if dcs.HAVE_MPL:  # pragma: no branch
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=DeprecationWarning)  # numpy v1
                warnings.filterwarnings(action="ignore", category=UserWarning)  # numpy v2
                out = dcs.convert_date(dates, "matplotlib", old_form="datetime")
            np.testing.assert_array_equal(out, np.array([self.mpl2, self.matplotlib]))  # type: ignore[arg-type]
        out = dcs.convert_date([self.seconds, self.seconds], "numpy", self.date_zero, old_form="sec")
        np.testing.assert_array_equal(out, np.array([self.numpy, self.numpy]))  # type: ignore[arg-type]

    @unittest.skipIf(not dcs.HAVE_MPL, "Skipping due to missing matplotlib dependency.")
    def test_seconds_vectors(self) -> None:
        dates = np.array([self.seconds, -inf, inf, nan])
        out = dcs.convert_date(dates, "matplotlib", self.date_zero, old_form="sec")
        np.testing.assert_array_equal(out, np.array([self.matplotlib, -inf, inf, nan]))  # type: ignore[arg-type]
        out = dcs.convert_date(dates, "numpy", self.date_zero, old_form="sec")
        np.testing.assert_array_equal(out, np.array([self.numpy, self.nat, self.nat, self.nat], dtype="datetime64[ns]"))  # type: ignore[arg-type]

    @unittest.skipIf(not dcs.HAVE_MPL, "Skipping due to missing matplotlib dependency.")
    def test_mpl_vectors(self) -> None:
        dates = np.array([self.matplotlib, -inf, inf, nan])
        out = dcs.convert_date(dates, "sec", self.date_zero, old_form="matplotlib")
        np.testing.assert_array_almost_equal(out, np.array([self.seconds, -inf, inf, nan]))  # type: ignore[arg-type]
        out = dcs.convert_date(dates, "numpy", self.date_zero, old_form="matplotlib")
        np.testing.assert_array_equal(out, np.array([self.numpy, self.nat, self.nat, self.nat], dtype="datetime64[ns]"))  # type: ignore[arg-type]

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_numpy_floats_and_ints(self) -> None:
        times1 = np.arange(10.0)
        times2 = np.arange(10, dtype=int)
        exp = self.numpy + 10**9 * np.arange(10).astype(np.int64)
        out = dcs.convert_date(times1, "numpy", self.datetime)
        np.testing.assert_array_equal(out, exp)  # type: ignore[arg-type]
        out = dcs.convert_date(times2, "numpy", self.datetime)
        np.testing.assert_array_equal(out, exp)  # type: ignore[arg-type]

    def test_future_forms(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.convert_date(self.seconds / dcs.ONE_MINUTE, "min", self.date_zero, old_form="sec")


# %% convert_time_units
class Test_convert_time_units(unittest.TestCase):
    r"""
    Tests the convert_time_units function with the following cases:
        Conversions
        Bad values
    """

    def test_sec(self) -> None:
        out = dcs.convert_time_units(3600, "sec", "sec")
        self.assertEqual(out, 3600.0)
        out = dcs.convert_time_units(3600, "sec", "min")
        self.assertEqual(out, 60.0)
        out = dcs.convert_time_units(3600, "sec", "hr")
        self.assertEqual(out, 1.0)
        out = dcs.convert_time_units(3600, "sec", "day")
        self.assertEqual(out, 1 / 24)

    def test_min(self) -> None:
        out = dcs.convert_time_units(100, "min", "sec")
        self.assertEqual(out, 6000.0)
        out = dcs.convert_time_units(100, "min", "min")
        self.assertEqual(out, 100.0)
        out = dcs.convert_time_units(100, "min", "hr")
        self.assertEqual(out, 5 / 3)
        out = dcs.convert_time_units(100, "min", "day")
        self.assertEqual(out, 5 / 3 / 24)

    def test_hr(self) -> None:
        out = dcs.convert_time_units(2.5, "hr", "sec")
        self.assertEqual(out, 2.5 * 3600)
        out = dcs.convert_time_units(2.5, "hr", "min")
        self.assertEqual(out, 2.5 * 60)
        out = dcs.convert_time_units(2.5, "hr", "hr")
        self.assertEqual(out, 2.5)
        out = dcs.convert_time_units(2.5, "hr", "day")
        self.assertAlmostEqual(out, 2.5 / 24, 12)

    def test_day(self) -> None:
        out = dcs.convert_time_units(1.5, "day", "sec")
        self.assertEqual(out, 1.5 * 86400)
        out = dcs.convert_time_units(1.5, "day", "min")
        self.assertEqual(out, 2160.0)
        out = dcs.convert_time_units(1.5, "day", "hr")
        self.assertEqual(out, 36)
        out = dcs.convert_time_units(1.5, "day", "day")
        self.assertEqual(out, 1.5)

    def test_bad(self) -> None:
        with self.assertRaises(ValueError):
            dcs.convert_time_units(1, "sec", "bad")


# %% convert_datetime_to_np
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_convert_datetime_to_np(unittest.TestCase):
    r"""
    Tests the convert_datetime_to_np function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        time = datetime.datetime(2020, 10, 1, 12, 34, 56, 789)
        out = dcs.convert_datetime_to_np(time)
        exp = np.datetime64("2020-10-01T12:34:56.000789000").astype(dcs.NP_DATETIME_FORM)
        self.assertEqual(out, exp)


# %% convert_duration_to_np
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_convert_duration_to_np(unittest.TestCase):
    r"""
    Tests the convert_duration_to_np function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        dt = datetime.timedelta(minutes=90)
        out = dcs.convert_duration_to_np(dt)
        exp = np.timedelta64(90, "m").astype(dcs.NP_TIMEDELTA_FORM)
        self.assertEqual(out, exp)


# %% convert_num_dt_to_np
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_convert_num_dt_to_np(unittest.TestCase):
    r"""
    Tests the convert_num_dt_to_np function with the following cases:
        Nominal
        Built-in numpy units
        Map conversions
    """

    def test_nominal(self) -> None:
        dt = 90 * 60
        out = dcs.convert_num_dt_to_np(dt)
        exp = np.timedelta64(5400, "s").astype(dcs.NP_TIMEDELTA_FORM)
        self.assertEqual(out, exp)

    def test_numpy_units(self) -> None:
        for key in ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"):
            dt = 105
            out = dcs.convert_num_dt_to_np(dt, units=key)
            exp = np.timedelta64(105, key).astype(dcs.NP_TIMEDELTA_FORM)
            self.assertEqual(out, exp)

    def test_conversions(self) -> None:
        map_ = {
            "year": "Y",
            "month": "M",
            "week": "W",
            "day": "D",
            "hour": "h",
            "hr": "h",
            "minute": "m",
            "min": "m",
            "second": "s",
            "sec": "s",
        }
        for key, value in map_.items():
            dt = 90
            out = dcs.convert_num_dt_to_np(dt, units=key)
            exp = np.timedelta64(90, value).astype(dcs.NP_TIMEDELTA_FORM)
            self.assertEqual(out, exp)


# %% get_delta_time_str
class Test_get_delta_time_str(unittest.TestCase):
    r"""
    Tests the get_delta_time_str function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.start_time = datetime.datetime.now()
        self.final_time = self.start_time + datetime.timedelta(seconds=5, microseconds=10000)
        self.exp_str1 = "00:00:05"
        self.exp_str2 = "00:00"
        self.format_ = "%H:%M"

    def test_nominal(self) -> None:
        self.assertEqual(dcs.get_delta_time_str(self.start_time, self.final_time), self.exp_str1)

    def test_format(self) -> None:
        self.assertEqual(dcs.get_delta_time_str(self.start_time, self.final_time, format_=self.format_), self.exp_str2)

    def test_duration(self) -> None:
        self.assertEqual(dcs.get_delta_time_str(datetime.timedelta(seconds=5402)), "01:30:02")


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
