r"""
Test file for the `classes` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.

"""

# %% Imports
import datetime
from typing import Final
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np


# %% aerospace constants
class Test_aerospace_gps_constants(unittest.TestCase):
    r"""
    Tests the aerospace GPS constants with the following cases:
        Nominal
        No numpy
    """

    def test_nominal(self) -> None:
        self.assertIsInstance(space.GPS_DATE_ZERO, datetime.datetime)
        self.assertIsInstance(space.ONE_DAY, int)
        self.assertIsInstance(space.DAYS_PER_WEEK, int)
        self.assertIsInstance(space.WEEK_ROLLOVER, int)

    def test_numpy(self) -> None:
        if HAVE_NUMPY:
            self.assertIsInstance(space.NP_GPS_DATE_ZERO, np.datetime64)
        else:
            self.assertIsNone(space.NP_GPS_DATE_ZERO)


# %% aerospace.bsl
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_bsl(unittest.TestCase):
    r"""
    Tests the aerospace.bsl function with the following cases:
        Nominal
        Multishift
        In-place
    """

    def setUp(self) -> None:
        self.bits = np.array([0, 0, 1, 1, 1])

    def test_nominal(self) -> None:
        out = space.bsl(self.bits)
        self.assertIsNot(self.bits, out)
        np.testing.assert_array_equal(out, np.array([0, 1, 1, 1, 0]))

    def test_inplace(self) -> None:
        out = space.bsl(self.bits, inplace=True)
        self.assertIs(self.bits, out)
        np.testing.assert_array_equal(out, np.array([0, 1, 1, 1, 0]))

    def test_multiple(self) -> None:
        out = space.bsl(self.bits, 3)
        self.assertIsNot(self.bits, out)
        np.testing.assert_array_equal(out, np.array([1, 1, 0, 0, 1]))


# %% aerospace.bsr
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_bsr(unittest.TestCase):
    r"""
    Tests the aerospace.bsr function with the following cases:
        Nominal
        Multishift
        In-place
    """

    def setUp(self) -> None:
        self.bits = np.array([0, 0, 1, 1, 1])

    def test_nominal(self) -> None:
        out = space.bsr(self.bits)
        self.assertIsNot(self.bits, out)
        np.testing.assert_array_equal(out, np.array([1, 0, 0, 1, 1]))

    def test_inplace(self) -> None:
        out = space.bsr(self.bits, inplace=True)
        self.assertIs(self.bits, out)
        np.testing.assert_array_equal(out, np.array([1, 0, 0, 1, 1]))

    def test_multiple(self) -> None:
        out = space.bsr(self.bits, 3)
        self.assertIsNot(self.bits, out)
        np.testing.assert_array_equal(out, np.array([1, 1, 1, 0, 0]))


# %% aerospace.prn_01_to_m11
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_prn_01_to_m11(unittest.TestCase):
    r"""
    Tests the aerospace.prn_01_to_m11 function with the following cases:
        Nominal
        In-place
    """

    def setUp(self) -> None:
        self.bits = np.array([ 1,  1,  1, 0, 0,  1,  1])  # fmt: skip
        self.exp  = np.array([-1, -1, -1, 1, 1, -1, -1])  # fmt: skip

    def test_nominal(self) -> None:
        out = space.prn_01_to_m11(self.bits)
        self.assertIsNot(self.bits, out)
        np.testing.assert_array_equal(out, self.exp)

    def test_inplace(self) -> None:
        out = space.prn_01_to_m11(self.bits, inplace=True)
        self.assertIs(self.bits, out)
        np.testing.assert_array_equal(out, self.exp)


# %% aerospace.get_prn_bits
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_get_prn_bits(unittest.TestCase):
    r"""
    Tests the aerospace.get_prn_bits function with the following cases:
        Spot check
        No errors
        Bad satellite number
    """

    def test_spot_check(self) -> None:
        (b1, b2) = space.get_prn_bits(19)
        self.assertEqual(b1, 3)
        self.assertEqual(b2, 6)

    def test_no_errors(self) -> None:
        valid_bits = frozenset(range(1, 11))
        for i in range(1, 38):
            (b1, b2) = space.get_prn_bits(i)
            self.assertIn(b1, valid_bits)
            self.assertIn(b2, valid_bits)

    def test_bad_sat(self) -> None:
        with self.assertRaises(ValueError) as err:
            space.get_prn_bits(38)
        self.assertEqual(str(err.exception), 'Unexpected satellite number: "38"')


# %% aerospace.correlate_prn
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_correlate_prn(unittest.TestCase):
    r"""
    Tests the aerospace.correlate_prn function with the following cases:
        All PRNs
        -1 to 1
    """

    def test_nominal(self) -> None:
        shift = np.arange(1023)
        form: Final = "zero-one"
        for i in range(1, 38):
            prn = space.generate_prn(i)
            cor = space.correlate_prn(prn, prn, shift, form)
            self.assertEqual(cor[0], 1)
            np.testing.assert_array_less(np.max(np.abs(cor[1:])), 0.1)

    def test_alt_form(self) -> None:
        prn = space.prn_01_to_m11(space.generate_prn(1))
        shift = np.arange(1023)
        form: Final = "one-one"
        cor = space.correlate_prn(prn, prn, shift, form)
        self.assertEqual(cor[0], 1)
        np.testing.assert_array_less(np.max(np.abs(cor[1:])), 0.1)


# %% aerospace.generate_prn
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_generate_prn(unittest.TestCase):
    r"""
    Tests the aerospace.generate_prn function with the following cases:
        All PRNs
    """

    def test_nominal(self) -> None:
        for i in range(1, 38):
            prn = space.generate_prn(i)
            self.assertTrue(np.all((prn == 0) | (prn == 1)))
            # TODO: need external validation source


# %% aerospace.gps_to_datetime
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_gps_to_datetime(unittest.TestCase):
    r"""
    Tests the aerospace.gps_to_datetime function with the following cases:
        Nominal
        Wrapped
        Alternate Form
    """

    def test_nominal(self) -> None:
        week = np.array([1782, 1783])
        time = np.array([425916, 4132])
        date_gps = space.gps_to_datetime(week, time)
        exp = [datetime.datetime(2014, 3, 6, 22, 18, 36), datetime.datetime(2014, 3, 9, 1, 8, 52)]
        self.assertEqual(date_gps, exp)

    def test_wrapped(self) -> None:
        week = 758  # 2806 when unrolled in 2021
        time = 425915.34
        date_gps = space.gps_to_datetime(week, time)
        exp = datetime.datetime(2033, 10, 20, 22, 18, 35, 340000)
        self.assertEqual(date_gps, exp)

    def test_alt_form(self) -> None:
        week = np.array([1782, 1783])
        time = np.array([425916.0, 4132.56])
        date_gps = space.gps_to_datetime(week, time, form="numpy")
        exp = np.array([np.datetime64("2014-03-06T22:18:36", "ns"), np.datetime64("2014-03-09T01:08:52.560000", "ns")])
        np.testing.assert_array_equal(date_gps, exp)

    def test_unknown_form(self) -> None:
        with self.assertRaises(ValueError):
            space.gps_to_datetime(1, 1, form="bad_form")  # type: ignore[call-overload]


# %% aerospace.get_gps_to_utc_offset
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_get_gps_to_utc_offset(unittest.TestCase):
    r"""
    Tests the aerospace.get_gps_to_utc_offset function with the following cases:
        Recent
        Bounds
        Close changes
    """

    def test_recent(self) -> None:
        gps_days = (datetime.datetime.now() - datetime.datetime(1980, 1, 6)).days
        gps_to_utc_offset = space.get_gps_to_utc_offset(gps_days)
        self.assertEqual(gps_to_utc_offset, -18)

    def test_bounds(self) -> None:
        for date, offset in [
            (datetime.datetime(1980, 3, 1),   0), (datetime.datetime(1980, 9, 1),   0),
            (datetime.datetime(1981, 3, 1),   0), (datetime.datetime(1981, 9, 1),  -1),
            (datetime.datetime(1982, 3, 1),  -1), (datetime.datetime(1982, 9, 1),  -2),
            (datetime.datetime(1983, 3, 1),  -2), (datetime.datetime(1983, 9, 1),  -3),
            (datetime.datetime(1984, 3, 1),  -3), (datetime.datetime(1984, 9, 1),  -3),
            (datetime.datetime(1985, 3, 1),  -3), (datetime.datetime(1985, 9, 1),  -4),
            (datetime.datetime(1986, 3, 1),  -4), (datetime.datetime(1986, 9, 1),  -4),
            (datetime.datetime(1987, 3, 1),  -4), (datetime.datetime(1987, 9, 1),  -4),
            (datetime.datetime(1988, 3, 1),  -5), (datetime.datetime(1988, 9, 1),  -5),
            (datetime.datetime(1989, 3, 1),  -5), (datetime.datetime(1989, 9, 1),  -5),
            (datetime.datetime(1990, 3, 1),  -6), (datetime.datetime(1990, 9, 1),  -6),
            (datetime.datetime(1991, 3, 1),  -7), (datetime.datetime(1991, 9, 1),  -7),
            (datetime.datetime(1992, 3, 1),  -7), (datetime.datetime(1992, 9, 1),  -8),
            (datetime.datetime(1993, 3, 1),  -8), (datetime.datetime(1993, 9, 1),  -9),
            (datetime.datetime(1994, 3, 1),  -9), (datetime.datetime(1994, 9, 1), -10),
            (datetime.datetime(1995, 3, 1), -10), (datetime.datetime(1995, 9, 1), -10),
            (datetime.datetime(1996, 3, 1), -11), (datetime.datetime(1996, 9, 1), -11),
            (datetime.datetime(1997, 3, 1), -11), (datetime.datetime(1997, 9, 1), -12),
            (datetime.datetime(1998, 3, 1), -12), (datetime.datetime(1998, 9, 1), -12),
            (datetime.datetime(1999, 3, 1), -13), (datetime.datetime(1999, 9, 1), -13),
            (datetime.datetime(2000, 3, 1), -13), (datetime.datetime(2000, 9, 1), -13),
            (datetime.datetime(2001, 3, 1), -13), (datetime.datetime(2001, 9, 1), -13),
            (datetime.datetime(2002, 3, 1), -13), (datetime.datetime(2002, 9, 1), -13),
            (datetime.datetime(2003, 3, 1), -13), (datetime.datetime(2003, 9, 1), -13),
            (datetime.datetime(2004, 3, 1), -13), (datetime.datetime(2004, 9, 1), -13),
            (datetime.datetime(2005, 3, 1), -13), (datetime.datetime(2005, 9, 1), -13),
            (datetime.datetime(2006, 3, 1), -14), (datetime.datetime(2006, 9, 1), -14),
            (datetime.datetime(2007, 3, 1), -14), (datetime.datetime(2007, 9, 1), -14),
            (datetime.datetime(2008, 3, 1), -14), (datetime.datetime(2008, 9, 1), -14),
            (datetime.datetime(2009, 3, 1), -15), (datetime.datetime(2009, 9, 1), -15),
            (datetime.datetime(2010, 3, 1), -15), (datetime.datetime(2010, 9, 1), -15),
            (datetime.datetime(2011, 3, 1), -15), (datetime.datetime(2011, 9, 1), -15),
            (datetime.datetime(2012, 3, 1), -15), (datetime.datetime(2012, 9, 1), -16),
            (datetime.datetime(2013, 3, 1), -16), (datetime.datetime(2013, 9, 1), -16),
            (datetime.datetime(2014, 3, 1), -16), (datetime.datetime(2014, 9, 1), -16),
            (datetime.datetime(2015, 3, 1), -16), (datetime.datetime(2015, 9, 1), -17),
            (datetime.datetime(2016, 3, 1), -17), (datetime.datetime(2016, 9, 1), -17),
            (datetime.datetime(2017, 3, 1), -18), (datetime.datetime(2017, 9, 1), -18),
            (datetime.datetime(2018, 3, 1), -18), (datetime.datetime(2018, 9, 1), -18),
        ]:  # fmt: skip
            gps_days = (date - datetime.datetime(1980, 1, 6)).days
            gps_to_utc_offset = space.get_gps_to_utc_offset(gps_days)
            self.assertEqual(gps_to_utc_offset, offset, f"For date {date}")

    def test_close_changes(self) -> None:
        gps_days_array = np.array([13509.999999, 13510.0, 13510.0 + 15 / 86400, 13510.0 + 20 / 86400])
        gps_to_utc_offset = space.get_gps_to_utc_offset(gps_days_array)
        exp = np.array([-17, -17, -17, -18])
        np.testing.assert_array_equal(gps_to_utc_offset, exp)


# %% aerospace.gps_to_utc_datetime
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_gps_to_utc_datetime(unittest.TestCase):
    r"""
    Tests the aerospace.gps_to_utc_datetime function with the following cases:
        Nominal
        Wrapped
        Alternate Form
        Manual Leap Seconds
    """

    def test_nominal(self) -> None:
        week = np.array([1782, 1783])
        time = np.array([425916, 4132])
        date_utc = space.gps_to_utc_datetime(week, time)
        exp = [datetime.datetime(2014, 3, 6, 22, 18, 20), datetime.datetime(2014, 3, 9, 1, 8, 36)]
        self.assertEqual(date_utc, exp)

    def test_wrapped(self) -> None:
        week = 758  # 2806 when unrolled in 2021
        time = 425915.34
        date_utc = space.gps_to_utc_datetime(week, time)
        exp = datetime.datetime(2033, 10, 20, 22, 18, 17, 340000)  # Note: will change with leap seconds
        self.assertEqual(date_utc, exp)

    def test_alt_form(self) -> None:
        week = np.array([1782, 1783])
        time = np.array([425916, 4132])
        date_utc = space.gps_to_utc_datetime(week, time, form="numpy")  # type: ignore[call-overload]
        exp = np.array([np.datetime64("2014-03-06T22:18:20", "ns"), np.datetime64("2014-03-09T01:08:36", "ns")])
        np.testing.assert_array_equal(date_utc, exp)

    def test_manual_leap_seconds(self) -> None:
        week = 2806
        time = 425915
        date_utc = space.gps_to_utc_datetime(week, time, gps_to_utc_offset=-200)
        exp = datetime.datetime(2033, 10, 20, 22, 15, 15)
        self.assertEqual(date_utc, exp)

    def test_unknown_form(self) -> None:
        with self.assertRaises(ValueError):
            space.gps_to_utc_datetime(1, 1, gps_to_utc_offset=0, form="bad_form")  # type: ignore[call-overload]


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
