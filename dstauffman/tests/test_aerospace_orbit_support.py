r"""
Test file for the `orbit_support` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in August 2021.
"""

# %% Imports
import datetime
import unittest

from dstauffman import convert_datetime_to_np, HAVE_NUMPY, issorted
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np
    from numpy import pi
else:
    from math import pi


# %% aerospace.d_2_r
class Test_d_2_r(unittest.TestCase):
    r"""
    Tests the aerospace.d_2_r function with the following cases:
        Single vector
        Multi-vector
    """

    def test_nominal(self) -> None:
        self.assertEqual(space.d_2_r(180.0), pi)

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_vector(self) -> None:
        np.testing.assert_array_almost_equal(
            space.d_2_r(np.array([0.0, 45.0, 90.0, 180.0])), np.array([0.0, pi / 4, pi / 2, pi]), 14
        )


# %% aerospace.r_2_d
class Test_r_2_d(unittest.TestCase):
    r"""
    Tests the aerospace.r_2_d function with the following cases:
        Single vector
        Multi-vector
    """

    def test_nominal(self) -> None:
        self.assertEqual(space.r_2_d(pi), 180.0)

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_vector(self) -> None:
        np.testing.assert_array_almost_equal(
            space.r_2_d(np.array([0.0, pi / 4, pi / 2, pi])), np.array([0.0, 45.0, 90.0, 180.0]), 14
        )


# %% aerospace.norm
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_norm(unittest.TestCase):
    r"""
    Tests the aerospace.norm function with the following cases:
        Single vector
        Multi-vector
    """

    def test_nominal(self) -> None:
        mag = space.norm(np.array([3.0, 4.0]))
        self.assertEqual(mag, 5.0)

    def test_vectors(self) -> None:
        mag = space.norm(np.array([[3.0, 0.0], [4.0, 10.0]]))
        np.testing.assert_array_equal(mag, np.array([5.0, 10.0]))


# %% aerospace.dot
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_dot(unittest.TestCase):
    r"""
    Tests the aerospace.dot function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        x = space.dot(np.array([1, 3, 5]), np.array([2, 4, 6]))
        self.assertEqual(x, 1 * 2 + 3 * 4 + 5 * 6)


# %% aerospace.cross
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_cross(unittest.TestCase):
    r"""
    Tests the aerospace.cross function with the following cases:
        TBD
    """

    def test_nominal(self) -> None:
        x = space.cross(np.array([1, 0, 0]), np.array([0, 1, 0]))
        np.testing.assert_array_equal(x, np.array([0, 0, 1]))

    def test_vectors(self) -> None:
        v1 = np.array([[1, 0, 0], [1, 2, 3], [1, 1, 0]]).T
        v2 = np.array([[0, 1, 0], [1, 2, 3], [1, -1, 0]]).T
        exp = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -2]]).T
        x = space.cross(v1, v2)
        np.testing.assert_array_equal(x, exp)


# %% aerospace.jd_to_numpy
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_jd_to_numpy(unittest.TestCase):
    r"""
    Tests the aerospace.jd_to_numpy function with the following cases:
        Nominal
        Vectorized
    """

    def test_nominal(self) -> None:
        jd = space.numpy_to_jd(np.datetime64("2000-01-01T12:00:00"))
        self.assertEqual(jd, 2451545.0)

    def test_vectorized(self) -> None:
        x = np.array([np.datetime64("2000-01-01T00:00:00"), np.datetime64("2000-01-01T12:00:00")])
        jd = space.numpy_to_jd(x)
        np.testing.assert_array_equal(jd, np.array([2451544.5, 2451545.0]))


# %% aerospace.jd_to_numpy
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_jd_to_numpy(unittest.TestCase):
    r"""
    Tests the aerospace.jd_to_numpy function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        date = space.jd_to_numpy(2451545.0)
        self.assertEqual(date, np.datetime64("2000-01-01T12:00:00"))

    def test_vectorized(self) -> None:
        jd = np.array([2451544.5, 2451545.0])
        date = space.jd_to_numpy(jd)
        exp = np.array([np.datetime64("2000-01-01T00:00:00"), np.datetime64("2000-01-01T12:00:00")])
        np.testing.assert_array_equal(date, exp)


# %% aerospace.d_2_dms
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_d_2_dms(unittest.TestCase):
    r"""
    Tests the aerospace.d_2_dms function with the following cases:
        Nominal
        Vectorized
    """

    def test_nominal(self) -> None:
        dms = space.d_2_dms(38.4541666666666666)
        np.testing.assert_array_almost_equal(dms, np.array([38.0, 27.0, 15.0]), 11)

    def test_vector(self) -> None:
        dms = space.d_2_dms(np.array([0.0, 38.4541666666666666]))
        np.testing.assert_array_almost_equal(dms, np.array([[0.0, 0.0, 0], [38.0, 27.0, 15.0]]).T, 11)

    def test_bad_size(self) -> None:
        with self.assertRaises(ValueError):
            space.d_2_dms(np.zeros((3, 10)))


# %% aerospace.dms_2_d
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_dms_2_d(unittest.TestCase):
    r"""
    Tests the aerospace.dms_2_d function with the following cases:
        Nominal
        Vectorized
    """

    def test_nominal(self) -> None:
        dms = space.dms_2_d(np.array([38.0, 27.0, 15.0]))
        np.testing.assert_array_almost_equal(dms, 38.4541666666666666, 11)

    def test_vector(self) -> None:
        dms = space.dms_2_d(np.array([[0.0, 0.0, 0], [38.0, 27.0, 15.0]]).T)
        np.testing.assert_array_almost_equal(dms, np.array([0.0, 38.4541666666666666]), 11)

    def test_bad_size(self) -> None:
        with self.assertRaises(ValueError):
            space.dms_2_d(np.zeros(10))


# %% aerospace.hms_2_r
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_hms_2_r(unittest.TestCase):
    r"""
    Tests the aerospace.hms_2_r function with the following cases:
        Nominal
        Vectorized
    """

    def setUp(self) -> None:
        self.hms = np.array([12.0, 5.0, 15.0])
        self.exp = 2 * pi * (12 / 24 + 5 / 24 / 60 + 15 / 24 / 60 / 60)

    def test_nominal(self) -> None:
        x = space.hms_2_r(self.hms)
        self.assertAlmostEqual(x, self.exp, 14)

    def test_vector(self) -> None:
        x = space.hms_2_r(np.array([[0.0, 0.0, 0], self.hms, [8.0, 0.0, 0.0], [23, 59, 60]]).T)
        np.testing.assert_array_almost_equal(x, np.array([0.0, self.exp, 2 * pi / 3, 2 * pi]), 11)


# %% aerospace.r_2_hms
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_r_2_hms(unittest.TestCase):
    r"""
    Tests the aerospace.r_2_hms function with the following cases:
        Nominal
        Vectorized
    """

    def setUp(self) -> None:
        self.r = 2 * pi * (12 / 24 + 5 / 24 / 60 + 15 / 24 / 60 / 60)
        self.exp = np.array([12.0, 5.0, 15.0])

    def test_nominal(self) -> None:
        hms = space.r_2_hms(self.r)
        np.testing.assert_array_almost_equal(hms, self.exp, 11)

    def test_vector(self) -> None:
        r = np.array([0.0, self.r, 2 * pi / 3, 2 * pi])
        hms = space.r_2_hms(r)
        exp = np.array([[0.0, 0.0, 0], self.exp, [8.0, 0.0, 0.0], [24, 0, 0]]).T
        np.testing.assert_array_almost_equal(hms, exp, 11)


# %% aerospace.aer_2_rdr

# %% aerospace.aer_2_sez

# %% aerospace.geo_loc_2_ijk

# %% aerospace.ijk_2_rdr

# %% aerospace.ijk_2_sez

# %% aerospace.long_2_sidereal

# %% aerospace.rdr_2_aer

# %% aerospace.rdr_2_ijk

# %% aerospace.sez_2_aer

# %% aerospace.sez_2_ijk

# %% aerospace.rv_aer_2_ijk

# %% aerospace.rv_aer_2_sez

# %% aerospace.rv_ijk_2_aer

# %% aerospace.rv_ijk_2_sez

# %% aerospace.rv_sez_2_aer

# %% aerospace.rv_sez_2_ijk

# %% aerospace.get_sun_radec_approx


# %% aerospace.get_sun_radec
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_get_sun_radec(unittest.TestCase):
    r"""
    Tests the get_sun_radec function with the following cases:
        Nominal
        Return early
    """

    def setUp(self) -> None:
        date = datetime.datetime(2010, 6, 20, 15, 30, 45)
        np_date = convert_datetime_to_np(date)
        self.time_jd: float = space.numpy_to_jd(np_date)

    def test_nominal(self) -> None:
        (ra, dec) = space.get_sun_radec(self.time_jd)
        self.assertAlmostEqual(ra, 1.5557002786752125, 14)  # TODO: get independent source for this
        self.assertAlmostEqual(dec, 0.4090272497793529, 14)  # TODO: get independent source for this

    def test_return_early(self) -> None:
        (Ls, ob) = space.get_sun_radec(self.time_jd, return_early=True)
        self.assertAlmostEqual(Ls, 1.5569456630415757, 14)  # TODO: get independent source for this
        self.assertAlmostEqual(ob, 0.40906883260993276, 14)  # TODO: get independent source for this


# %% aerospace.get_sun_distance
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_get_sun_distance(unittest.TestCase):
    r"""
    Tests the get_sun_distance function with the following cases:
        Close & Far
        Vectorized
    """

    def setUp(self) -> None:
        date_jan = datetime.datetime(2023, 1, 5)
        date_jul = datetime.datetime(2023, 7, 7)
        np_date_jan = convert_datetime_to_np(date_jan)
        np_date_jul = convert_datetime_to_np(date_jul)
        self.time_jd_jan: float = space.numpy_to_jd(np_date_jan)
        self.time_jd_jul: float = space.numpy_to_jd(np_date_jul)
        self.dist_jan = 0.983_2959  # from Astronomical Almanac 2023, page C6
        self.dist_jul = 1.016_6805  # from Astronomical Almanac 2023, page C14
        self.limit = 4  # sig fig limit limits of current algorithm (0.0003 AU)

    def test_nominal(self) -> None:
        sun_dist_jan = space.get_sun_distance(self.time_jd_jan)
        self.assertAlmostEqual(sun_dist_jan, self.dist_jan, self.limit)
        sun_dist_jul = space.get_sun_distance(self.time_jd_jul)
        self.assertAlmostEqual(sun_dist_jul, self.dist_jul, self.limit)

    def test_vectorized(self) -> None:
        sun_dist = space.get_sun_distance(np.array([self.time_jd_jan, self.time_jd_jul]))
        exp = np.array([self.dist_jan, self.dist_jul])
        np.testing.assert_array_almost_equal(sun_dist, exp, self.limit)


# %% aerospace.beta_from_oe
pass  # TODO: write this


# %% aerospace.eclipse_fraction
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_eclipse_fraction(unittest.TestCase):
    r"""
    Tests the aerospace.eclipse_fraction with the following cases:
        Nominal
        Bad Values
    """

    def setUp(self) -> None:
        self.altitude = 16000.0
        self.beta = np.pi / 6
        self.exp = 0.4739855149323683  # TODO: get independently

    def test_nominal(self) -> None:
        fe = space.eclipse_fraction(self.altitude, self.beta)
        self.assertAlmostEqual(fe, self.exp, 14)

    def test_bad_values(self) -> None:
        altitude = np.array([self.altitude, np.nan, -1000000.0])
        beta = np.array([self.beta, np.nan, self.beta])
        exp = np.array([self.exp, np.nan, np.nan])
        fe = space.eclipse_fraction(altitude, beta)
        np.testing.assert_array_almost_equal(fe, exp, 14)


# %% aerospace.earth_radius_by_latitude
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_earth_radius_by_latitude(unittest.TestCase):
    r"""
    Tests the aerospace.earth_radius_by_latitude function withthe following cases:
        Equator
        Pole
        In-between
    """

    def test_equator(self) -> None:
        earth_rad = space.earth_radius_by_latitude(0.0)
        self.assertEqual(earth_rad, space.EARTH["a"])

    def test_poles(self) -> None:
        earth_rad = space.earth_radius_by_latitude(pi / 2)
        self.assertEqual(earth_rad, space.EARTH["b"])

    def test_between(self) -> None:
        latitude = np.arange(0.05, 1.55, 0.05)
        earth_rad = space.earth_radius_by_latitude(latitude)
        self.assertTrue(np.all(earth_rad < space.EARTH["a"]))
        self.assertTrue(np.all(earth_rad > space.EARTH["b"]))
        self.assertTrue(issorted(earth_rad, descend=True))


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
