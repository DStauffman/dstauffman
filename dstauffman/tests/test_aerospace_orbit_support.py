r"""
Test file for the `orbit_support` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in August 2021.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np
    from numpy import pi
else:
    from math import pi

#%% aerospace.d_2_r
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


#%% aerospace.r_2_d
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


#%% aerospace.norm
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


#%% aerospace.dot
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_dot(unittest.TestCase):
    r"""
    Tests the aerospace.dot function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        x = space.dot(np.array([1, 3, 5]), np.array([2, 4, 6]))
        self.assertEqual(x, 1 * 2 + 3 * 4 + 5 * 6)


#%% aerospace.cross
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


#%% aerospace.jd_to_numpy
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_jd_to_numpy(unittest.TestCase):
    r"""
    Tests the aerospace.jd_to_numpy function with the following cases:
        Nominal
        Vectorized
    """

    def test_nominal(self) -> None:
        jd = space.numpy_to_jd(np.datetime64("2000-01-01T00:00:00"))
        self.assertEqual(jd, 2451545.0)

    def test_vectorized(self) -> None:
        x = np.array([np.datetime64("2000-01-01T00:00:00"), np.datetime64("2000-01-01T12:00:00")])
        jd = space.numpy_to_jd(x)
        np.testing.assert_array_equal(jd, np.array([2451545.0, 2451545.5]))


#%% aerospace.jd_to_numpy
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_jd_to_numpy(unittest.TestCase):
    r"""
    Tests the aerospace.jd_to_numpy function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        date = space.jd_to_numpy(2451545.0)
        self.assertEqual(date, np.datetime64("2000-01-01T00:00:00"))

    def test_vectorized(self) -> None:
        jd = np.array([2451545.0, 2451545.5])
        date = space.jd_to_numpy(jd)
        exp = np.array([np.datetime64("2000-01-01T00:00:00"), np.datetime64("2000-01-01T12:00:00")])
        np.testing.assert_array_equal(date, exp)


#%% aerospace.d_2_dms
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


#%% aerospace.dms_2_d
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


#%% aerospace.hms_2_r
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


#%% aerospace.r_2_hms
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


#%% aerospace.aer_2_rdr

#%% aerospace.aer_2_sez

#%% aerospace.geo_loc_2_ijk

#%% aerospace.ijk_2_rdr

#%% aerospace.ijk_2_sez

#%% aerospace.long_2_sidereal

#%% aerospace.rdr_2_aer

#%% aerospace.rdr_2_ijk

#%% aerospace.sez_2_aer

#%% aerospace.sez_2_ijk

#%% aerospace.rv_aer_2_ijk

#%% aerospace.rv_aer_2_sez

#%% aerospace.rv_ijk_2_aer

#%% aerospace.rv_ijk_2_sez

#%% aerospace.rv_sez_2_aer

#%% aerospace.rv_sez_2_ijk

#%% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
