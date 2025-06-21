r"""
Test file for the `orbit_conv` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

# %% Imports
import unittest
from unittest.mock import patch

from slog import LogLevel

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np


# %% aerospace.orbit_conv._any
class Test_aerospace_orbit_conv__any(unittest.TestCase):
    r"""
    Tests the aerospace.orbit_conv._any function with the following cases:
        Bool
        Int
        Float
        ndarray
        list of bools
    """

    def test_bool(self) -> None:
        self.assertTrue(space.orbit_conv._any(True))
        self.assertFalse(space.orbit_conv._any(False))

    def test_int(self) -> None:
        self.assertTrue(space.orbit_conv._any(2))
        self.assertFalse(space.orbit_conv._any(0))

    def test_float(self) -> None:
        self.assertTrue(space.orbit_conv._any(1.5))
        self.assertFalse(space.orbit_conv._any(0.0))

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_ndarray(self) -> None:
        self.assertTrue(space.orbit_conv._any(np.array([True, False, True], dtype=bool)))
        self.assertFalse(space.orbit_conv._any(np.array([False, False, False], dtype=bool)))

    def test_nubs(self) -> None:
        self.assertTrue(space.orbit_conv._any([True, False, True]))
        self.assertFalse(space.orbit_conv._any([False, False, False]))


# %% aerospace.anomaly_eccentric_2_mean
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_anomaly_eccentric_2_mean(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_eccentric_2_mean function with the following cases:
        Nominal
        Vectorized (x3)
    """

    def setUp(self) -> None:
        self.E1 = 0.0
        self.E2 = np.pi / 4
        self.e1 = 0.0
        self.e2 = 0.5
        self.exp1 = 0.0
        self.exp2 = 0.4318447728041745  # TODO: come up with independently

    def test_nominal(self) -> None:
        M = space.anomaly_eccentric_2_mean(self.E2, self.e2)
        self.assertAlmostEqual(M, self.exp2, 14)  # type: ignore[misc]

    def test_vector1(self) -> None:
        E = np.array([self.E1, self.E2])
        e = self.e2
        M = space.anomaly_eccentric_2_mean(E, e)
        assert isinstance(M, np.ndarray)
        self.assertEqual(M.shape, (2,))
        self.assertAlmostEqual(M[1], self.exp2, 14)

    def test_vector2(self) -> None:
        E = self.E2
        e = np.array([self.e1, self.e2])
        M = space.anomaly_eccentric_2_mean(E, e)
        assert isinstance(M, np.ndarray)
        self.assertEqual(M.shape, (2,))
        self.assertAlmostEqual(M[1], self.exp2, 14)

    def test_vector3(self) -> None:
        E = np.array([self.E1, self.E2])
        e = np.array([self.e1, self.e2])
        M = space.anomaly_eccentric_2_mean(E, e)
        assert isinstance(M, np.ndarray)
        self.assertEqual(M.shape, (2,))
        self.assertEqual(M[0], self.exp1)
        self.assertAlmostEqual(M[1], self.exp2, 14)

    def test_hyperbolic(self) -> None:
        with self.assertRaises(ValueError):
            space.anomaly_eccentric_2_mean(self.E2, 1.1)

    def test_range_loop(self) -> None:
        with patch("dstauffman.aerospace.orbit_conv.logger") as mock_logger:
            M = space.anomaly_eccentric_2_mean(self.E2 + 2 * np.pi, self.e2)
        mock_logger.log.assert_called_with(LogLevel.L6, "The eccentric anomaly was outside the range of 0 to 2*pi")
        self.assertAlmostEqual(M, self.exp2, 14)  # type: ignore[misc]


# %% aerospace.anomaly_eccentric_2_true
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_anomaly_eccentric_2_true(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_eccentric_2_true function with the following cases:
        Nominal
        Vectorized (x3)
    """

    def setUp(self) -> None:
        self.E1 = 0.0
        self.E2 = np.pi / 4
        self.e1 = 0.0
        self.e2 = 0.5
        self.exp1 = 0.0
        self.exp2 = 1.2446686345053115  # TODO: come up with independently

    def test_nominal(self) -> None:
        nu = space.anomaly_eccentric_2_true(self.E2, self.e2)
        self.assertAlmostEqual(nu, self.exp2, 14)  # type: ignore[misc]

    def test_vector1(self) -> None:
        E = np.array([self.E1, self.E2])
        e = self.e2
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2,))
        self.assertAlmostEqual(nu[1], self.exp2, 14)

    def test_vector2(self) -> None:
        E = self.E2
        e = np.array([self.e1, self.e2])
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2,))
        self.assertAlmostEqual(nu[1], self.exp2, 14)

    def test_vector3(self) -> None:
        E = np.array([self.E1, self.E2])
        e = np.array([self.e1, self.e2])
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2,))
        self.assertEqual(nu[0], self.exp1)
        self.assertAlmostEqual(nu[1], self.exp2, 14)

    def test_hyperbolic(self) -> None:
        with self.assertRaises(ValueError):
            space.anomaly_eccentric_2_true(self.E2, 1.1)

    def test_range_loop(self) -> None:
        with patch("dstauffman.aerospace.orbit_conv.logger") as mock_logger:
            nu = space.anomaly_eccentric_2_true(self.E2 + 2 * np.pi, self.e2)
        mock_logger.log.assert_called_with(LogLevel.L6, "The eccentric anomaly was outside the range of 0 to 2*pi")
        self.assertAlmostEqual(nu, self.exp2, 14)  # type: ignore[misc]


# %% aerospace.anomaly_hyperbolic_2_mean
class Test_aerospace_anomaly_hyperbolic_2_mean(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_hyperbolic_2_mean function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.anomaly_hyperbolic_2_true
class Test_aerospace_anomaly_hyperbolic_2_true(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_hyperbolic_2_true function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.anomaly_mean_2_eccentric
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_anomaly_mean_2_eccentric(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_mean_2_eccentric function with the following cases:
        Nominal
        Vectorized (x3)
    """

    def setUp(self) -> None:
        self.M1 = 0.0
        self.M2 = np.pi / 4
        self.e1 = 0.0
        self.e2 = 0.5
        self.exp1 = 0.0
        self.exp2 = 1.2617030552531017  # TODO: come up with independently

    def test_nominal(self) -> None:
        E = space.anomaly_mean_2_eccentric(self.M2, self.e2)
        self.assertEqual(E, self.exp2)

    def test_vector1(self) -> None:
        M = np.array([self.M1, self.M2])
        e = self.e2
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2,))
        self.assertEqual(E[1], self.exp2)

    def test_vector2(self) -> None:
        M = self.M2
        e = np.array([self.e1, self.e2])
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2,))
        self.assertEqual(E[1], self.exp2)

    def test_vector3(self) -> None:
        M = np.array([self.M1, self.M2])
        e = np.array([self.e1, self.e2])
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2,))
        self.assertEqual(E[0], self.exp1)
        self.assertEqual(E[1], self.exp2)


# %% aerospace.anomaly_mean_2_true
class Test_aerospace_anomaly_mean_2_true(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_mean_2_true function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.anomaly_true_2_eccentric
class Test_aerospace_anomaly_true_2_eccentric(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_true_2_eccentric function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.anomaly_true_2_hyperbolic
class Test_aerospace_anomaly_true_2_hyperbolic(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_true_2_hyperbolic function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.anomaly_true_2_mean
class Test_aerospace_anomaly_true_2_mean(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_true_2_mean function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.mean_motion_2_semimajor
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_mean_motion_2_semimajor(unittest.TestCase):
    r"""
    Tests the aerospace.mean_motion_2_semimajor function with the following cases:
        Nominal
        Vectorized (x3)
    """

    def setUp(self) -> None:
        self.n1 = 1.0
        self.n2 = 7.2922e-5
        self.mu1 = 1.0
        self.mu2 = 3.986e14
        self.exp1 = 1.0
        self.exp2 = 4.216382970079457e7  # TODO: come up with independently

    def test_nominal(self) -> None:
        a = space.mean_motion_2_semimajor(self.n2, self.mu2)
        assert isinstance(a, float)
        self.assertEqual(a, self.exp2)

    def test_vector1(self) -> None:
        n = np.array([self.n1, self.n2])
        mu = self.mu2
        a = space.mean_motion_2_semimajor(n, mu)
        assert isinstance(a, np.ndarray)
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[1], self.exp2)

    def test_vector2(self) -> None:
        n = self.n2
        mu = np.array([self.mu1, self.mu2])
        a = space.mean_motion_2_semimajor(n, mu)
        assert isinstance(a, np.ndarray)
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[1], self.exp2)

    def test_vector3(self) -> None:
        n = np.array([self.n1, self.n2])
        mu = np.array([self.mu1, self.mu2])
        a = space.mean_motion_2_semimajor(n, mu)
        assert isinstance(a, np.ndarray)
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[0], self.exp1)
        self.assertEqual(a[1], self.exp2)


# %% aerospace.period_2_semimajor
class Test_aerospace_period_2_semimajor(unittest.TestCase):
    r"""
    Tests the aerospace.period_2_semimajor function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.semimajor_2_mean_motion
class Test_aerospace_semimajor_2_mean_motion(unittest.TestCase):
    r"""
    Tests the aerospace.semimajor_2_mean_motion function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.semimajor_2_period
class Test_aerospace_semimajor_2_period(unittest.TestCase):
    r"""
    Tests the aerospace.semimajor_2_period function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.sidereal_2_long
class Test_aerospace_sidereal_2_long(unittest.TestCase):
    r"""
    Tests the aerospace.sidereal_2_long function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.raan_2_mltan
class Test_aerospace_raan_2_mltan(unittest.TestCase):
    r"""
    Tests the aerospace.raan_2_mltan function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.jd_2_sidereal
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_jd_2_sidereal(unittest.TestCase):
    r"""
    Tests the aerospace.jd_2_sidereal function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        time_jd = 2460126.5  # UT1
        lst = space.jd_2_sidereal(time_jd)
        exp_lst = space.hms_2_r(np.array([18, 35, 10.4399]))  # from 2023 Astronomical Almanac, page B16
        self.assertAlmostEqual(lst, exp_lst)  # type: ignore[misc]

    def test_vector(self) -> None:
        time_jd = np.array([2460218.5, 2460264.5])  # UT1
        lst = space.jd_2_sidereal(time_jd)
        exp_lst = space.hms_2_r(np.array([[0, 3], [37, 39], [53.5338, 15.0808]]))
        np.testing.assert_array_almost_equal(lst, exp_lst)


# %% aerospace.quat_eci_2_ecf_approx
class Test_aerospace_quat_eci_2_ecf_approx(unittest.TestCase):
    r"""
    Tests the aerospace.quat_eci_2_ecf_approx function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.quat_eci_2_ecf
class Test_aerospace_quat_eci_2_ecf(unittest.TestCase):
    r"""
    Tests the aerospace.quat_eci_2_ecf function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
