r"""
Test file for the `orbit_conv` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.anomaly_eccentric_2_true
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_anomaly_eccentric_2_true(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_eccentric_2_true function with the following cases:
        Nominal
        Vectorized (x3)
    """
    def setUp(self) -> None:
        self.E1 = 0.
        self.E2 = np.pi / 4
        self.e1 = 0.
        self.e2 = 0.5
        self.exp1 = 0.
        self.exp2 = 1.2446686345053115  # TODO: come up with independently

    def test_nominal(self) -> None:
        nu = space.anomaly_eccentric_2_true(self.E2, self.e2)
        assert isinstance(nu, float)
        self.assertEqual(nu, self.exp2)

    def test_vector1(self) -> None:
        E = np.array([self.E1, self.E2])
        e = self.e2
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2, ))
        self.assertEqual(nu[1], self.exp2)

    def test_vector2(self) -> None:
        E = self.E2
        e = np.array([self.e1, self.e2])
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2, ))
        self.assertEqual(nu[1], self.exp2)

    def test_vector3(self) -> None:
        E = np.array([self.E1, self.E2])
        e = np.array([self.e1, self.e2])
        nu = space.anomaly_eccentric_2_true(E, e)
        assert isinstance(nu, np.ndarray)
        self.assertEqual(nu.shape, (2, ))
        self.assertEqual(nu[0], self.exp1)
        self.assertEqual(nu[1], self.exp2)

#%% aerospace.anomaly_mean_2_eccentric
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_anomaly_mean_2_eccentric(unittest.TestCase):
    r"""
    Tests the aerospace.anomaly_mean_2_eccentric function with the following cases:
        Nominal
        Vectorized (x3)
    """
    def setUp(self) -> None:
        self.M1 = 0.
        self.M2 = np.pi / 4
        self.e1 = 0.
        self.e2 = 0.5
        self.exp1 = 0.
        self.exp2 = 1.2617030552531017  # TODO: come up with independently

    def test_nominal(self) -> None:
        E = space.anomaly_mean_2_eccentric(self.M2, self.e2)
        assert isinstance(E, float)
        self.assertEqual(E, self.exp2)

    def test_vector1(self) -> None:
        M = np.array([self.M1, self.M2])
        e = self.e2
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2, ))
        self.assertEqual(E[1], self.exp2)

    def test_vector2(self) -> None:
        M = self.M2
        e = np.array([self.e1, self.e2])
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2, ))
        self.assertEqual(E[1], self.exp2)

    def test_vector3(self) -> None:
        M = np.array([self.M1, self.M2])
        e = np.array([self.e1, self.e2])
        E = space.anomaly_mean_2_eccentric(M, e)
        assert isinstance(E, np.ndarray)
        self.assertEqual(E.shape, (2, ))
        self.assertEqual(E[0], self.exp1)
        self.assertEqual(E[1], self.exp2)

#%% aerospace.mean_motion_2_semimajor
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_mean_motion_2_semimajor(unittest.TestCase):
    r"""
    Tests the aerospace.mean_motion_2_semimajor function with the following cases:
        Nominal
        Vectorized (x3)
    """
    def setUp(self) -> None:
        self.n1 = 1.
        self.n2 = 7.2922e-5
        self.mu1 = 1.
        self.mu2 = 3.986e14
        self.exp1 = 1.
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
        self.assertEqual(a.shape, (2, ))
        self.assertEqual(a[1], self.exp2)

    def test_vector2(self) -> None:
        n = self.n2
        mu = np.array([self.mu1, self.mu2])
        a = space.mean_motion_2_semimajor(n, mu)
        assert isinstance(a, np.ndarray)
        self.assertEqual(a.shape, (2, ))
        self.assertEqual(a[1], self.exp2)

    def test_vector3(self) -> None:
        n = np.array([self.n1, self.n2])
        mu = np.array([self.mu1, self.mu2])
        a = space.mean_motion_2_semimajor(n, mu)
        assert isinstance(a, np.ndarray)
        self.assertEqual(a.shape, (2, ))
        self.assertEqual(a[0], self.exp1)
        self.assertEqual(a[1], self.exp2)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
