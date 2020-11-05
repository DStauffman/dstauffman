r"""
Test file for the `kalman` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.estimation as estm

if HAVE_NUMPY:
    import numpy as np

#%% estimation.calculate_kalman_gain
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_calculate_kalman_gain(unittest.TestCase):
    r"""
    Tests the estimation.calculate_kalman_gain function with the following cases:
        Nominal
        With inverse
    """
    def setUp(self) -> None:
        self.P = 1e-3 * np.eye(5)
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
        self.R = 0.5 * np.eye(3)
        self.exp = 0.0019950134608610927 # TODO: come up with something that can be known better

    def test_nominal(self) -> None:
        K = estm.calculate_kalman_gain(self.P, self.H, self.R)
        self.assertAlmostEqual(K[0, 0], self.exp, 14)

    def test_inverse(self) -> None:
        K = estm.calculate_kalman_gain(self.P, self.H, self.R, use_inverse=True)
        self.assertAlmostEqual(K[0, 0], self.exp, 12)

#%% estimation.calculate_prediction
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_calculate_prediction(unittest.TestCase):
    r"""
    Tests the estimation.calculate_prediction function with the following cases:
        Nominal
        Constant offset
    """
    def setUp(self) -> None:
        self.H     = np.array([[1., 0.], [0., 1.], [0., 0.]])
        self.state = np.array([1e-3, 5e-3])
        self.const = np.array([0.5, 0.])
        self.exp   = np.array([0.001, 0.005, 0.])

    def test_nominal(self) -> None:
        u_pred = estm.calculate_prediction(self.H, self.state)
        np.testing.assert_almost_equal(u_pred, self.exp)

    def test_const(self) -> None:
        u_pred = estm.calculate_prediction(self.H, self.state, const=self.const)
        np.testing.assert_almost_equal(u_pred, self.exp + np.array([0.5, 0., 0.]))

#%% estimation.calculate_innovation
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_calculate_innovation(unittest.TestCase):
    r"""
    Tests the estimation.calculate_innovation function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.u_meas = np.array([1., 2.1, -3.])
        self.u_pred = np.array([1.1, 2.0, -3.1])
        self.exp    = np.array([-0.1, 0.1, 0.1])

    def test_nominal(self) -> None:
        z = estm.calculate_innovation(self.u_meas, self.u_pred)
        np.testing.assert_array_almost_equal(z, self.exp)

#%% estimation.calculate_normalized_innovation
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_calculate_normalized_innovation(unittest.TestCase):
    r"""
    Tests the estimation.calculate_normalized_innovation function with the following cases:
        Nominal
        With explicit inverse
    """
    def setUp(self) -> None:
        self.z = np.array([0.1, 0.05, -0.2])
        self.Pz = np.array([[0.1, 0.01, 0.001], [0.01, 0.1, 0.001], [0., 0., 0.2]])
        self.exp = np.array([0.96868687, 0.41313131, -1.]) # TODO: come up with more independent solution

    def test_nominal(self) -> None:
        nu = estm.calculate_normalized_innovation(self.z, self.Pz)
        np.testing.assert_array_almost_equal(nu, self.exp)

    def test_inverse(self) -> None:
        nu = estm.calculate_normalized_innovation(self.z, self.Pz, use_inverse=True)
        np.testing.assert_array_almost_equal(nu, self.exp)

#%% estimation.calculate_delta_state
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_calculate_delta_state(unittest.TestCase):
    r"""
    Tests the estimation.calculate_delta_state function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.K = np.array([[0.1, 0.01, 0.001], [0.01, 0.1, 0.001], [0., 0., 0.2]])
        self.z = np.array([0.1, 0.05, -0.2])
        self.exp = np.array([0.0103, 0.0058, -0.04])

    def test_nominal(self) -> None:
        dx = estm.calculate_delta_state(self.K, self.z)
        np.testing.assert_array_almost_equal(dx, self.exp)

#%% estimation.propagate_covariance
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_propagate_covariance(unittest.TestCase):
    r"""
    Tests the estimation.propagate_covariance function with the following cases:
        Nominal
        With gamma
        Not inplace (x2)
    """
    def setUp(self) -> None:
        self.phi   = np.diag(np.array([1., 1, 1, -1, -1, -1]))
        self.P     = 1e-3 * np.eye(6)
        self.Q     = np.diag(np.array([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7]))
        self.gamma = -1 * self.phi
        self.exp   = 0.002
        self.orig  = 0.001

    def test_nominal(self) -> None:
        out = estm.propagate_covariance(self.P, self.phi, self.Q)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_gamma(self) -> None:
        out = estm.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_nominal_out(self) -> None:
        out = estm.propagate_covariance(self.P, self.phi, self.Q, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

    def test_gamma_out(self) -> None:
        out = estm.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

#%% estimation.update_covariance
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_update_covariance(unittest.TestCase):
    r"""
    Tests the estimation.update_covariance function with the following cases:
        Nominal inplace
        Nominal not inplace
    """
    def setUp(self) -> None:
        self.P = 1e-3 * np.eye(6)
        self.P[0, -1] = 5e-2
        self.K = np.ones((6, 3))
        self.H = np.hstack((np.eye(3), np.eye(3)))
        self.exp = -0.05
        self.orig = 0.001

    def test_nominal(self) -> None:
        out = estm.update_covariance(self.P, self.K, self.H, inplace=True)
        self.assertIsNone(out)
        self.assertEqual(self.P[-1, -1], self.exp)

    def test_out(self) -> None:
        out = estm.update_covariance(self.P, self.K, self.H, inplace=False)
        self.assertEqual(self.P[-1, -1], self.orig)
        self.assertEqual(out[-1, -1], self.exp)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
