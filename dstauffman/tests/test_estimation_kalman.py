r"""
Test file for the `kalman` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman.estimation as estm

#%% estimation.calc_kalman_gain
class Test_estimation_calc_kalman_gain(unittest.TestCase):
    r"""
    Tests the estimation.calc_kalman_gain function with the following cases:
        Nominal
        With inverse
    """
    def setUp(self):
        self.P = 1e-3 * np.eye(5)
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
        self.R = 0.5 * np.eye(3)
        self.exp = 0.0019950134608610927 # TODO: come up with something that can be known better

    def test_nominal(self):
        K = estm.calc_kalman_gain(self.P, self.H, self.R)
        self.assertAlmostEqual(K[0, 0], self.exp, 14)

    def test_inverse(self):
        K = estm.calc_kalman_gain(self.P, self.H, self.R, use_inverse=True)
        self.assertAlmostEqual(K[0, 0], self.exp, 12)

#%% estimation.propagate_covariance
class Test_estimation_propagate_covariance(unittest.TestCase):
    r"""
    Tests the estimation.propagate_covariance function with the following cases:
        Nominal
        With gamma
        Not inplace (x2)
    """
    def setUp(self):
        self.phi   = np.diag(np.array([1., 1, 1, -1, -1, -1]))
        self.P     = 1e-3 * np.eye(6)
        self.Q     = np.diag(np.array([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7]))
        self.gamma = -1 * self.phi
        self.exp   = 0.002
        self.orig  = 0.001

    def test_nominal(self):
        out = estm.propagate_covariance(self.P, self.phi, self.Q)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_gamma(self):
        out = estm.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_nominal_out(self):
        out = estm.propagate_covariance(self.P, self.phi, self.Q, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

    def test_gamma_out(self):
        out = estm.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

#%% estimation.update_covariance
class Test_estimation_update_covariance(unittest.TestCase):
    r"""
    Tests the estimation.update_covariance function with the following cases:
        Nominal inplace
        Nominal not inplace
    """
    def setUp(self):
        self.P = 1e-3 * np.eye(6)
        self.P[0, -1] = 5e-2
        self.K = np.ones((6, 3))
        self.H = np.hstack((np.eye(3), np.eye(3)))
        self.exp = -0.05
        self.orig = 0.001

    def test_nominal(self):
        out = estm.update_covariance(self.P, self.K, self.H, inplace=True)
        self.assertIsNone(out)
        self.assertEqual(self.P[-1, -1], self.exp)

    def test_out(self):
        out = estm.update_covariance(self.P, self.K, self.H, inplace=False)
        self.assertEqual(self.P[-1, -1], self.orig)
        self.assertEqual(out[-1, -1], self.exp)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
