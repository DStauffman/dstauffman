# -*- coding: utf-8 -*-
r"""
Test file for the `kalman` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman as dcs

#%% KfInnov
class Test_KfInnov(unittest.TestCase):
    r"""
    Tests the KfInnov class with the following cases:
        TBD
    """
    def test_nominal(self):
        innov = dcs.KfInnov()
        self.assertTrue(isinstance(innov, dcs.KfInnov)) # TODO: test better

#%% KfOut
class Test_KfOut(unittest.TestCase):
    r"""
    Tests the KfOut class with the following cases:
        TBD
    """
    def test_nominal(self):
        kf = dcs.Kf()
        self.assertTrue(isinstance(kf, dcs.Kf)) # TODO: test better

#%% calc_kalman_gain
class Test_kalman_gain(unittest.TestCase):
    r"""
    Tests the kalman_gain function with the following cases:
        Nominal
        With inverse
    """
    def setUp(self):
        self.P = 1e-3 * np.eye(5)
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
        self.R = 0.5 * np.eye(3)
        self.exp = 0.0019950134608610927 # TODO: come up with something that can be known better

    def test_nominal(self):
        K = dcs.calc_kalman_gain(self.P, self.H, self.R)
        self.assertAlmostEqual(K[0, 0], self.exp, 14)

    def test_inverse(self):
        K = dcs.calc_kalman_gain(self.P, self.H, self.R, use_inverse=True)
        self.assertAlmostEqual(K[0, 0], self.exp, 12)

#%% propagate_covariance
class Test_propagate_covariance(unittest.TestCase):
    r"""
    Tests the propagate_covariance function with the following cases:
        Nominal
        With gamma
        Not inplace (x2)
    """
    def setUp(self):
        self.phi   = np.diag([1., 1, 1, -1, -1, -1])
        self.P     = 1e-3 * np.eye(6)
        self.Q     = np.diag([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7])
        self.gamma = -1 * self.phi
        self.exp   = 0.002
        self.orig  = 0.001

    def test_nominal(self):
        out = dcs.propagate_covariance(self.P, self.phi, self.Q)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_gamma(self):
        out = dcs.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma)
        self.assertIsNone(out)
        self.assertEqual(self.P[0, 0], self.exp)

    def test_nominal_out(self):
        out = dcs.propagate_covariance(self.P, self.phi, self.Q, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

    def test_gamma_out(self):
        out = dcs.propagate_covariance(self.P, self.phi, self.Q, gamma=self.gamma, inplace=False)
        self.assertEqual(out[0, 0], self.exp)
        self.assertEqual(self.P[0, 0], self.orig)

#%% update_covariance
class Test_update_covariance(unittest.TestCase):
    r"""
    Tests the update_covariance function with the following cases:
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
        out = dcs.update_covariance(self.P, self.K, self.H, inplace=True)
        self.assertIsNone(out)
        self.assertEqual(self.P[-1, -1], self.exp)

    def test_out(self):
        out = dcs.update_covariance(self.P, self.K, self.H, inplace=False)
        self.assertEqual(self.P[-1, -1], self.orig)
        self.assertEqual(out[-1, -1], self.exp)

#%% plot_attitude
pass # TODO: write this

#%% plot_los
pass # TODO: write this

#%% plot_position
pass # TODO: write this

#%% plot_velocity
pass # TODO: write this

#%% plot_innovations
pass # TODO: write this

#%% plot_covariance
pass # TODO: write this

#%% plot_states
pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
