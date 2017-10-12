# -*- coding: utf-8 -*-
r"""
Test file for the `linalg` module of the dstauffman code.  It is intented to contain test cases to
demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in April 2016.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman as dcs

#%% orth
class Test_orth(unittest.TestCase):
    r"""
    Tests the orth function with the following cases:
        Rank 3 matrix
        Rank 2 matrix
    """
    def setUp(self):
        self.A1 = np.array([[1, 0, 1], [-1, -2, 0], [0, 1, -1]])
        self.r1 = 3
        self.Q1 = np.array([[-0.12000026, -0.80971228, 0.57442663], [ 0.90175265, 0.15312282, \
            0.40422217], [-0.41526149, 0.5664975, 0.71178541]])
        self.A2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.r2 = 2
        self.Q2 = np.array([[-0.70710678, 0.,], [0., 1.], [-0.70710678, 0.]])

    def test_rank3(self):
        r = np.linalg.matrix_rank(self.A1)
        self.assertEqual(r, self.r1)
        Q = dcs.orth(self.A1)
        np.testing.assert_array_almost_equal(Q, self.Q1)

    def test_rank2(self):
        r = np.linalg.matrix_rank(self.A2)
        self.assertEqual(r, self.r2)
        Q = dcs.orth(self.A2)
        np.testing.assert_array_almost_equal(Q, self.Q2)

#%% subspace
class Test_subspace(unittest.TestCase):
    r"""
    Tests the subspace function with the followinc cases:
        Nominal
    """
    def setUp(self):
        self.A = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, -1], \
            [1, -1, -1], [-1, -1, 1]])
        self.B = np.array([[1, 1, 1, 1], [1, -1, 1, -1],[1, 1, -1, -1], [1, -1, -1, 1], [-1, -1, -1, -1], \
            [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1]])
        self.theta = np.pi / 2

    def test_nominal(self):
        theta = dcs.subspace(self.A, self.B)
        self.assertAlmostEqual(theta, self.theta)

    def test_swapped_rank(self):
        theta = dcs.subspace(self.B, self.A)
        self.assertAlmostEqual(theta, self.theta)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
