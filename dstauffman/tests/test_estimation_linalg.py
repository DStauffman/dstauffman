r"""
Test file for the `linalg` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in April 2016.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.estimation as estm

if HAVE_NUMPY:
    import numpy as np

#%% estimation.orth
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_orth(unittest.TestCase):
    r"""
    Tests the estimation.orth function with the following cases:
        Rank 3 matrix
        Rank 2 matrix
    """

    def setUp(self) -> None:
        self.A1 = np.array([[1, 0, 1], [-1, -2, 0], [0, 1, -1]])
        self.r1 = 3
        self.Q1 = np.array(
            [[-0.12000026, -0.80971228, 0.57442663], [0.90175265, 0.15312282, 0.40422217], [-0.41526149, 0.5664975, 0.71178541]]
        )
        self.A2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.r2 = 2
        self.Q2 = np.array([[-0.70710678, 0.0,], [0.0, 1.0], [-0.70710678, 0.0]])

    def test_rank3(self) -> None:
        r = np.linalg.matrix_rank(self.A1)
        self.assertEqual(r, self.r1)
        Q = estm.orth(self.A1)
        np.testing.assert_array_almost_equal(Q, self.Q1)

    def test_rank2(self) -> None:
        r = np.linalg.matrix_rank(self.A2)
        self.assertEqual(r, self.r2)
        Q = estm.orth(self.A2)
        np.testing.assert_array_almost_equal(Q, self.Q2)


#%% estimation.subspace
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_subspace(unittest.TestCase):
    r"""
    Tests the estimation.subspace function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.A = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
        self.B = np.array([
            [ 1,  1,  1,  1],
            [ 1, -1,  1, -1],
            [ 1,  1, -1, -1],
            [ 1, -1, -1,  1],
            [-1, -1, -1, -1],
            [-1,  1, -1,  1],
            [-1, -1,  1,  1],
            [-1,  1,  1, -1],
        ])
        self.theta = np.pi / 2

    def test_nominal(self) -> None:
        theta = estm.subspace(self.A, self.B)
        self.assertAlmostEqual(theta, self.theta)

    def test_swapped_rank(self) -> None:
        theta = estm.subspace(self.B, self.A)
        self.assertAlmostEqual(theta, self.theta)


#%% estimation.mat_divide
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_estimation_mat_divide(unittest.TestCase):
    r"""
    Tests the estimation.mat_divide function with the following cases:
        Nominal
        Poorly-conditioned
    """

    def test_nominal(self) -> None:
        a = np.array([[1, 2], [3, 4]], dtype=float)
        exp = np.array([1, -1], dtype=float)
        b = a @ exp
        x = estm.mat_divide(a, b)
        np.testing.assert_array_almost_equal(x, exp, 14)

    def test_rcond(self) -> None:
        a = np.array([[1e6, 1e6], [1e6, 1e6 + 1e-8]], dtype=float)
        exp = np.array([1, -1], dtype=float)
        b = a @ exp
        x1 = estm.mat_divide(a, b, rcond=1e-16)
        x2 = estm.mat_divide(a, b, rcond=1e-6)
        np.testing.assert_array_almost_equal(x1, exp, 2)
        np.testing.assert_array_almost_equal(x2, np.zeros(2), 2)


#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
