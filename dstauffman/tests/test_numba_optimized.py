r"""
Test file for the `optimized` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
from typing import Any, Callable, Union
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.numba as nub

if HAVE_NUMPY:
    import numpy as np
    pi = np.pi
    inf = np.inf
else:
    from math import inf, pi
try:
    from numba.typed import List
    _HAVE_NUMBA = True
except ModuleNotFoundError:
    List: Callable[[Any], Any] = lambda x: x  # type: ignore[no-redef]
    _HAVE_NUMBA = False

#%% np_any
class Test_np_any(unittest.TestCase):
    r"""
    Tests the np_any function with the following cases:
        All false
        Some true
    """
    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_false(self) -> None:
        x = np.zeros(1000, dtype=bool)
        self.assertFalse(nub.np_any(x))

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_true(self) -> None:
        x = np.zeros(1000, dtype=bool)
        x[333] = True
        self.assertTrue(nub.np_any(x))

    def test_lists(self) -> None:
        x: List[bool] = List([False for i in range(1000)])
        self.assertFalse(nub.np_any(x))
        x[333] = True
        self.assertTrue(nub.np_any(x))

#%% np_all
class Test_np_all(unittest.TestCase):
    r"""
    Tests the np_all function with the following cases:
        All true
        Some false
    """
    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_true(self) -> None:
        x = np.ones(1000, dtype=bool)
        self.assertTrue(nub.np_all(x))

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_false(self) -> None:
        x = np.ones(1000, dtype=bool)
        x[333] = False
        self.assertFalse(nub.np_all(x))

    def test_lists(self) -> None:
        x: List[bool] = List([True for i in range(1000)])
        self.assertTrue(nub.np_all(x))
        x[333] = False
        self.assertFalse(nub.np_all(x))

#%% issorted_opt
class Test_issorted_opt(unittest.TestCase):
    r"""
    Tests the issorted_opt function with the following cases:
        Sorted
        Not sorted
        Reverse sorted (x2)
        Lists
    """
    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(nub.issorted_opt(x))

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(nub.issorted_opt(x))

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_reverse_sorted(self) -> None:
        x = np.array([4, np.pi, 1., -1.])
        self.assertFalse(nub.issorted_opt(x))
        self.assertTrue(nub.issorted_opt(x, descend=True))

    def test_lists(self) -> None:
        x: List[Union[float, int]] = List([-inf, 0, 1, pi, 5, inf])
        self.assertTrue(nub.issorted_opt(x))
        self.assertFalse(nub.issorted_opt(x, descend=True))

#%% prob_to_rate_opt
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_prob_to_rate_opt(unittest.TestCase):
    r"""
    Tests the prob_to_rate_opt function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """
    def setUp(self) -> None:
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = np.hstack((0., -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_conversion(self) -> None:
        rate = nub.prob_to_rate_opt(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = nub.prob_to_rate_opt(self.prob[15], self.time)
        self.assertAlmostEqual(rate, self.rate[15])
        rate = nub.prob_to_rate_opt(float(self.prob[15]), self.time)
        self.assertAlmostEqual(rate, float(self.rate[15]))
        rate = nub.prob_to_rate_opt(1, 1)
        self.assertEqual(rate, np.inf)
        rate = nub.prob_to_rate_opt(0, 1)
        self.assertEqual(rate, 0.)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            nub.prob_to_rate_opt(np.array([0., 0.5, -1.]), 1.)

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            nub.prob_to_rate_opt(np.array([0., 0.5, 1.5]), 1.)

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_circular(self) -> None:
        rate = nub.prob_to_rate_opt(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = nub.rate_to_prob_opt(rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

#%% rate_to_prob_opt
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_rate_to_prob_opt(unittest.TestCase):
    r"""
    Tests the rate_to_prob_opt function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """
    def setUp(self) -> None:
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = np.hstack((0., -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_conversion(self) -> None:
        prob = nub.rate_to_prob_opt(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = nub.rate_to_prob_opt(self.rate[20], self.time)
        self.assertAlmostEqual(prob, self.prob[20])
        prob = nub.rate_to_prob_opt(float(self.rate[20]), self.time)
        self.assertAlmostEqual(prob, float(self.prob[20]))
        prob = nub.rate_to_prob_opt(0, 1)
        self.assertEqual(prob, 0.)
        prob = nub.rate_to_prob_opt(np.inf, 1)
        self.assertEqual(prob, 1)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            nub.rate_to_prob_opt(np.array([0., 0.5, -1.]), 1)

    def test_infinity(self) -> None:
        prob = nub.rate_to_prob_opt(np.inf, 1)
        self.assertAlmostEqual(prob, 1.)

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_circular(self) -> None:
        prob = nub.rate_to_prob_opt(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = nub.prob_to_rate_opt(prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

#%% zero_divide
class Test_zero_divide(unittest.TestCase):
    r"""
    Tests the zero_divide function with the following cases:
        Scalars
        Vectors
        Broadcasting1
        Broadcasting2
    """
    def test_scalars(self) -> None:
        self.assertEqual(nub.zero_divide(1, 2), 0.5)
        self.assertEqual(nub.zero_divide(5, 0), 0.)
        self.assertEqual(nub.zero_divide(0, 0), 0.)
        self.assertEqual(nub.zero_divide(1., 2.), 0.5)
        self.assertEqual(nub.zero_divide(5., 0.), 0.)
        self.assertEqual(nub.zero_divide(0., 0.), 0.)

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_vectors(self) -> None:
        out = nub.zero_divide(np.array([4., 3.14, 0.]), np.array([2., 0., 0.]))
        exp = np.array([2., 0., 0.])
        np.testing.assert_array_equal(out, exp)
        out = nub.zero_divide(np.array([0, -1, -2]), np.array([1, 0, 2]))
        np.testing.assert_array_equal(out, np.array([0, 0, -1]))

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    def test_broadcasting1(self) -> None:
        out = nub.zero_divide(np.array([4., 3.14, 0.]), 2.)
        exp = np.array([2., 1.57, 0.])
        np.testing.assert_array_equal(out, exp)
        out = nub.zero_divide(np.array([4., 3.14, 0.]), 0.)
        exp = np.array([0., 0., 0.])
        np.testing.assert_array_equal(out, exp)

    @unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
    @unittest.expectedFailure
    def test_broadcasting2(self) -> None:
        # Numba broadcasting seems to fail here
        vec = np.array([[1., 0., 0.], [3., 4., 0.], [0., 0., 0.]]).T
        mag = np.array([1., 5., 0.])
        exp = np.array([[1., 0., 0.], [0.6, 0.8, 0.], [0., 0., 0.]]).T
        out = nub.zero_divide(vec, mag)
        np.testing.assert_array_equal(out, exp)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
