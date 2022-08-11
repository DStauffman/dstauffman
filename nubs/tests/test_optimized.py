r"""
Test file for the `optimized` module of the "nubs" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
from typing import Any, Callable, List, Union
import unittest

import nubs as nubs

if nubs.HAVE_NUMPY:
    import numpy as np

    pi = np.pi
    inf = np.inf
else:
    from math import inf, pi
try:
    from numba.typed import List as nubList

    _HAVE_NUMBA = True
except ModuleNotFoundError:
    nubList: Callable[[Any], Any] = lambda x: x  # type: ignore[no-redef]
    _HAVE_NUMBA = False

#%% np_any
class Test_np_any(unittest.TestCase):
    r"""
    Tests the np_any function with the following cases:
        All false
        Some true
    """

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_false(self) -> None:
        x = np.zeros(1000, dtype=bool)
        self.assertFalse(nubs.np_any(x))

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_true(self) -> None:
        x = np.zeros(1000, dtype=bool)
        x[333] = True
        self.assertTrue(nubs.np_any(x))

    def test_lists(self) -> None:
        x: List[bool] = nubList([False for i in range(1000)])
        self.assertFalse(nubs.np_any(x))
        x[333] = True
        self.assertTrue(nubs.np_any(x))


#%% np_all
class Test_np_all(unittest.TestCase):
    r"""
    Tests the np_all function with the following cases:
        All true
        Some false
    """

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_true(self) -> None:
        x = np.ones(1000, dtype=bool)
        self.assertTrue(nubs.np_all(x))

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_false(self) -> None:
        x = np.ones(1000, dtype=bool)
        x[333] = False
        self.assertFalse(nubs.np_all(x))

    def test_lists(self) -> None:
        x: List[bool] = nubList([True for i in range(1000)])
        self.assertTrue(nubs.np_all(x))
        x[333] = False
        self.assertFalse(nubs.np_all(x))


#%% issorted_opt
class Test_issorted_opt(unittest.TestCase):
    r"""
    Tests the issorted_opt function with the following cases:
        Sorted
        Not sorted
        Reverse sorted (x2)
        Lists
    """

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(nubs.issorted_opt(x))

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(nubs.issorted_opt(x))

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_reverse_sorted(self) -> None:
        x = np.array([4, np.pi, 1.0, -1.0])
        self.assertFalse(nubs.issorted_opt(x))
        self.assertTrue(nubs.issorted_opt(x, descend=True))

    def test_lists(self) -> None:
        x: List[Union[float, int]] = nubList([-inf, 0, 1, pi, 5, inf])
        self.assertTrue(nubs.issorted_opt(x))
        self.assertFalse(nubs.issorted_opt(x, descend=True))


#%% prob_to_rate_opt
@unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
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
        self.rate = np.hstack((0.0, -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    @unittest.skipIf(not _HAVE_NUMBA, "Skipping due to missing numba dependency.")
    def test_conversion(self) -> None:
        rate = nubs.prob_to_rate_opt(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = nubs.prob_to_rate_opt(self.prob[15], self.time)
        self.assertAlmostEqual(rate, self.rate[15])
        rate = nubs.prob_to_rate_opt(float(self.prob[15]), self.time)
        self.assertAlmostEqual(rate, float(self.rate[15]))
        rate = nubs.prob_to_rate_opt(1, 1)
        self.assertEqual(rate, np.inf)
        rate = nubs.prob_to_rate_opt(0, 1)
        self.assertEqual(rate, 0.0)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            nubs.prob_to_rate_opt(np.array([0.0, 0.5, -1.0]), 1.0)

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            nubs.prob_to_rate_opt(np.array([0.0, 0.5, 1.5]), 1.0)

    @unittest.skipIf(not _HAVE_NUMBA, "Skipping due to missing numba dependency.")
    def test_circular(self) -> None:
        rate = nubs.prob_to_rate_opt(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = nubs.rate_to_prob_opt(rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)


#%% rate_to_prob_opt
@unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
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
        self.rate = np.hstack((0.0, -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    @unittest.skipIf(not _HAVE_NUMBA, "Skipping due to missing numba dependency.")
    def test_conversion(self) -> None:
        prob = nubs.rate_to_prob_opt(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = nubs.rate_to_prob_opt(self.rate[20], self.time)
        self.assertAlmostEqual(prob, self.prob[20])
        prob = nubs.rate_to_prob_opt(float(self.rate[20]), self.time)
        self.assertAlmostEqual(prob, float(self.prob[20]))
        prob = nubs.rate_to_prob_opt(0, 1)
        self.assertEqual(prob, 0.0)
        prob = nubs.rate_to_prob_opt(np.inf, 1)
        self.assertEqual(prob, 1)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            nubs.rate_to_prob_opt(np.array([0.0, 0.5, -1.0]), 1)

    def test_infinity(self) -> None:
        prob = nubs.rate_to_prob_opt(np.inf, 1)
        self.assertAlmostEqual(prob, 1.0)

    @unittest.skipIf(not _HAVE_NUMBA, "Skipping due to missing numba dependency.")
    def test_circular(self) -> None:
        prob = nubs.rate_to_prob_opt(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = nubs.prob_to_rate_opt(prob, self.time)
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
        self.assertEqual(nubs.zero_divide(1, 2), 0.5)
        self.assertEqual(nubs.zero_divide(5, 0), 0.0)
        self.assertEqual(nubs.zero_divide(0, 0), 0.0)
        self.assertEqual(nubs.zero_divide(1.0, 2.0), 0.5)
        self.assertEqual(nubs.zero_divide(5.0, 0.0), 0.0)
        self.assertEqual(nubs.zero_divide(0.0, 0.0), 0.0)

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_vectors(self) -> None:
        out = nubs.zero_divide(np.array([4.0, 3.14, 0.0]), np.array([2.0, 0.0, 0.0]))
        exp = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)
        out = nubs.zero_divide(np.array([0, -1, -2]), np.array([1, 0, 2]))
        np.testing.assert_array_equal(out, np.array([0, 0, -1]))

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_broadcasting1(self) -> None:
        out = nubs.zero_divide(np.array([4.0, 3.14, 0.0]), 2.0)
        exp = np.array([2.0, 1.57, 0.0])
        np.testing.assert_array_equal(out, exp)
        out = nubs.zero_divide(np.array([4.0, 3.14, 0.0]), 0.0)
        exp = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)

    @unittest.skipIf(not nubs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_broadcasting2(self) -> None:
        # Numba broadcasting fails here prior to v0.53
        vec = np.array([[1.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]).T
        mag = np.array([1.0, 5.0, 0.0])
        exp = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.0]]).T
        try:
            out = nubs.zero_divide(vec, mag)
        except FloatingPointError:
            self.skipTest("Skipping due to numba vectorize optimization bug.")
        np.testing.assert_array_equal(out, exp)  # pragma: no cover


#%% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
