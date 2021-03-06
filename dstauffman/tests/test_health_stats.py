r"""
Test file for the `stats` module of the "dstauffman.health" library.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

#%% Imports
from typing import Dict
import unittest

from dstauffman import HAVE_NUMPY

import dstauffman.health as health

if HAVE_NUMPY:
    import numpy as np

#%% health.convert_annual_to_monthly_probability
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_convert_annual_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the health.convert_annual_to_monthly_probability function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """
    def setUp(self) -> None:
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self) -> None:
        monthly = health.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_scalar(self) -> None:
        monthly = health.convert_annual_to_monthly_probability(self.annuals[5])
        self.assertAlmostEqual(monthly, self.monthly[5])

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.convert_annual_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            health.convert_annual_to_monthly_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self) -> None:
        monthly = health.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)
        annual = health.convert_monthly_to_annual_probability(monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_alias(self) -> None:
        monthly = health.ca2mp(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

#%% health.convert_monthly_to_annual_probability
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_convert_monthly_to_annual_probability(unittest.TestCase):
    r"""
    Tests the health.convert_monthly_to_annual_probability function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
        check alias to other name
    """
    def setUp(self) -> None:
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self) -> None:
        annual = health.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_scalar(self) -> None:
        annual = health.convert_monthly_to_annual_probability(self.monthly[5])
        self.assertAlmostEqual(annual, self.annuals[5])

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.convert_monthly_to_annual_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            health.convert_monthly_to_annual_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self) -> None:
        annual = health.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)
        monthly = health.convert_annual_to_monthly_probability(annual)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_alias(self) -> None:
        annual = health.cm2ap(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

#%% health.prob_to_rate
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_prob_to_rate(unittest.TestCase):
    r"""
    Tests the health.prob_to_rate function with the following cases:
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

    def test_conversion(self) -> None:
        rate = health.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = health.prob_to_rate(self.prob[15], time=self.time)
        self.assertAlmostEqual(rate, self.rate[15])
        rate = health.prob_to_rate(float(self.prob[15]), time=self.time)
        self.assertAlmostEqual(rate, float(self.rate[15]))
        rate = health.prob_to_rate(1)
        self.assertEqual(rate, np.inf)
        rate = health.prob_to_rate(0)
        self.assertEqual(rate, 0.)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.prob_to_rate(np.array([0., 0.5, -1.]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            health.prob_to_rate(np.array([0., 0.5, 1.5]))

    def test_circular(self) -> None:
        rate = health.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = health.rate_to_prob(rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

#%% health.rate_to_prob
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_rate_to_prob(unittest.TestCase):
    r"""
    Tests the health.rate_to_prob function with the following cases:
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

    def test_conversion(self) -> None:
        prob = health.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = health.rate_to_prob(self.rate[20], time=self.time)
        self.assertAlmostEqual(prob, self.prob[20])
        prob = health.rate_to_prob(float(self.rate[20]), time=self.time)
        self.assertAlmostEqual(prob, float(self.prob[20]))
        prob = health.rate_to_prob(0)
        self.assertEqual(prob, 0.)
        prob = health.rate_to_prob(np.inf)
        self.assertEqual(prob, 1)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.rate_to_prob(np.array([0., 0.5, -1.]))

    def test_infinity(self) -> None:
        prob = health.rate_to_prob(np.inf)
        self.assertAlmostEqual(prob, 1.)

    def test_circular(self) -> None:
        prob = health.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = health.prob_to_rate(prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

#%% health.annual_rate_to_monthly_probability
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_annual_rate_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the health.annual_rate_to_monthly_probability function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """
    def setUp(self) -> None:
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self) -> None:
        prob = health.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = health.annual_rate_to_monthly_probability(0)
        self.assertIn(prob, self.prob)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.annual_rate_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_infinity(self) -> None:
        prob = health.annual_rate_to_monthly_probability(np.inf)
        self.assertAlmostEqual(prob, 1.)

    def test_circular(self) -> None:
        prob = health.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = health.monthly_probability_to_annual_rate(prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_alias(self) -> None:
        prob = health.ar2mp(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

#%% health.monthly_probability_to_annual_rate
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_monthly_probability_to_annual_rate(unittest.TestCase):
    r"""
    Tests the health.monthly_probability_to_annual_rate function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """
    def setUp(self) -> None:
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self) -> None:
        rate = health.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = health.monthly_probability_to_annual_rate(0)
        self.assertIn(rate, self.rate)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            health.monthly_probability_to_annual_rate(np.array([0., 0.5, -1.]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            health.monthly_probability_to_annual_rate(np.array([0., 0.5, 1.5]))

    def test_circular(self) -> None:
        rate = health.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = health.rate_to_prob(rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_alias(self) -> None:
        rate = health.ar2mp(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

#%% health.combine_sets
class Test_health_combine_sets(unittest.TestCase):
    r"""
    Tests the health.combine_sets function with the following cases:
        Normal use
        No deviation
        Empty set 1
        Empty set 2
        All empty
        Exactly one point
        Negative values (should silently fail)
        Negative values, weird exception case (should raise error)
        Array cases (should raise error)
    """
    def setUp(self) -> None:
        self.n1 = 5
        self.u1 = 1
        self.s1 = 0.5
        self.n2 = 10
        self.u2 = 2
        self.s2 = 0.25
        self.n  = 15
        self.u  = 1.6666666666666667
        self.s  = 0.59135639081046598

    def test_nominal(self) -> None:
        (n, u, s) = health.combine_sets(self.n1, self.u1, self.s1, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n)
        self.assertAlmostEqual(u, self.u)
        self.assertAlmostEqual(s, self.s)

    def test_no_deviation(self) -> None:
        (n, u, s) = health.combine_sets(self.n1, self.u1, 0, self.n1, self.u1, 0)
        self.assertEqual(n, 2*self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, 0)

    def test_empty1(self) -> None:
        (n, u, s) = health.combine_sets(0, 0, 0, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n2)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_empty2(self) -> None:
        (n, u, s) = health.combine_sets(self.n1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_all_empty(self) -> None:
        (n, u, s) = health.combine_sets(0, 0, 0, 0, 0, 0)
        self.assertEqual(n, 0)
        self.assertEqual(u, 0)
        self.assertEqual(s, 0)

    def test_exactly_one_point1(self) -> None:
        (n, u, s) = health.combine_sets(1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_exactly_one_point2(self) -> None:
        (n, u, s) = health.combine_sets(0, 0, 0, 1, self.u2, self.s2)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_negatives(self) -> None:
        with self.assertRaises((AssertionError, ValueError)):
            health.combine_sets(-self.n1, -self.u1, -self.s1, -self.n2, -self.u2, -self.s2)

    def test_negative_weird(self) -> None:
        with self.assertRaises((AssertionError, ValueError)):
            health.combine_sets(5, self.u1, self.s1, -4, self.u2, self.s2)

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_broadcasting(self) -> None:
        with self.assertRaises(ValueError):
            (n, u, s) = health.combine_sets(np.array([self.n1, self.n1]), self.u1, self.s1, self.n2, self.u2, self.s2)

#%% health.bounded_normal_draw
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_bounded_normal_draw(unittest.TestCase):
    r"""
    Tests the health.bounded_normal_draw function with the following cases:
        Nominal
        No bounds
        Optional values
        Zero STD
    """
    def setUp(self) -> None:
        self.num    = 100000
        self.mean   = 100.
        self.std    = 50.
        self.min    = 20.
        self.max    = 200.
        self.values = {'test_mean': self.mean, 'test_std': self.std, 'test_min': self.min, 'test_max': self.max}
        self.field  = 'test'
        self.prng   = np.random.RandomState()

    def test_nominal(self) -> None:
        out = health.bounded_normal_draw(self.num, self.values, self.field, self.prng)
        self.assertTrue(np.min(out) >= self.min)
        self.assertTrue(np.max(out) <= self.max)

    def test_bounds(self) -> None:
        self.values['no_bounds_mean'] = self.mean
        self.values['no_bounds_std']  = self.std
        out = health.bounded_normal_draw(self.num, self.values, 'no_bounds', self.prng)
        self.assertTrue(np.min(out) < self.min)
        self.assertTrue(np.max(out) > self.max)
        mean = np.mean(out)
        std  = np.std(out)
        self.assertTrue(self.mean - 1 < mean < self.mean + 1)
        self.assertTrue(self.std - 1 < std < self.std + 1)

    def test_optional_values(self) -> None:
        values: Dict[str, float] = {}
        out = health.bounded_normal_draw(self.num, values, '', self.prng)
        mean = np.mean(out)
        std  = np.std(out)
        self.assertTrue(np.abs(mean - 0) < 1e-2, 'Bad mean.')
        self.assertTrue(np.abs(std - 1) < 1e-2, 'Bad Std.')

    def test_no_std(self) -> None:
        self.values['test_std'] = 0
        out = health.bounded_normal_draw(self.num, self.values, self.field, self.prng)
        self.assertTrue(np.all(np.abs(out - self.mean) < 1e-8))

#%% health.rand_draw
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_rand_draw(unittest.TestCase):
    r"""
    Tests the health.rand_draw function with the following cases:
        Nominal
        Negative and zero values
        Large values
        No quality checks
    """
    def setUp(self) -> None:
        self.chances = np.array([0.1, 0.2, 0.8, 0.9])
        self.prng = np.random.RandomState()

    def test_nominal(self) -> None:
        is_set = health.rand_draw(self.chances, self.prng)
        self.assertTrue(is_set.dtype == bool)

    def test_zeros(self) -> None:
        is_set = health.rand_draw(np.array([-5, 0, 0.5]), self.prng)
        self.assertFalse(is_set[0])
        self.assertFalse(is_set[1])

    def test_large_values(self) -> None:
        is_set = health.rand_draw(np.array([5, 1, 0.5, 1000, np.inf]), self.prng)
        self.assertTrue(is_set[0])
        self.assertTrue(is_set[1])
        self.assertTrue(is_set[3])
        self.assertTrue(is_set[4])

    def test_without_checks(self) -> None:
        is_set = health.rand_draw(np.array([-5, 0.5, 5]), self.prng, check_bounds=False)
        self.assertFalse(is_set[0])
        self.assertTrue(is_set[2])

#%% health.ecdf
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_ecdf(unittest.TestCase):
    r"""
    Tests the health.ecdf function with the following cases:
        Nominal
        Integers
        List
        Not unique
    """
    def test_nominal(self) -> None:
        y = np.random.rand(10000)
        (x, f) = health.ecdf(y)
        dt = x[1] - x[0]
        exp = np.arange(x[0], x[-1]+dt, dt)
        np.testing.assert_array_almost_equal(f, exp, 1)

    def test_integers(self) -> None:
        y = np.array([0, 0, 0, 1, 1])
        (x, f) = health.ecdf(y)
        np.testing.assert_array_almost_equal(x, np.array([0.6, 1.]), 14)
        np.testing.assert_array_equal(f, np.array([0, 1]))

    def test_list(self) -> None:
        y = [0, 0.1, 0.2, 0.8, 0.9, 1.0]
        (x, f) = health.ecdf(y)
        np.testing.assert_array_almost_equal(x, np.arange(1, 7)/6, 14)
        np.testing.assert_array_equal(f, y)

    def test_unique(self) -> None:
        y = np.array([0., 0., 0., 0.5, 0.5, 0.5, 1.])
        (x, f) = health.ecdf(y)
        np.testing.assert_array_almost_equal(x, np.array([3/7, 6/7, 1.]), 14)
        np.testing.assert_array_equal(f, np.array([0., 0.5, 1.]))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
