# -*- coding: utf-8 -*-
r"""
Test file for the `stats` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

#%% Imports
import sys
import unittest

import numpy as np

import dstauffman as dcs

#%% convert_annual_to_monthly_probability
class Test_convert_annual_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with these cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
    """
    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_scalar(self):
        monthly = dcs.convert_annual_to_monthly_probability(0)
        self.assertIn(monthly, self.monthly)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)
        annual = dcs.convert_monthly_to_annual_probability(monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_alias(self):
        monthly = dcs.ca2mp(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

#%% convert_monthly_to_annual_probability
class Test_convert_monthly_to_annual_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with these cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """
    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_scalar(self):
        annual = dcs.convert_monthly_to_annual_probability(0)
        self.assertIn(annual, self.annuals)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)
        monthly = dcs.convert_annual_to_monthly_probability(annual)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_alias(self):
        annual = dcs.cm2ap(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

#%% prob_to_rate
class Test_prob_to_rate(unittest.TestCase):
    r"""
    Tests the prob_to_rate function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = -np.log(1 - self.prob) / self.time

    def test_conversion(self):
        rate = dcs.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self):
        rate = dcs.prob_to_rate(0)
        self.assertIn(rate, self.rate)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.prob_to_rate(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.prob_to_rate(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        rate = dcs.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = dcs.rate_to_prob(rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

#%% rate_to_prob
class Test_rate_to_prob(unittest.TestCase):
    r"""
    Tests the rate_to_prob function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = -np.log(1 - self.prob) / self.time

    def test_conversion(self):
        prob = dcs.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self):
        prob = dcs.rate_to_prob(0)
        self.assertIn(prob, self.prob)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.rate_to_prob(np.array([0., 0.5, -1.]))

    def test_infinity(self):
        prob = dcs.rate_to_prob(np.inf)
        self.assertAlmostEqual(prob, 1.)

    def test_circular(self):
        prob = dcs.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = dcs.prob_to_rate(prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

#%% annual_rate_to_monthly_probability
class Test_annual_rate_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the annual_rate_to_monthly_probability function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self):
        prob = dcs.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self):
        prob = dcs.annual_rate_to_monthly_probability(0)
        self.assertIn(prob, self.prob)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.annual_rate_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_infinity(self):
        prob = dcs.annual_rate_to_monthly_probability(np.inf)
        self.assertAlmostEqual(prob, 1.)

    def test_circular(self):
        prob = dcs.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = dcs.monthly_probability_to_annual_rate(prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_alias(self):
        prob = dcs.ar2mp(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

#%% monthly_probability_to_annual_rate
class Test_monthly_probability_to_annual_rate(unittest.TestCase):
    r"""
    Tests the monthly_probability_to_annual_rate function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self):
        rate = dcs.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self):
        rate = dcs.monthly_probability_to_annual_rate(0)
        self.assertIn(rate, self.rate)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.monthly_probability_to_annual_rate(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.monthly_probability_to_annual_rate(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        rate = dcs.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = dcs.rate_to_prob(rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_alias(self):
        rate = dcs.ar2mp(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

#%% combine_sets
class Test_combine_sets(unittest.TestCase):
    r"""
    Tests the combine_sets function with the following cases:
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
    def setUp(self):
        self.n1 = 5
        self.u1 = 1
        self.s1 = 0.5
        self.n2 = 10
        self.u2 = 2
        self.s2 = 0.25
        self.n  = 15
        self.u  = 1.6666666666666667
        self.s  = 0.59135639081046598

    def test_nominal(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, self.s1, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n)
        self.assertAlmostEqual(u, self.u)
        self.assertAlmostEqual(s, self.s)

    def test_no_deviation(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, 0, self.n1, self.u1, 0)
        self.assertEqual(n, 2*self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, 0)

    def test_empty1(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n2)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_empty2(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_all_empty(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, 0, 0, 0)
        self.assertEqual(n, 0)
        self.assertEqual(u, 0)
        self.assertEqual(s, 0)

    def test_exactly_one_point1(self):
        (n, u, s) = dcs.combine_sets(1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_exactly_one_point2(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, 1, self.u2, self.s2)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_negatives(self):
        try:
            dcs.combine_sets(-self.n1, -self.u1, -self.s1, -self.n2, -self.u2, -self.s2)
        except:
            self.assertTrue(sys.exc_info()[0] in [AssertionError, ValueError])

    def test_negative_weird(self):
        try:
            dcs.combine_sets(5, self.u1, self.s1, -4, self.u2, self.s2)
        except:
            self.assertTrue(sys.exc_info()[0] in [AssertionError, ValueError])

    def test_broadcasting(self):
        with self.assertRaises(ValueError):
            (n, u, s) = dcs.combine_sets(np.array([self.n1, self.n1]), self.u1, self.s1, self.n2, self.u2, self.s2)

#%% bounded_normal_draw
class Test_bounded_normal_draw(unittest.TestCase):
    r"""
    Tests the bounded_normal_draw function with the following cases:
        TBD
    """
    def setUp(self):
        self.num    = 100000
        self.mean   = 100
        self.std    = 50
        self.min    = 20
        self.max    = 200
        self.values = {'test_mean': self.mean, 'test_std': self.std, 'test_min': self.min, 'test_max': self.max}
        self.field  = 'test'
        self.prng   = np.random.RandomState()

    def test_nominal(self):
        out = dcs.bounded_normal_draw(self.num, self.values, self.field, self.prng)
        self.assertTrue(np.min(out) >= self.min)
        self.assertTrue(np.max(out) <= self.max)

    def test_bounds(self):
        self.values['no_bounds_mean'] = self.mean
        self.values['no_bounds_std']  = self.std
        out = dcs.bounded_normal_draw(self.num, self.values, 'no_bounds', self.prng)
        self.assertTrue(np.min(out) < self.min)
        self.assertTrue(np.max(out) > self.max)

    def test_mean(self):
        self.values['no_bounds_mean'] = self.mean
        self.values['no_bounds_std']  = self.std
        out = dcs.bounded_normal_draw(self.num, self.values, 'no_bounds', self.prng)
        mean = np.mean(out)
        std  = np.std(out)
        self.assertTrue(self.mean - 1 < mean < self.mean + 1)
        self.assertTrue(self.std - 1 < std < self.std + 1)

    def test_optional_values(self):
        values = {}
        out = dcs.bounded_normal_draw(self.num, values, '', self.prng)
        mean = np.mean(out)
        std  = np.std(out)
        self.assertTrue(np.abs(mean - 0) < 1e-2, 'Bad mean.')
        self.assertTrue(np.abs(std - 1) < 1e-2, 'Bad Std.')

    def test_no_std(self):
        self.values['test_std'] = 0
        out = dcs.bounded_normal_draw(self.num, self.values, self.field, self.prng)
        self.assertTrue(np.all(np.abs(out - self.mean) < 1e-8))

#%% z_from_ci
class Test_z_from_ci(unittest.TestCase):
    r"""
    Tests the z_from_ci function with the following cases:
        Nominal with 4 common values found online
    """
    def setUp(self):
        self.cis = [0.90,  0.95,  0.98,  0.99]
        self.zs  = [1.645, 1.96, 2.326, 2.576]

    def test_nominal(self):
        for (ci, exp_z) in zip(self.cis, self.zs):
            z = dcs.z_from_ci(ci)
            self.assertTrue(abs(z - exp_z) < 0.001, '{} and {} are more than 0.001 from each other.'.format(z, exp_z))

#%% rand_draw
class Test_rand_draw(unittest.TestCase):
    r"""
    Tests the rand_draw function with the following cases:
        Nominal
        Negative values
        Large values
    """
    def setUp(self):
        self.chances = np.array([0.1, 0.2, 0.8, 0.9])
        self.prng = np.random.RandomState()

    def test_nominal(self):
        is_set = dcs.rand_draw(self.chances, self.prng)
        self.assertTrue(is_set.dtype == bool)

    def test_zeros(self):
        is_set = dcs.rand_draw(np.array([-5, 0, 0.5]), self.prng)
        self.assertFalse(is_set[0])
        self.assertFalse(is_set[1])

    def test_ones(self):
        is_set = dcs.rand_draw(np.array([5, 1, 0.5, 1000, np.inf]), self.prng)
        self.assertTrue(is_set[0])
        self.assertTrue(is_set[1])
        self.assertTrue(is_set[3])
        self.assertTrue(is_set[4])

    def test_without_checks(self):
        is_set = dcs.rand_draw(np.array([-5, 0.5, 5]), self.prng, check_bounds=False)
        self.assertFalse(is_set[0])
        self.assertTrue(is_set[2])

#%% intersect
class Test_intersect(unittest.TestCase):
    r"""
    Tests the intersect function with the following cases:
        Nominal
        Floats
        Assume unique
    """
    def test_nominal(self):
        a = np.array([1, 2, 4, 4, 6], dtype=int)
        b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
        (c, ia, ib) = dcs.intersect(a, b)
        np.testing.assert_array_equal(c, np.array([2, 6], dtype=int))
        np.testing.assert_array_equal(ia, np.array([1, 4], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 6], dtype=int))

    def test_floats(self):
        a = np.array([1, 2.5, 4, 6], dtype=float)
        b = np.array([0, 8, 2.5, 4, 6], dtype=float)
        (c, ia, ib) = dcs.intersect(a, b)
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

    def test_unique(self):
        a = np.array([1, 2.5, 4, 6], dtype=float)
        b = np.array([0, 8, 2.5, 4, 6], dtype=float)
        (c, ia, ib) = dcs.intersect(a, b, assume_unique=True)
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
