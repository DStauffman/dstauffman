# -*- coding: utf-8 -*-
r"""
Test file for the `stats` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

#%% Imports
import numpy as np
import sys
import unittest
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
        np.testing.assert_almost_equal(monthly, self.monthly)

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
        np.testing.assert_almost_equal(monthly, self.monthly)
        annual = dcs.convert_monthly_to_annual_probability(monthly)
        np.testing.assert_almost_equal(annual, self.annuals)

    def test_alias(self):
        monthly = dcs.ca2mp(self.annuals)
        np.testing.assert_almost_equal(monthly, self.monthly)

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
        np.testing.assert_almost_equal(annual, self.annuals)

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
        np.testing.assert_almost_equal(annual, self.annuals)
        monthly = dcs.convert_annual_to_monthly_probability(annual)
        np.testing.assert_almost_equal(monthly, self.monthly)

    def test_alias(self):
        annual = dcs.cm2ap(self.monthly)
        np.testing.assert_almost_equal(annual, self.annuals)

#%% prob_to_rate
class Test_prob_to_rate(unittest.TestCase):
    r"""
    Tests the prob_to_rate function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        with np.errstate(divide='ignore'):
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
        np.testing.assert_almost_equal(rate, self.rate)
        prob = dcs.rate_to_prob(rate, self.time)
        np.testing.assert_almost_equal(prob, self.prob)

#%% rate_to_prob
class Test_rate_to_prob(unittest.TestCase):
    r"""
    Tests the rate_to_prob function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        with np.errstate(divide='ignore'):
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
        np.testing.assert_almost_equal(prob, self.prob)
        rate = dcs.prob_to_rate(prob, self.time)
        np.testing.assert_almost_equal(rate, self.rate)

#%% month_prob_mult_ratio
class Test_month_prob_mult_ratio(unittest.TestCase):
    r"""
    Tests the month_prob_mult_ratio function with these cases:
        TBD
    """
    def setUp(self):
        self.prob = np.arange(0, 1.01, 0.01)
        rate           = dcs.prob_to_rate(self.prob, time=1./12)
        mult_rate_2    = rate * 2
        mult_rate_half = rate * 0.5
        self.mult_prob_2    = dcs.rate_to_prob(mult_rate_2, time=1./12)
        self.mult_prob_half = dcs.rate_to_prob(mult_rate_half, time=1./12)
        
    def test_no_mult(self):
        mult_prob = dcs.month_prob_mult_ratio(self.prob, 1)
        np.testing.assert_array_almost_equal(mult_prob, self.prob)
        
    def test_gt_one_mult(self):
        mult_prob = dcs.month_prob_mult_ratio(self.prob, 2)
        np.testing.assert_array_almost_equal(mult_prob, self.mult_prob_2)
    
    def test_lt_one_mult(self):
        mult_prob = dcs.month_prob_mult_ratio(self.prob, 0.5)
        np.testing.assert_array_almost_equal(mult_prob, self.mult_prob_half)
    
    def test_zero_mult(self):
        mult_prob = dcs.month_prob_mult_ratio(self.prob[:-1], 0)
        np.testing.assert_array_almost_equal(mult_prob, np.zeros(len(self.prob)-1))
    
#%% annual_rate_to_monthly_probability
class Test_annual_rate_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the annual_rate_to_monthly_probability function with these cases:
        TBD
    """
    rate = np.array([0, 0.5, 1, 5, np.inf])
    prob = dcs.annual_rate_to_monthly_probability(rate)

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

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
