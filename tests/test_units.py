# -*- coding: utf-8 -*-
r"""
Test file for the `units` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in February 2016.
"""

#%% Imports
import unittest
import dstauffman as dcs

#%% Units
class Test_Units(unittest.TestCase):
    r"""
    Tests the Units class with the following cases:
        Probability and rate permutations (x9)
        Multiplication conversions (x3)
        Bad value errors (x3)
    """
    def test_prob_rate_conversions1a(self):
        new = dcs.Units.convert(0.5, dcs.Units.MP, dcs.Units.MP)
        exp = 0.5
        self.assertEqual(new, exp)
    def test_prob_rate_conversions1b(self):
        new = dcs.Units.convert(0.5, dcs.Units.MP, dcs.Units.AP)
        exp = dcs.convert_monthly_to_annual_probability(0.5)
        self.assertEqual(new, exp)
    def test_prob_rate_conversions1c(self):
        new = dcs.Units.convert(0.5, dcs.Units.MP, dcs.Units.AR)
        exp = dcs.prob_to_rate(dcs.convert_monthly_to_annual_probability(0.5))
        self.assertEqual(new, exp)
    def test_prob_rate_conversions2a(self):
        new = dcs.Units.convert(0.5, dcs.Units.AP, dcs.Units.MP)
        exp = dcs.convert_annual_to_monthly_probability(0.5)
        self.assertEqual(new, exp)
    def test_prob_rate_conversions2b(self):
        new = dcs.Units.convert(0.5, dcs.Units.AP, dcs.Units.AP)
        exp = 0.5
        self.assertEqual(new, exp)
    def test_prob_rate_conversions2c(self):
        new = dcs.Units.convert(0.5, dcs.Units.AP, dcs.Units.AR)
        exp = dcs.prob_to_rate(0.5)
        self.assertEqual(new, exp)
    def test_prob_rate_conversions3a(self):
        new = dcs.Units.convert(0.5, dcs.Units.AR, dcs.Units.MP)
        exp = dcs.rate_to_prob(0.5/12)
        self.assertEqual(new, exp)
    def test_prob_rate_conversions3b(self):
        new = dcs.Units.convert(0.5, dcs.Units.AR, dcs.Units.AP)
        exp = dcs.rate_to_prob(0.5)
        self.assertEqual(new, exp)
    def test_prob_rate_conversions3c(self):
        new = dcs.Units.convert(0.5, dcs.Units.AR, dcs.Units.AR)
        exp = 0.5
        self.assertEqual(new, exp)
    def test_mult1(self):
        self.assertEqual(dcs.Units.convert(1., dcs.Units.ND, dcs.Units.ND), 1.)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.ND, dcs.Units.PER), 100.)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.ND, dcs.Units.P100K), 100000.)
    def test_mult2(self):
        self.assertEqual(dcs.Units.convert(1., dcs.Units.PER, dcs.Units.ND), 0.01)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.PER, dcs.Units.PER), 1.)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.PER, dcs.Units.P100K), 1000.)
    def test_mult3(self):
        self.assertEqual(dcs.Units.convert(1., dcs.Units.P100K, dcs.Units.ND), 0.00001)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.P100K, dcs.Units.PER), 0.001)
        self.assertEqual(dcs.Units.convert(1., dcs.Units.P100K, dcs.Units.P100K), 1.)
    def test_prob_to_bad(self):
        with self.assertRaises(ValueError):
            dcs.Units.convert(1, dcs.Units.MP, dcs.Units.P100K)
    def test_bad_units1(self):
        with self.assertRaises(KeyError):
            dcs.Units.convert(1, dcs.Units.ND, 100)
    def test_bad_units2(self):
        with self.assertRaises(KeyError):
            dcs.Units.convert(1, 100, dcs.Units.PER)

#%% Param
pass

#%% get_factors
class Test_get_factors(unittest.TestCase):
    r"""
    Tests the modd function with the following cases:
        Nominal
        Bad prefix
    """
    def setUp(self):
        self.prefix = ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca',\
            'unity', 'deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto']
        self.mult   = [1e24, 1e21, 1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e2, 1e1, 1e0, \
            1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
        self.label  = ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da', '', \
            'd', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y']

    def test_nominal(self):
        for i in range(len(self.prefix)):
            (mult, label) = dcs.get_factors(self.prefix[i])
            self.assertEqual(mult, self.mult[i])
            self.assertEqual(label, self.label[i])

    def test_bad_prefix(self):
        with self.assertRaises(ValueError):
            dcs.get_factors('bad_prefix_name')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
