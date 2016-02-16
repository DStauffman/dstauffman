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
pass

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
