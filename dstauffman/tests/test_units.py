r"""
Test file for the `units` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in February 2016.
"""

#%% Imports
import unittest

import dstauffman as dcs

#%% Unit Conversions
class Test_Constants(unittest.TestCase):
    r"""
    Tests the UnitConversions class with the following methods:
        Nominal definitions
    """
    def setUp(self):
        self.ints = ['ONE_MINUTE', 'ONE_HOUR', 'ONE_DAY', 'MONTHS_PER_YEAR']
        self.flts = ['RAD2DEG', 'DEG2RAD', 'ARCSEC2RAD', 'RAD2ARCSEC', 'FT2M', 'M2FT', 'IN2CM', 'CM2IN']
        self.master = set(self.ints) | set(self.flts)

    def test_values(self):
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.flts:
            self.assertTrue(isinstance(getattr(dcs, key), float))

    def test_pairs(self):
        self.assertEqual(60 * 60 * 24, dcs.ONE_DAY)
        self.assertAlmostEqual(dcs.DEG2RAD * dcs.RAD2DEG, 1, 14)
        self.assertAlmostEqual(dcs.ARCSEC2RAD * dcs.RAD2ARCSEC, 1, 14)
        self.assertAlmostEqual(dcs.FT2M * dcs.M2FT, 1, 14)
        self.assertAlmostEqual(dcs.IN2CM * dcs.CM2IN, 1, 14)

    def test_missing(self):
        for field in vars(dcs.units):
            if field.isupper():
                self.assertTrue(field in self.master, 'Test is missing: {}'.format(field))

#%% get_factors
class Test_get_factors(unittest.TestCase):
    r"""
    Tests the get_factors function with the following cases:
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

    def test_int(self):
        (mult, label) = dcs.get_factors(10)
        self.assertEqual(mult, 10)
        self.assertEqual(label, '')

    def test_float(self):
        (mult, label) = dcs.get_factors(0.3)
        self.assertEqual(mult, 0.3)
        self.assertEqual(label, '')

#%% get_time_factor
class Test_get_time_factor(unittest.TestCase):
    r"""
    Tests the get_time_factor function with the following cases:
        TBD
    """
    def setUp(self):
        self.units = ['sec', 'min', 'hr', 'day']
        self.mults = [1, 60, 3600, 86400]

    def test_nominal(self):
        for (unit, mult) in zip(self.units, self.mults):
            out = dcs.get_time_factor(unit)
            self.assertEqual(out, mult)

    def test_bad(self):
        with self.assertRaises(ValueError):
            dcs.get_time_factor('bad')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
