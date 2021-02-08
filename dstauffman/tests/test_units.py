r"""
Test file for the `units` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in February 2016.
"""

#%% Imports
from typing import List
import unittest

import dstauffman as dcs

#%% Unit Conversions
class Test_Constants(unittest.TestCase):
    r"""
    Tests the UnitConversions class with the following methods:
        Nominal definitions
    """
    def setUp(self) -> None:
        self.ints: List[str] = ['ONE_MINUTE', 'ONE_HOUR', 'ONE_DAY', 'MONTHS_PER_YEAR']
        self.flts: List[str] = ['RAD2DEG', 'DEG2RAD', 'ARCSEC2RAD', 'RAD2ARCSEC', 'FT2M', 'M2FT', 'IN2CM', 'CM2IN']
        self.strs: List[str] = ['DEGREE_SIGN', 'MICRO_SIGN']
        self.master = set(self.ints) | set(self.flts) | set(self.strs)

    def test_values(self) -> None:
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.flts:
            self.assertTrue(isinstance(getattr(dcs, key), float))
        for key in self.strs:
            self.assertTrue(isinstance(getattr(dcs, key), str))

    def test_pairs(self) -> None:
        self.assertEqual(60 * 60 * 24, dcs.ONE_DAY)
        self.assertAlmostEqual(dcs.DEG2RAD * dcs.RAD2DEG, 1, 14)
        self.assertAlmostEqual(dcs.ARCSEC2RAD * dcs.RAD2ARCSEC, 1, 14)
        self.assertAlmostEqual(dcs.FT2M * dcs.M2FT, 1, 14)
        self.assertAlmostEqual(dcs.IN2CM * dcs.CM2IN, 1, 14)

    def test_missing(self) -> None:
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
    def setUp(self) -> None:
        self.prefix = ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca',\
            'unity', 'deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto']
        self.mult   = [1e24, 1e21, 1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e2, 1e1, 1e0, \
            1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
        self.label  = ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da', '', \
            'd', 'c', 'm', 'µ', 'n', 'p', 'f', 'a', 'z', 'y']

    def test_nominal(self) -> None:
        for i in range(len(self.prefix)):
            (mult, label) = dcs.get_factors(self.prefix[i])
            self.assertEqual(mult, self.mult[i])
            self.assertEqual(label, self.label[i])

    def test_bad_prefix(self) -> None:
        with self.assertRaises(ValueError):
            dcs.get_factors('bad_prefix_name')

    def test_int(self) -> None:
        (mult, label) = dcs.get_factors(10)
        self.assertEqual(mult, 10)
        self.assertEqual(label, '')

    def test_float(self) -> None:
        (mult, label) = dcs.get_factors(0.3)
        self.assertEqual(mult, 0.3)
        self.assertEqual(label, '')

    def test_english_units(self) -> None:
        names  = ['arcminute', 'arcsecond', 'milliarcsecond', 'microarcsecond']
        mults  = [dcs.DEG2RAD/60, dcs.DEG2RAD/3600, dcs.DEG2RAD/3.6, 1e3*dcs.DEG2RAD/3.6]
        labels = ['amin', 'asec', 'mas', 'µas']
        for (fact, exp_mult, exp_label) in zip(names, mults, labels):
            (mult, label) = dcs.get_factors(fact)
            self.assertAlmostEqual(mult, exp_mult, 14, 'Bad multiplication factor for {}'.format(fact))
            self.assertEqual(label, exp_label,'Bad label for {}'.format(fact))

#%% get_time_factor
class Test_get_time_factor(unittest.TestCase):
    r"""
    Tests the get_time_factor function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.units = ['sec', 'min', 'hr', 'day']
        self.mults = [1, 60, 3600, 86400]

    def test_nominal(self) -> None:
        for (unit, mult) in zip(self.units, self.mults):
            out = dcs.get_time_factor(unit)
            self.assertEqual(out, mult)

    def test_bad(self) -> None:
        with self.assertRaises(ValueError):
            dcs.get_time_factor('bad')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
