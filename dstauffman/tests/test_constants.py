# -*- coding: utf-8 -*-
r"""
Test file for the `constants` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import unittest
import dstauffman as dcs

#%% Classes for testing
class Test_all_values(unittest.TestCase):

    def setUp(self):
        self.ints = ['MONTHS_PER_YEAR', 'INT_TOKEN', 'QUAT_SIZE']
        self.strs = ['DEFAULT_COLORMAP']
        self.master = set(self.ints) | set(self.strs)

    def test_values(self):
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.strs:
            self.assertTrue(isinstance(getattr(dcs, key), str))

    def test_missing(self):
        for field in vars(dcs.constants):
            if field.isupper():
                self.assertTrue(field in self.ints or field in self.strs, 'Test is missing: {}'.format(field))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)