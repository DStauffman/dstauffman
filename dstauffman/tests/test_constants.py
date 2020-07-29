r"""
Test file for the `constants` module of the "dstauffman" library.

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
        self.ints = ['INT_TOKEN']
        self.strs = []
        self.bool = ['IS_WINDOWS']
        self.master = set(self.ints) | set(self.strs) | set(self.bool)

    def test_values(self):
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.strs:
            self.assertTrue(isinstance(getattr(dcs, key), str))
        for key in self.bool:
            self.assertTrue(isinstance(getattr(dcs, key), bool))

    def test_missing(self):
        for field in vars(dcs.constants):
            if field.isupper():
                self.assertTrue(field in self.master, 'Test is missing: {}'.format(field))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
