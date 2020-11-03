r"""
Test file for the `constants` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from typing import List
import unittest

import dstauffman as dcs

#%% Classes for testing
class Test_all_values(unittest.TestCase):

    def setUp(self) -> None:
        self.ints: List[str] = ['INT_TOKEN', 'NP_INT64_PER_SEC']
        self.strs: List[str] = ['NP_DATETIME_FORM', 'NP_DATETIME_UNITS', 'NP_TIMEDELTA_FORM']
        self.bool: List[str] = ['HAVE_H5PY', 'HAVE_NUMPY', 'IS_WINDOWS']
        self.master = set(self.ints) | set(self.strs) | set(self.bool)

    def test_values(self) -> None:
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.strs:
            self.assertTrue(isinstance(getattr(dcs, key), str))
        for key in self.bool:
            self.assertTrue(isinstance(getattr(dcs, key), bool))

    def test_missing(self) -> None:
        for field in vars(dcs.constants):
            if field.isupper():
                self.assertTrue(field in self.master, 'Test is missing: {}'.format(field))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
