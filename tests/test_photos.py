# -*- coding: utf-8 -*-
r"""
Test file for the `photos` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import unittest
import dstauffman as dcs

#%% Classes for testing
# find_missing_nums
class Test_find_missing_nums(unittest.TestCase):
    r"""
    Tests the find_missing_nums function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.find_missing_nums(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('No number found: "'))

# find_unexpected_ext
class Test_find_unexpected_ext(unittest.TestCase):
    r"""
    Tests the find_unexpected_ext function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.find_unexpected_ext(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Finding any unexpected file extensions...\n Unexpected: "'))
        self.assertTrue(output.endswith('"\nDone.'))

# rename_old_picasa_files
class Test_rename_old_picasa_files(unittest.TestCase):
    r"""
    Tests the rename_old_picasa_files function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.rename_old_picasa_files(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')

# rename_upper_ext
class Test_rename_upper_ext(unittest.TestCase):
    r"""
    Tests the rename_upper_ext function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.rename_upper_ext(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Searching for file extensions to rename...'))
        self.assertTrue(output.endswith('\nDone.'))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
