# -*- coding: utf-8 -*-
r"""
Test file for the `commands.runtests` module of the "dstauffman" library.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import unittest

import dstauffman as dcs
import dstauffman.commands as commands

#%% commands.parse_tests
class Test_parse_tests(unittest.TestCase):
    r"""
    Tests the parse_tests function with the following cases:
        Nominal
    """
    def setUp(self):
        self.folder               = dcs.get_root_dir()
        self.expected             = argparse.Namespace()
        self.expected.verbose     = False
        self.expected.docstrings  = False

    def test_nominal(self):
        args = commands.parse_tests([])
        self.assertEqual(args, self.expected)

    def test_verbose(self):
        self.expected.verbose = True
        args = commands.parse_tests(['-v'])
        self.assertEqual(args, self.expected)

    def test_docstrings(self):
        self.expected.docstrings = True
        args = commands.parse_tests(['-d'])
        self.assertEqual(args, self.expected)

#%% commands.execute_tests
pass #TODO: write thisS

#%% commands.parse_coverage
pass #TODO: write thisS

#%% commands.execute_coverage
pass #TODO: write thisS

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
