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
from unittest.mock import patch

import dstauffman as dcs
import dstauffman.commands as commands

#%% commands.parse_tests
class Test_parse_tests(unittest.TestCase):
    r"""
    Tests the parse_tests function with the following cases:
        Nominal
    """
    def setUp(self):
        self.folder              = dcs.get_root_dir()
        self.expected            = argparse.Namespace()
        self.expected.docstrings = False
        self.expected.library    = None
        self.expected.verbose    = False

    def test_nominal(self):
        args = commands.parse_tests([])
        self.assertEqual(args, self.expected)

    def test_docstrings(self):
        self.expected.docstrings = True
        args = commands.parse_tests(['-d'])
        self.assertEqual(args, self.expected)

    def test_verbose(self):
        self.expected.verbose = True
        args = commands.parse_tests(['-v'])
        self.assertEqual(args, self.expected)

    def test_library(self):
        self.expected.library = 'other'
        args = commands.parse_tests(['-l', 'other'])
        self.assertEqual(args, self.expected)

#%% commands.execute_tests
class Test_execute_tests(unittest.TestCase):
    r"""
    Tests the execute_tests function with the following cases:
        Nominal
        TBD
    """
    def setUp(self):
        self.folder = dcs.get_root_dir()
        self.args = argparse.Namespace(docstrings=False, library=None, verbose=False)
        self.patch_args = {'folder': self.folder, 'extensions': frozenset({'m', 'py'}), 'list_all': False, \
                           'check_tabs': True, 'trailing': False, 'exclusions': None, 'check_eol': None, \
                           'show_execute': False}

    @patch('dstauffman.commands.runtests.run_pytests')
    def test_nominal(self, mocker):
        commands.execute_tests(self.args)
        mocker.assert_called_once_with(self.folder)

    @patch('dstauffman.commands.runtests.run_pytests')
    def test_verbose(self, mocker):
        self.args.verbose = True
        commands.execute_tests(self.args)
        # Note: this doesn't add anything with pytest
        mocker.assert_called_once_with(self.folder)

    @patch('dstauffman.commands.runtests.run_docstrings')
    def test_docstrings(self, mocker):
        self.args.docstrings=True
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertFalse(kwargs['verbose'])
        self.assertTrue(len(pos_args) > 0)

    @patch('dstauffman.commands.runtests.run_docstrings')
    def test_docstrings_verbose(self, mocker):
        self.args.docstrings = True
        self.args.verbose = True
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertTrue(kwargs['verbose'])
        self.assertTrue(len(pos_args) > 0)

    @patch('dstauffman.commands.runtests.run_pytests')
    def test_library(self, mocker):
        self.args.library = 'other_folder'
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertEqual(len(pos_args), 1)
        self.assertTrue(pos_args[0].endswith('other_folder'))

#%% commands.parse_coverage
pass # TODO: write this

#%% commands.execute_coverage
pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
