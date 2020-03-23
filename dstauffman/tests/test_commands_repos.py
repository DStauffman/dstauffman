# -*- coding: utf-8 -*-
r"""
Test file for the `commands.repos` module of the "dstauffman" library.  It is intented to contain
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

#%% commands.parse_enforce
class Test_parse_enforce(unittest.TestCase):
    r"""
    Tests the parse_enforce function with the following cases:
        Nominal
    """
    def setUp(self):
        self.folder               = dcs.get_root_dir()
        self.expected             = argparse.Namespace()
        self.expected.execute     = False
        self.expected.extensions  = None
        self.expected.folder      = self.folder
        self.expected.ignore_tabs = False
        self.expected.list_all    = False
        self.expected.skip        = None
        self.expected.trailing    = False
        self.expected.unix        = False
        self.expected.windows     = False

    def test_nominal(self):
        args = commands.parse_enforce([self.folder])
        self.assertEqual(args, self.expected)

    def test_list_all(self):
        self.expected.list_all = True
        args = commands.parse_enforce([self.folder, '-l'])
        self.assertEqual(args, self.expected)

#%% commands.execute_enforce
@patch('dstauffman.commands.repos.find_repo_issues')
class Test_execute_enfore(unittest.TestCase):
    r"""
    Tests the execute_enforce function with the following cases:
        Nominal
        TBD
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()
        self.args = argparse.Namespace(execute=False, extensions=None, folder=self.folder, ignore_tabs=False, \
                                       list_all=False, skip=None, trailing=False, unix=False, windows=False)
        self.patch_args = {'folder': self.folder, 'extensions': frozenset({'m', 'py'}), 'list_all': False, \
                           'check_tabs': True, 'trailing': False, 'exclusions': None, 'check_eol': None, \
                           'show_execute': False}

    def test_nominal(self, mocker):
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_windows(self, mocker):
        self.args.windows = True
        self.patch_args['check_eol'] = '\r\n'
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_unix(self, mocker):
        self.args.unix = True
        self.patch_args['check_eol'] = '\n'
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_all_extensions(self, mocker):
        self.args.extensions = '*'
        self.patch_args['extensions'] = None
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_extensions(self, mocker):
        self.args.extensions = ['f90']
        self.patch_args['extensions'] = ['f90']
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
