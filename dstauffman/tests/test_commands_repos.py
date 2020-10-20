r"""
Test file for the `repos` module of the "dstauffman.commands" library.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import os
import unittest
from unittest.mock import Mock, patch

import dstauffman as dcs
import dstauffman.commands as commands

#%% commands.parse_enforce
class Test_commands_parse_enforce(unittest.TestCase):
    r"""
    Tests the commands.parse_enforce function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
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

    def test_nominal(self) -> None:
        args = commands.parse_enforce([self.folder])
        self.assertEqual(args, self.expected)

    def test_extensions(self) -> None:
        self.expected.extensions = ['f', 'f90']
        args = commands.parse_enforce([self.folder, '-e', 'f', '-e', 'f90'])
        self.assertEqual(args, self.expected)

    def test_list_all(self) -> None:
        self.expected.list_all = True
        args = commands.parse_enforce([self.folder, '-l'])
        self.assertEqual(args, self.expected)

    def test_ignore_tabs(self) -> None:
        self.expected.ignore_tabs = True
        args = commands.parse_enforce([self.folder, '-i'])
        self.assertEqual(args, self.expected)

    def test_trailing(self) -> None:
        self.expected.trailing = True
        args = commands.parse_enforce([self.folder, '-t'])
        self.assertEqual(args, self.expected)

    def test_skip(self) -> None:
        self.expected.skip = ['m']
        args = commands.parse_enforce([self.folder, '-s', 'm'])
        self.assertEqual(args, self.expected)

    def test_windows(self) -> None:
        self.expected.windows = True
        self.expected.unix    = False
        args = commands.parse_enforce([self.folder, '-w'])
        self.assertEqual(args, self.expected)

    def test_unix(self) -> None:
        self.expected.windows = False
        self.expected.unix    = True
        args = commands.parse_enforce([self.folder, '-u'])
        self.assertEqual(args, self.expected)

    def test_bad_os_combination(self) -> None:
        with dcs.capture_output('err') as err:
            with self.assertRaises(SystemExit):
                commands.parse_enforce([self.folder, '-w', '-u'])
        stderr = err.getvalue().strip()
        err.close()
        self.assertTrue(stderr.startswith('usage: dcs enforce'))

    def test_execute(self) -> None:
        self.expected.execute = True
        args = commands.parse_enforce([self.folder, '-x'])
        self.assertEqual(args, self.expected)

#%% commands.execute_enforce
@patch('dstauffman.commands.repos.find_repo_issues')
class Test_commands_execute_enforce(unittest.TestCase):
    r"""
    Tests the commands.execute_enforce function with the following cases:
        Nominal
        TBD
    """
    def setUp(self) -> None:
        self.folder = dcs.get_tests_dir()
        self.args = argparse.Namespace(execute=False, extensions=None, folder=self.folder, ignore_tabs=False, \
                                       list_all=False, skip=None, trailing=False, unix=False, windows=False)
        self.patch_args = {'folder': self.folder, 'extensions': frozenset({'m', 'py'}), 'list_all': False, \
                           'check_tabs': True, 'trailing': False, 'exclusions': None, 'check_eol': None, \
                           'show_execute': False}

    def test_nominal(self, mocker: Mock) -> None:
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_windows(self, mocker: Mock) -> None:
        self.args.windows = True
        self.patch_args['check_eol'] = '\r\n'
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_unix(self, mocker: Mock) -> None:
        self.args.unix = True
        self.patch_args['check_eol'] = '\n'
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_all_extensions(self, mocker: Mock) -> None:
        self.args.extensions = '*'
        self.patch_args['extensions'] = None
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

    def test_extensions(self, mocker: Mock) -> None:
        self.args.extensions = ['f90']
        self.patch_args['extensions'] = ['f90']
        commands.execute_enforce(self.args)
        mocker.assert_called_once_with(**self.patch_args)

#%% commands.parse_make_init
class Test_commands_parse_make_init(unittest.TestCase):
    r"""
    Tests the commands.parse_make_init function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.folder           = dcs.get_root_dir()
        self.expected         = argparse.Namespace()
        self.expected.dry_run = False
        self.expected.folder  = self.folder
        self.expected.lineup  = False
        self.expected.outfile = '__init__.py'
        self.expected.wrap    = 100

    def test_nominal(self) -> None:
        args = commands.parse_make_init([self.folder])
        self.assertEqual(args, self.expected)

    def test_dry_num(self) -> None:
        self.expected.dry_run = True
        args = commands.parse_make_init([self.folder, '-n'])
        self.assertEqual(args, self.expected)

    def test_lineup(self) -> None:
        self.expected.lineup = True
        args = commands.parse_make_init([self.folder, '-l'])
        self.assertEqual(args, self.expected)

    def test_outfile(self) -> None:
        self.expected.outfile = 'init_file.py'
        args = commands.parse_make_init([self.folder, '-o', 'init_file.py'])
        self.assertEqual(args, self.expected)

    def test_wrap(self) -> None:
        self.expected.wrap = 50
        args = commands.parse_make_init([self.folder, '-w', '50'])
        self.assertEqual(args, self.expected)

#%% commands.execute_make_init
@patch('dstauffman.commands.repos.make_python_init')
class Test_commands_execute_make_init(unittest.TestCase):
    r"""
    Tests the commands.execute_make_init function with the following cases:
        Nominal
        TBD
    """
    def setUp(self) -> None:
        self.folder = dcs.get_tests_dir()
        self.init_file = os.path.join(self.folder, 'temp_init.py')
        self.args = argparse.Namespace(dry_run=False, folder=self.folder, lineup=False, outfile=self.init_file, wrap=100)
        self.patch_args = {'lineup': False, 'wrap': 100, 'filename': os.path.join(self.folder, 'temp_init.py')}

    def test_nominal(self, mocker: Mock) -> None:
        commands.execute_make_init(self.args)
        mocker.assert_called_once_with(self.folder, **self.patch_args)

    def test_dry_num(self, mocker: Mock) -> None:
        self.args.dry_run = True
        with dcs.capture_output() as out:
            commands.execute_make_init(self.args)
        output = out.getvalue().strip()
        out.close()
        mocker.assert_not_called()
        self.assertTrue(output.startswith('Would execute "make_python_init('))

    def test_lineup(self, mocker: Mock) -> None:
        self.args.lineup = True
        self.patch_args['lineup'] = True
        commands.execute_make_init(self.args)
        mocker.assert_called_once_with(self.folder, **self.patch_args)

    def test_outfile(self, mocker: Mock) -> None:
        self.args.outfile = os.path.join(self.folder, 'init_file.py')
        self.patch_args['filename'] = self.args.outfile
        commands.execute_make_init(self.args)
        mocker.assert_called_once_with(self.folder, **self.patch_args)

    def test_wrap(self, mocker: Mock) -> None:
        self.args.wrap = 500
        self.patch_args['wrap'] = 500
        commands.execute_make_init(self.args)
        mocker.assert_called_once_with(self.folder, **self.patch_args)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
