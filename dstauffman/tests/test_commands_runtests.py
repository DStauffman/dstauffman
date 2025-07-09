r"""
Test file for the `runtests` module of the "dstauffman.commands" library.

Notes
-----
#.  Written by David C. Stauffer in March 2020.

"""

# %% Imports
import argparse
import unittest
from unittest.mock import Mock, patch

import dstauffman as dcs
import dstauffman.commands as commands


# %% commands.parse_tests
class Test_commands_parse_tests(unittest.TestCase):
    r"""
    Tests the commands.parse_tests function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        # fmt: off
        self.folder              = dcs.get_root_dir()
        self.expected            = argparse.Namespace()
        self.expected.docstrings = False
        self.expected.unittest   = False
        self.expected.verbose    = False
        self.expected.library    = None
        self.expected.coverage   = False
        self.expected.cov_file   = None
        # fmt: on

    def test_nominal(self) -> None:
        args = commands.parse_tests([])
        self.assertEqual(args, self.expected)

    def test_docstrings(self) -> None:
        self.expected.docstrings = True
        args = commands.parse_tests(["-d"])
        self.assertEqual(args, self.expected)

    def test_verbose(self) -> None:
        self.expected.verbose = True
        args = commands.parse_tests(["-v"])
        self.assertEqual(args, self.expected)

    def test_library(self) -> None:
        self.expected.library = "other"
        args = commands.parse_tests(["-l", "other"])
        self.assertEqual(args, self.expected)

    def test_unittest(self) -> None:
        self.expected.unittest = True
        args = commands.parse_tests(["-u"])
        self.assertEqual(args, self.expected)

    def test_coverage(self) -> None:
        self.expected.coverage = True
        args = commands.parse_tests(["-c"])
        self.assertEqual(args, self.expected)


# %% commands.execute_tests
class Test_commands_execute_tests(unittest.TestCase):
    r"""
    Tests the commands.execute_tests function with the following cases:
        Nominal
        TBD
    """

    def setUp(self) -> None:
        self.folder = dcs.get_root_dir()
        self.args = argparse.Namespace(
            docstrings=False, unittest=False, verbose=False, library=None, coverage=False, cov_file=None
        )
        self.patch_args = {
            "folder": self.folder,
            "extensions": frozenset({"m", "py"}),
            "list_all": False,
            "check_tabs": True,
            "trailing": False,
            "exclusions": None,
            "check_eol": None,
            "show_execute": False,
        }

    @patch("dstauffman.commands.runtests.run_pytests")
    def test_nominal(self, mocker: Mock) -> None:
        commands.execute_tests(self.args)
        mocker.assert_called_once_with(self.folder)

    @patch("dstauffman.commands.runtests.run_pytests")
    def test_verbose(self, mocker: Mock) -> None:
        self.args.verbose = True
        commands.execute_tests(self.args)
        # Note: this doesn't add anything with pytest
        mocker.assert_called_once_with(self.folder)

    @patch("dstauffman.commands.runtests.run_docstrings")
    def test_docstrings(self, mocker: Mock) -> None:
        self.args.docstrings = True
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertFalse(kwargs["verbose"])
        self.assertTrue(len(pos_args) > 0)

    @patch("dstauffman.commands.runtests.run_docstrings")
    def test_docstrings_verbose(self, mocker: Mock) -> None:
        self.args.docstrings = True
        self.args.verbose = True
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertTrue(kwargs["verbose"])
        self.assertTrue(len(pos_args) > 0)

    @patch("dstauffman.commands.runtests.run_pytests")
    def test_library(self, mocker: Mock) -> None:
        self.args.library = "other_folder"
        commands.execute_tests(self.args)
        (pos_args, kwargs) = mocker.call_args
        self.assertEqual(len(pos_args), 1)
        self.assertTrue(str(pos_args[0]).endswith("other_folder"))
        self.assertEqual(kwargs, {})

    @patch("dstauffman.commands.runtests.run_coverage")
    def test_coverage(self, mocker: Mock) -> None:
        self.args.coverage = True
        commands.execute_tests(self.args)
        mocker.assert_called_once_with(self.folder, cov_file=None, report=False)


# %% commands.parse_coverage
class Test_commands_parse_coverage(unittest.TestCase):
    r"""
    Tests the commands.parse_coverage function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% commands.execute_coverage
class Test_commands_execute_coverage(unittest.TestCase):
    r"""
    Tests the commands.execute_coverage function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
