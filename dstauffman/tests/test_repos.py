r"""
Test file for the `repos` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2019.
"""

# %% Imports
from __future__ import annotations

import unittest
from unittest.mock import patch

from slog import capture_output, ReturnCodes

import dstauffman as dcs


# %% run_docstrings
class Test_run_docstrings(unittest.TestCase):
    r"""
    Tests the run_docstrings function with the following cases:
        No Failures
        With Failures
    """

    def test_no_failures(self) -> None:
        files = [dcs.get_tests_dir() / x for x in ("test_matlab.py", "test_repos.py")]
        verbose = False
        with patch("dstauffman.repos.doctest.testfile", return_value=(0, 0)) as mock_tester:
            return_code = dcs.run_docstrings(files, verbose)
        self.assertEqual(return_code, ReturnCodes.clean)
        mock_tester.assert_any_call(files[0], report=True, verbose=verbose, module_relative=False)
        mock_tester.assert_called_with(files[1], report=True, verbose=verbose, module_relative=False)

    def test_with_failures(self) -> None:
        files = [dcs.get_tests_dir() / x for x in ("test_matlab.py", "test_repos.py")]
        verbose = True
        with patch("dstauffman.repos.doctest.testfile", return_value=(1, 0)) as mock_tester:
            with capture_output() as ctx:
                return_code = dcs.run_docstrings(files, verbose)
        lines = ctx.get_output().split("\n")
        self.assertEqual(return_code, ReturnCodes.test_failures)
        exp = f"Testing \"{dcs.get_tests_dir().joinpath('test_matlab.py')}\":"
        self.assertIn(exp, lines)
        exp = f"Testing \"{dcs.get_tests_dir().joinpath('test_repos.py')}\":"
        self.assertIn(exp, lines)
        mock_tester.assert_any_call(files[0], report=True, verbose=verbose, module_relative=False)
        mock_tester.assert_called_with(files[1], report=True, verbose=verbose, module_relative=False)


# %% run_unittests
class Test_run_unittests(unittest.TestCase):
    r"""
    Tests the run_unittests function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% run_pytests
class Test_run_pytests(unittest.TestCase):
    r"""
    Tests the run_pytests function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% run_coverage
class Test_run_coverage(unittest.TestCase):
    r"""
    Tests the run_coverage function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
