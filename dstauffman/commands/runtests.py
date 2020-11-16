r"""
Functions related to running all the included tests.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import doctest
import os
import platform
import subprocess
import sys
from typing import List
import unittest

from dstauffman import get_root_dir, get_tests_dir, list_python_files, run_coverage, \
    run_docstrings, run_pytests, run_unittests

#%% Functions - parse_tests
def parse_tests(input_args: List[str]) -> argparse.Namespace:
    r"""
    Parser for the tests command.

    Parameters
    ----------
    input_args : list of str
        Input arguments as passed to sys.argv for this command

    Returns
    -------
    args : class Namespace
        Arguments as parsed by argparse.parse_args

    Notes
    -----
    #.  Written by David C. Stauffer in November 2019.

    Examples
    --------
    >>> from dstauffman.commands import parse_tests
    >>> input_args = []
    >>> args = parse_tests(input_args)
    >>> print(args)
    Namespace(docstrings=False, unittest=False, verbose=False, library=None)

    """
    parser = argparse.ArgumentParser(prog='dcs tests', description='Runs all the built-in unit tests.')

    parser.add_argument('-d', '--docstrings', help='Run the docstrings instead of the unittests.', action='store_true')

    parser.add_argument('-u', '--unittest', help='Use unittest instead of pytest for the test runner.', action='store_true')

    parser.add_argument('-v', '--verbose', help='Run tests in verbose mode.', action='store_true')

    parser.add_argument('-l', '--library', type=str, nargs='?', help='Library to run the unit tests from, default is yourself.')

    args = parser.parse_args(input_args)
    return args

#%% Functions - execute_tests
def execute_tests(args: argparse.Namespace) -> int:
    r"""
    Executes the tests commands.

    Parameters
    ----------
    args : class argparse.Namespace, with fields:
        .docstrings : bool
        .verbose : bool

    Returns
    -------
    return_code : int
        Return code for whether the command executed cleanly

    Notes
    -----
    #.  Written by David C. Stauffer in November 2019.

    Examples
    --------
    >>> from dstauffman.commands import execute_tests
    >>> from argparse import Namespace
    >>> args = Namespace(docstrings=False, library=None, unittest=False, verbose=False)
    >>> execute_tests(args) # doctest: +SKIP

    """
    # alias options
    docstrings = args.docstrings
    library    = args.library
    verbose    = args.verbose
    use_pytest = not args.unittest

    # get test location information
    if library is None:
        folder = get_root_dir()
    else:
        folder = os.path.abspath(library)
        if folder not in sys.path:
            # Note: pytest seems to work without this step?
            sys.path.append(folder)

    if docstrings:
        # run the docstring tests
        files = list_python_files(folder, recursive=True)
        return_code = run_docstrings(files, verbose=verbose)
    else:
        if use_pytest:
            # run the unittests using pytest
            return_code = run_pytests(folder)
        else:
            # run the unittests using unittest (which is core python)
            test_names = library if library is not None else 'dstauffman.tests'
            return_code = run_unittests(test_names)
    return return_code

#%% Functions - parse_coverage
def parse_coverage(input_args: List[str]) -> argparse.Namespace:
    r"""
    Parser for the coverage command.

    Parameters
    ----------
    input_args : list of str
        Input arguments as passed to sys.argv for this command

    Returns
    -------
    args : class Namespace
        Arguments as parsed by argparse.parse_args

    Notes
    -----
    #.  Written by David C. Stauffer in March 2020.

    Examples
    --------
    >>> from dstauffman.commands import parse_coverage
    >>> input_args = []
    >>> args = parse_coverage(input_args)
    >>> print(args)
    Namespace(no_report=False)

    """
    parser = argparse.ArgumentParser(prog='dcs coverage', description='Runs all the built-in unit tests and produces a coverage report.')

    parser.add_argument('-n', '--no-report', help='Suppresses the generation of the HTML report.', action='store_true')

    args = parser.parse_args(input_args)
    return args

#%% Functions - execute_coverage
def execute_coverage(args: argparse.Namespace) -> int:
    r"""
    Executes the coverage commands.

    Parameters
    ----------
    args : class argparse.Namespace, with fields:
        .no_report : bool

    Returns
    -------
    return_code : int
        Return code for whether the command executed cleanly

    Notes
    -----
    #.  Written by David C. Stauffer in March 2020.

    Examples
    --------
    >>> from dstauffman.commands import execute_coverage
    >>> from argparse import Namespace
    >>> args = Namespace(no_report=False)
    >>> execute_coverage(args) # doctest: +SKIP

    """
    #alias options
    report = not args.no_report

    # get test location information
    folder = get_root_dir()

    # run coverage
    return_code = run_coverage(folder, report=report)

    # open the report
    if report:
        filename = os.path.join(get_tests_dir(), 'coverage_html_report', 'index.html')
        if platform.system() == 'Darwin':
            subprocess.call(['open', filename])
        elif platform.system() == 'Windows':
            os.startfile(filename)
        else:
            subprocess.call(['xdg-open', filename])
    return return_code

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_commands_runtests', exit=False)
    doctest.testmod(verbose=False)
