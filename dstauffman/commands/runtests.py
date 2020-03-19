# -*- coding: utf-8 -*-
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
import unittest

from dstauffman.paths import get_root_dir, list_python_files
from dstauffman.repos import run_docstrings, run_pytests

#%% Functions - parse_tests
def parse_tests(input_args):
    r"""
    Parser for the tests commands.

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
    Namespace(docstrings=False, verbose=False)

    """
    parser = argparse.ArgumentParser(prog='redy tests', description='Runs all the built-in unit tests.')

    parser.add_argument('-v', '--verbose', help='Run tests in verbose mode.', action='store_true')

    parser.add_argument('-d', '--docstrings', help='Run the docstrings instead of the unittests.', action='store_true')

    args = parser.parse_args(input_args)
    return args

#%% Functions - execute_tests
def execute_tests(args):
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
    >>> args = Namespace(docstrings=False, verbose=False)
    >>> execute_tests(args) # doctest: +SKIP

    """
    # alias options
    docstrings = args.docstrings
    verbose    = args.verbose

    # get test location information
    folder     = get_root_dir()

    if docstrings:
        # run the docstring tests
        files = list_python_files(folder)
        files.extend(list_python_files(os.path.join(folder, 'commands')))
        return_code = run_docstrings(files, verbose=verbose)
    else:
        # run the unittests using pytest
        return_code = run_pytests(folder)
    return return_code

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_commands_tests', exit=False)
    doctest.testmod(verbose=False)
