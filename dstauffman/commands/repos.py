# -*- coding: utf-8 -*-
r"""
Functions related to `repos` command.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import doctest
import unittest

from dstauffman.enums import ReturnCodes
from dstauffman.repos import find_repo_issues

#%% Functions - parse_enforce
def parse_enforce(input_args):
    r"""
    Parser for enforce command.

    Parameters
    ----------
    input_args : list of str
        Input arguments as passed to sys.argv for this command

    Returns
    -------
    args : class Namespace
        Arguments as parsed by argparse.parse_args

    Examples
    --------
    >>> from dstauffman.commands import parse_enforce
    >>> input_args = []
    >>> args = parse_enforce(input_args)
    >>> print(args)
    Namespace()

    """
    parser = argparse.ArgumentParser(prog='lms enforce')

    args = parser.parse_args(input_args)
    return args

#%% Functions - execute_enforce
def execute_enforce(args):
    r"""
    Executes the enforce command.

    Parameters
    ----------
    args : class Namespace
        Arguments as parsed by argparse.parse_args, in this case they can be empty or ommitted

    Returns
    -------
    return_code : int
        Return code for whether the command completed successfully

    Examples
    --------
    >>> from dstauffman.commands import execute_enforce
    >>> args = []
    >>> execute_enforce(args) # doctest: +SKIP

    """
    return_code = ReturnCodes.clean
    return return_code

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_commands_repos', exit=False)
    doctest.testmod(verbose=False)
