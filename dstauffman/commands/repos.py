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
import os
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
    >>> from dstauffman import get_root_dir
    >>> from dstauffman.commands import parse_enforce
    >>> input_args = [get_root_dir()]
    >>> args = parse_enforce(input_args)
    >>> print(args) # doctest: +ELLIPSIS
    Namespace(execute=False, extensions=None, folder='...', ignore_tabs=False, list_all=False, skip=None, trailing=False, unix=False, windows=False)

    """
    parser = argparse.ArgumentParser(prog='dcs enforce', description='Enforce consistency in the repo ' + \
            'for things like tabs, trailing whitespace, line endings and file execute permissions.')

    parser.add_argument('folder', help='Folder to search for source files')
    parser.add_argument('-e', '--extensions', help='Extensions to search through.', action='append')
    parser.add_argument('-l', '--list-all', help='List all files, even ones without problems.', action='store_true')
    parser.add_argument('-i', '--ignore-tabs', help='Ignore tabs within the source code.', action='store_true')
    parser.add_argument('-t', '--trailing', help='Show files with trailing whitespace', action='store_true')
    parser.add_argument('-s', '--skip', help='Exclusions to not search.', action='append')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-w', '--windows', help='Use Windows (CRLF) line-endings', action='store_true')
    group.add_argument('-u', '--unix', help='Use Unix (LF) line-endings', action='store_true')

    parser.add_argument('-x', '--execute', help='List files with execute permissions.', action='store_true')

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
    >>> from dstauffman import get_root_dir
    >>> from dstauffman.commands import execute_enforce
    >>> from argparse import Namespace
    >>> args = Namespace(extensions=None, folder=get_root_dir(), ignore_tabs=False, list_all=False, \
    ...     skip=None, trailing=False, unix=False, windows=False, execute=False)
    >>> return_code = execute_enforce(args) # doctest: +SKIP

    """
    # defaults
    def_extensions = {'m', 'py'}
    # get settings from input arguments
    folder     = os.path.abspath(args.folder)
    list_all   = args.list_all
    check_tabs = not args.ignore_tabs
    trailing   = args.trailing
    exclusions = args.skip
    show_execute = args.execute
    if args.windows:
        check_eol = '\r\n'
    elif args.unix:
        check_eol = '\n'
    else:
        check_eol = None
    if args.extensions is None:
        extensions = frozenset(def_extensions)
    elif len(args.extensions) == 1 and args.extensions[0] == '*':
        extensions = None
    else:
        extensions = args.extensions

    # call the function to do the checks
    is_clean = find_repo_issues(folder=folder, extensions=extensions, list_all=list_all, \
            check_tabs=check_tabs, trailing=trailing, exclusions=exclusions, check_eol=check_eol, \
            show_execute=show_execute)
    # return a status based on whether anything was found
    return_code = ReturnCodes.clean if is_clean else ReturnCodes.test_failures
    return return_code

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_commands_repos', exit=False)
    doctest.testmod(verbose=False)
