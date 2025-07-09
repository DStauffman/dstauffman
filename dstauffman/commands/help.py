r"""
Functions related to `help` and `version` commands.

Notes
-----
#.  Written by David C. Stauffer in March 2020.

"""

# %% Imports
import argparse
import doctest
from pathlib import Path
import unittest

from slog import ReturnCodes

from dstauffman import get_root_dir, version_info


# %% Functions - print_help
def print_help(help_file: Path | None = None) -> int:
    r"""
    Prints the contents of the README.rst file.

    Returns
    -------
    return_code : int
        Return code for whether the help file was successfully loaded

    Examples
    --------
    >>> from dstauffman.commands import print_help
    >>> print_help()  # doctest: +SKIP

    """
    if help_file is None:
        help_file = get_root_dir().parent / "README.rst"
    if not help_file.is_file():
        print(f'Warning: help file at "{help_file}" was not found.')
        return ReturnCodes.bad_help_file
    with open(help_file, encoding="utf-8") as file:
        text = file.read()
    print(text)
    return ReturnCodes.clean


# %% Functions - print_version
def print_version() -> int:
    r"""
    Prints the version of the library.

    Returns
    -------
    return_code : int
        Return code for whether the version was successfully read

    Examples
    --------
    >>> from dstauffman.commands import print_version
    >>> print_version()  # doctest: +SKIP

    """
    try:
        version = ".".join(str(x) for x in version_info)
        return_code = ReturnCodes.clean
    except Exception:  # pylint: disable=broad-exception-caught
        version = "unknown"
        return_code = ReturnCodes.bad_version
    print(version)
    return return_code


# %% Functions - parse_help
def parse_help(input_args: list[str]) -> argparse.Namespace:
    r"""
    Parser for help command.

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
    >>> from dstauffman.commands import parse_help
    >>> input_args = []
    >>> args = parse_help(input_args)
    >>> print(args)
    Namespace()

    """
    parser = argparse.ArgumentParser(prog="dcs help")

    args = parser.parse_args(input_args)
    return args


# %% Functions - parse_version
def parse_version(input_args: list[str]) -> argparse.Namespace:
    r"""
    Parser for version command.

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
    >>> from dstauffman.commands import parse_version
    >>> input_args = []
    >>> args = parse_version(input_args)
    >>> print(args)
    Namespace()

    """
    parser = argparse.ArgumentParser(prog="dcs version")

    args = parser.parse_args(input_args)
    return args


# %% Functions - execute_help
def execute_help(args: argparse.Namespace) -> int:  # pylint: disable=unused-argument
    r"""
    Executes the help command.

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
    >>> from dstauffman.commands import execute_help
    >>> args = []
    >>> execute_help(args)  # doctest: +SKIP

    """
    return_code = print_help()
    return return_code


# %% Functions - execute_version
def execute_version(args: argparse.Namespace) -> int:  # pylint: disable=unused-argument
    r"""
    Executes the version command.

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
    >>> from dstauffman.commands import execute_version
    >>> args = []
    >>> execute_version(args)  # doctest: +SKIP

    """
    return_code = print_version()
    return return_code


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_commands_help", exit=False)
    doctest.testmod(verbose=False)
