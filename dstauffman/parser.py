r"""
Parser used to parse all commands from the terminal and pass to the revelant command functions.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

# %% Imports
from __future__ import annotations

import argparse
from dataclasses import dataclass
import doctest
import logging
import sys
from typing import Callable
import unittest

from slog import activate_logging, LogLevel, ReturnCodes

# %% Globals
logger = logging.getLogger(__name__)
_VALID_COMMANDS = frozenset({"coverage", "help", "tests", "version"})


@dataclass(frozen=True)
class _Flags:
    log_level: int | None
    use_display: bool
    use_plotting: bool


# %% Functions - _print_bad_command
def _print_bad_command(command: str) -> None:
    r"""Prints to the command line when a command name is not understood."""
    print(f'Command "{command}" is not understood.')


# %% Functions - main
def main() -> int:
    r"""Main function called when executed using the command line api."""
    try:
        (command, args) = parse_wrapper(sys.argv[1:])
    except ValueError:
        _print_bad_command(" ".join(sys.argv[1:]))
        return ReturnCodes.bad_command
    rc = execute_command(command, args)
    return sys.exit(rc)


# %% Functions - parse_wrapper
def parse_wrapper(args: list[str]) -> tuple[str, argparse.Namespace]:
    r"""Wrapper function to parse out the command name from the rest of the arguments."""
    # check for no command option
    if len(args) >= 1:
        command = args[0]
    else:
        command = "help"
    # check for alternative forms of help with the base dcs command
    if command in {"--help", "-h"}:
        command = "help"
    elif command in {"--version", "-v"}:
        command = "version"
    # pass the command and remaining arguments to the command parser
    parsed_args = parse_commands(command, args[1:])
    return (command, parsed_args)


# %% Functions - parse_commands
def parse_commands(command: str, args: list[str]) -> argparse.Namespace:
    r"""
    Splits the parsing based on the name of the command.

    Parameters
    ----------
    command : str
        Name of command to parse
    args : list
        Command line arguments

    Returns
    -------
    parsed_args : class argparse.Namespace
        Parsed arguments ready to be passed to command to execute

    Examples
    --------
    >>> from dstauffman import parse_commands
    >>> command = "help"
    >>> args = []
    >>> parsed_args = parse_commands(command, args)
    >>> print(parsed_args)
    Namespace()

    """
    # delayed import of commands
    import dstauffman.commands as commands  # pylint: disable=import-outside-toplevel

    # check for valid commands
    if command in _VALID_COMMANDS:
        # If valid, then parse the arguments with the appropiate method, so help calls parse_help etc.
        func = getattr(commands, "parse_" + command)
        parsed_args: argparse.Namespace = func(args)
    else:
        raise ValueError(f'Unexpected command "{command}".')
    return parsed_args


# %% Functions - execute_command
def execute_command(command: str, args: argparse.Namespace) -> int:
    r"""Executes the given command."""
    # delayed import of commands
    import dstauffman.commands as commands  # pylint: disable=import-outside-toplevel

    # check for valid commands
    if command in _VALID_COMMANDS:
        # If valid, then call the appropriate method, so help calls execute_help etc.
        func = getattr(commands, "execute_" + command)
        rc: int | None = func(args)
    else:
        _print_bad_command(command)
        rc = ReturnCodes.bad_command
    if rc is None:
        rc = ReturnCodes.clean
    return rc


# %% process_command_line_options
def process_command_line_options(log_start: bool | str | None = None) -> _Flags:
    r"""
    Parses sys.argv to determine any command line options for use in scripts.

    Parameters
    ----------
    log_start : bool or str, optional
        Whether to log the time of the start, and if a string, then log the name that started it

    Returns
    -------
    flags : dataclass _Flags
        Flags equivalent to the command line arguments

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.
    #.  Expanded by David C. Stauffer in April 2022 to optionally specify the file doing the logging.

    Examples
    --------
    >>> from dstauffman import process_command_line_options
    >>> flags = process_command_line_options()
    >>> print(flags.use_display)
    True

    """
    # get logger settings
    log_level = None
    for opt in sys.argv[1:]:
        if opt.startswith("-l"):
            if hasattr(LogLevel, level := opt[1:].upper()):
                log_level = getattr(LogLevel, level)
            elif hasattr(logging, level := opt[2:].upper()):
                log_level = getattr(logging, level)
            else:
                raise ValueError(f'Unexpected logging input of: "{opt}".')
            activate_logging(log_level, log_start=log_start)
            logger.log(log_level, "Configuring Log Level at: %s", log_level)

    # get other settings
    use_display = "-nodisp" not in sys.argv
    use_plotting = "-noplot" not in sys.argv

    # log any non-defaults
    # fmt: off
    print_func: Callable[[str], None] = lambda x: print(x) if log_level is None else lambda x: logger.log(LogLevel.L3, x)  # type: ignore[assignment]
    # fmt: on
    if not use_display:
        print_func("Running without displaying any plots.")
    if not use_plotting:
        print_func("Running without making any plots.")

    # do operations based on those settings
    if use_plotting and not use_display:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel

        suppress_plots()

    # return the settings
    flags = _Flags(log_level=log_level, use_display=use_display, use_plotting=use_plotting)

    return flags


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_parser", exit=False)
    doctest.testmod(verbose=False)
