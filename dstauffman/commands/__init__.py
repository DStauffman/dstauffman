r"""
dstauffman python commands.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

# %% Imports
# fmt: off
from .help     import print_help, print_version, parse_help, parse_version, execute_help, \
                          execute_version
from .runtests import parse_tests, execute_tests, parse_coverage, execute_coverage
# fmt: on

# %% Unit test
if __name__ == "__main__":
    pass
