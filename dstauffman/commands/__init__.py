r"""
dstauffman python commands.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
from .help     import print_help, print_version, parse_help, parse_version, execute_help, execute_version
from .repos    import parse_enforce, execute_enforce, parse_make_init, execute_make_init
from .runtests import parse_tests, execute_tests, parse_coverage, execute_coverage

#%% Unittest
if __name__ == '__main__':
    pass
