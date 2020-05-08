# -*- coding: utf-8 -*-
r"""
dstauffman python commands.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
from dstauffman.commands.help     import print_help, parse_help, execute_help
from dstauffman.commands.repos    import parse_enforce, execute_enforce, parse_make_init, \
                                             execute_make_init
from dstauffman.commands.runtests import parse_tests, execute_tests, parse_coverage, \
                                             execute_coverage

#%% Unittest
if __name__ == '__main__':
    pass
