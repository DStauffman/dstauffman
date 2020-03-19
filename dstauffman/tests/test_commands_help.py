# -*- coding: utf-8 -*-
r"""
Test file for the `commands.help` module of the "dstauffman" library.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import unittest

import dstauffman.commands as commands

#%% commands.print_help
pass

#%% commands.parse_help
class Test_commands_parse_help(unittest.TestCase):
    r"""
    Tests the commands.parse_help function with the following cases:
        TBD
    """
    def setUp(self):
        self.args = []

    def test_nominal(self):
        commands.parse_help(self.args)

#%% commands.execute_help
pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
