# -*- coding: utf-8 -*-
r"""
Test file for the `commands.help` module of the "dstauffman" library.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import unittest

import dstauffman as dcs
import dstauffman.commands as commands

#%% commands.print_help
class Test_commands_print_help(unittest.TestCase):
    r"""
    Tests the commands.print_help function with the following cases:
        Nominal
    """
    def test_nominal(self):
        with dcs.capture_output() as out:
            commands.print_help()
        output = out.getvalue().strip()
        out.close()
        expected_header = output.startswith('#######\nlmspace\n#######\n') or \
                          output.startswith('##########\ndstauffman\n##########\n')
        self.assertTrue(expected_header)

#%% commands.parse_help
class Test_commands_parse_help(unittest.TestCase):
    r"""
    Tests the commands.parse_help function with the following cases:
        Nominal
    """
    def setUp(self):
        self.args = []

    def test_nominal(self):
        commands.parse_help(self.args)

#%% commands.execute_help
class Test_commands_execute_help(unittest.TestCase):
    r"""
    Tests the commands.execute_help function with the following cases:
        Nominal
    """
    def setUp(self):
        self.args = argparse.Namespace()

    def test_nominal(self):
        with dcs.capture_output() as out:
            commands.execute_help(self.args)
        output = out.getvalue().strip()
        out.close()
        expected_header = output.startswith('#######\nlmspace\n#######\n') or \
                          output.startswith('##########\ndstauffman\n##########\n')
        self.assertTrue(expected_header)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
