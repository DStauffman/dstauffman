r"""
Test file for the `help` module of the "dstauffman.commands" library.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
from typing import List
import unittest

import dstauffman as dcs
import dstauffman.commands as commands

#%% commands.print_help
class Test_commands_print_help(unittest.TestCase):
    r"""
    Tests the commands.print_help function with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            commands.print_help()
        output = out.getvalue().strip()
        out.close()
        expected_header = output.startswith('#######\nlmspace\n#######\n') or \
                          output.startswith('##########\ndstauffman\n##########\n')
        self.assertTrue(expected_header)

#%% commands.print_version
class Test_commands_print_version(unittest.TestCase):
    r"""
    Tests the commands.print_version function with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            commands.print_version()
        output = out.getvalue().strip()
        out.close()
        self.assertIn('.', output)

#%% commands.parse_help
class Test_commands_parse_help(unittest.TestCase):
    r"""
    Tests the commands.parse_help function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.args: List[str] = []
        self.expected = argparse.Namespace()

    def test_nominal(self) -> None:
        args = commands.parse_help(self.args)
        self.assertEqual(args, self.expected)

#%% commands.parse_version
class Test_commands_parse_version(unittest.TestCase):
    r"""
    Tests the commands.parse_version function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.args: List[str] = []
        self.expected = argparse.Namespace()

    def test_nominal(self) -> None:
        args = commands.parse_version(self.args)
        self.assertEqual(args, self.expected)

#%% commands.execute_help
class Test_commands_execute_help(unittest.TestCase):
    r"""
    Tests the commands.execute_help function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.args = argparse.Namespace()

    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            commands.execute_help(self.args)
        output = out.getvalue().strip()
        out.close()
        expected_header = output.startswith('#######\nlmspace\n#######\n') or \
                          output.startswith('##########\ndstauffman\n##########\n')
        self.assertTrue(expected_header)

#%% commands.execute_version
class Test_commands_execute_version(unittest.TestCase):
    r"""
    Tests the commands.execute_version function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.args = argparse.Namespace()

    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            commands.execute_version(self.args)
        output = out.getvalue().strip()
        out.close()
        self.assertIn('.', output)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
