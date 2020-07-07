r"""
Test file for the `parser` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import unittest

import dstauffman as dcs

#%% _VALID_COMMANDS
class Test__VALID_COMMANDS(unittest.TestCase):
    r"""
    Tests the _VALID_COMMANDS enumerator for expected values.
    """
    def test_nominal(self):
        self.assertIn('enforce', dcs.parser._VALID_COMMANDS)
        self.assertIn('help',    dcs.parser._VALID_COMMANDS)
        self.assertIn('tests',   dcs.parser._VALID_COMMANDS)

#%% _print_bad_command
class Test__print_bad_command(unittest.TestCase):
    r"""
    Tests the _print_bad_command function with the following cases:
        Nominal
    """
    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.parser._print_bad_command('garbage')
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Command "garbage" is not understood.')

#%% main
class Test_main(unittest.TestCase):
    pass

#%% parse_wrapper
class Test_parse_wrapper(unittest.TestCase):
    r"""
    Tests the parse_wrapper function with the following cases:
        Nominal
        Help (x2)
    """
    def test_nominal(self):
        (command, parsed_args) = dcs.parse_wrapper(['tests', '-dv'])
        self.assertEqual(command, 'tests')
        self.assertEqual(parsed_args, argparse.Namespace(docstrings=True, library=None, verbose=True))

    def test_help1(self):
        (command, parsed_args) = dcs.parse_wrapper([])
        self.assertEqual(command, 'help')

    def test_help2(self):
        (command, parsed_args) = dcs.parse_wrapper(['-h'])
        self.assertEqual(command, 'help')

#%% parse_commands
class Test_parse_commands(unittest.TestCase):
    r"""
    Tests the parse_commands function with the following cases:
        Valid command
        Bad command
    """
    def test_valid_command(self):
        parsed_args = dcs.parse_commands('tests', ['-dv'])
        self.assertEqual(parsed_args, argparse.Namespace(docstrings=True, library=None, verbose=True))

    def test_bad_command(self):
        with self.assertRaises(ValueError) as context:
            dcs.parse_commands('bad', [])
        self.assertEqual(str(context.exception), 'Unexpected command "bad".')

#%% parse_wrapper
class Test_execute_command(unittest.TestCase):
    r"""
    Tests the execute_command function with the following cases:
        Good command
        Bad command
    """
    def test_good_command(self):
        with dcs.capture_output() as out:
            rc = dcs.execute_command('help', [])
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(rc, dcs.ReturnCodes.clean)
        self.assertTrue(output)

    def test_bad_command(self):
        with dcs.capture_output() as out:
            rc = dcs.execute_command('bad', [])
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Command "bad" is not understood.')
        self.assertEqual(rc, dcs.ReturnCodes.bad_command)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
