r"""
Test file for the `parser` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import argparse
import unittest
from unittest.mock import patch

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
    r"""
    Tests the main function with the following cases:
        TBD
    """
    pass # TODO: write this

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

#%% execute_command
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

#%% process_command_line_options
class Test_process_command_line_options(unittest.TestCase):
    r"""
    Tests the process_command_line_options function with the following cases:
        Nominal
        No display
        No plotting
        No HDF5
    """
    def test_nominal(self):
        flags = dcs.process_command_line_options()
        # check expected defaults
        self.assertTrue(flags.use_display)
        self.assertTrue(flags.use_plotting)
        self.assertTrue(flags.use_hdf5)
        # check that only the expected keys exist
        keys = {x for x in vars(flags) if not x.startswith('_')}
        self.assertEqual(keys, {'use_display', 'use_hdf5', 'use_plotting'})

    def test_no_display(self):
        with dcs.capture_output() as out:
            with patch('sys.argv', ['name.py', '-nodisp']):
                flags = dcs.process_command_line_options()
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(flags.use_display)
        self.assertTrue(flags.use_plotting)
        self.assertTrue(flags.use_hdf5)
        self.assertEqual(output, 'Running without displaying any plots.')

    def test_no_plotting(self):
        with dcs.capture_output() as out:
            with patch('sys.argv', ['name.py', '-noplot']):
                flags = dcs.process_command_line_options()
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(flags.use_display)
        self.assertFalse(flags.use_plotting)
        self.assertTrue(flags.use_hdf5)
        self.assertEqual(output, 'Running without making any plots.')


    def test_no_hdf5(self):
        with dcs.capture_output() as out:
            with patch('sys.argv', ['name.py', '-nohdf5']):
                flags = dcs.process_command_line_options()
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(flags.use_display)
        self.assertTrue(flags.use_plotting)
        self.assertFalse(flags.use_hdf5)
        self.assertEqual(output, 'Running without saving to HDF5 files.')


#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
