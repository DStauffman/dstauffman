# -*- coding: utf-8 -*-
r"""
Test file for the `parser` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
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
    pass

#%% parse_commands
class Test_parse_commands(unittest.TestCase):
    pass

#%% parse_wrapper
class Test_execute_command(unittest.TestCase):
    pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
