r"""
Test file for the `logs` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in July 2019.
"""

#%% Imports
import logging
import os
import time
import unittest
from datetime import datetime

import dstauffman as dcs

#%% activate_logging and deactivate_logging
class Test_act_deact_logging(unittest.TestCase):
    r"""
    Tests the activate_logging and deactivate_logging functions with the following cases:
        Nominal
        Default filename
    """
    def setUp(self):
        self.level    = logging.DEBUG
        self.filename = os.path.join(dcs.get_tests_dir(), 'testlog.txt')

    def test_nominal(self):
        self.assertFalse(os.path.isfile(self.filename))
        dcs.activate_logging(self.level, self.filename)
        self.assertTrue(os.path.isfile(self.filename))
        self.assertTrue(dcs.logs.root_logger.hasHandlers())
        with self.assertLogs(level='DEBUG') as cm:
            logger = logging.getLogger('Test')
            logger.debug('Test message')
        lines = cm.output
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], 'DEBUG:Test:Test message')
        dcs.deactivate_logging()
        self.assertFalse(dcs.logs.root_logger.handlers)

    def test_default_filename(self):
        default_filename = os.path.join(dcs.get_output_dir(), 'log_file_' + datetime.now().strftime('%Y-%m-%d') + '.txt')
        was_there = os.path.isfile(default_filename)
        dcs.activate_logging(self.level)
        self.assertTrue(dcs.logs.root_logger.hasHandlers())
        time.sleep(0.01)
        dcs.deactivate_logging()
        self.assertFalse(dcs.logs.root_logger.handlers)
        if not was_there:
            os.remove(default_filename)

    def tearDown(self):
        dcs.deactivate_logging()
        if os.path.isfile(self.filename):
            os.remove(self.filename)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
