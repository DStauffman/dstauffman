r"""
Test file for the `logs` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2019.
"""

#%% Imports
import contextlib
import datetime
import logging
import os
import time
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

#%% activate_logging and deactivate_logging
class Test_act_deact_logging(unittest.TestCase):
    r"""
    Tests the activate_logging and deactivate_logging functions with the following cases:
        Nominal
        Default filename
    """
    def setUp(self) -> None:
        self.level    = dcs.LogLevel.L5
        self.filename = os.path.join(dcs.get_tests_dir(), 'testlog.txt')

    def test_nominal(self) -> None:
        self.assertFalse(os.path.isfile(self.filename))
        dcs.activate_logging(self.level, self.filename)
        self.assertTrue(os.path.isfile(self.filename))
        self.assertTrue(dcs.logs.root_logger.hasHandlers())
        with self.assertLogs(level='L5') as logs:
            logger = logging.getLogger('Test')
            logger.log(dcs.LogLevel.L5, 'Test message')
        lines = logs.output
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], 'L5:Test:Test message')
        dcs.deactivate_logging()
        self.assertFalse(dcs.logs.root_logger.handlers)

    def test_default_filename(self) -> None:
        default_filename = os.path.join(dcs.get_output_dir(), 'log_file_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.txt')
        was_there = os.path.isfile(default_filename)
        dcs.activate_logging(self.level)
        self.assertTrue(dcs.logs.root_logger.hasHandlers())
        time.sleep(0.01)
        dcs.deactivate_logging()
        self.assertFalse(dcs.logs.root_logger.handlers)
        if not was_there:
            with contextlib.suppress(FileNotFoundError):
                os.remove(default_filename)

    def tearDown(self) -> None:
        dcs.deactivate_logging()
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.filename)

#%% log_multiline
class Test_log_multiline(unittest.TestCase):
    r"""
    Tests the log_multiline function with the following cases:
        TBD
    """
    level: int
    logger: logging.Logger

    @classmethod
    def setUpClass(cls) -> None:
        cls.level = dcs.LogLevel.L5
        cls.logger = logging.getLogger('Test')
        dcs.activate_logging(cls.level)

    def test_normal(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, 'Normal message.')
        lines = logs.output
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], 'L5:Test:Normal message.')

    def test_multiline1(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, 'Multi-line\nMessage.')
        lines = logs.output
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], 'L5:Test:Multi-line')
        self.assertEqual(lines[1], 'L5:Test:Message.')

    def test_multiline2(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, 'Multi-line', 'Message.')
        lines = logs.output
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], 'L5:Test:Multi-line')
        self.assertEqual(lines[1], 'L5:Test:Message.')

    def test_multiline3(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, 'List value:', [1, 2, 3])
        lines = logs.output
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], 'L5:Test:List value:')
        self.assertEqual(lines[1], 'L5:Test:[1, 2, 3]')

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_numpy1(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        lines = logs.output
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], 'L5:Test:[[1 2 3]')
        self.assertEqual(lines[1], 'L5:Test: [4 5 6]')
        self.assertEqual(lines[2], 'L5:Test: [7 8 9]]')

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_numpy2(self) -> None:
        with self.assertLogs(logger=self.logger, level=self.level) as logs:
            dcs.log_multiline(self.logger, self.level, 'Numpy solution:',  np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        lines = logs.output
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], 'L5:Test:Numpy solution:')
        self.assertEqual(lines[1], 'L5:Test:[[1 2 3]')
        self.assertEqual(lines[2], 'L5:Test: [4 5 6]')
        self.assertEqual(lines[3], 'L5:Test: [7 8 9]]')

    @classmethod
    def tearDownClass(cls) -> None:
        dcs.deactivate_logging()

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
