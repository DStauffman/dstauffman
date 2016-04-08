# -*- coding: utf-8 -*-
r"""
Test file for the `bpe` module of the dstauffman code.  It is intented to contain test cases to
demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""

#%% Imports
import unittest
import dstauffman as dcs

#%% Logger
class Test_Logger(unittest.TestCase):
    r"""
    Tests the Logger class with the following cases:
        Get level
        Set level
        Bad level (raises ValueError)
        printing
    """
    def setUp(self):
        self.level  = 8
        self.logger = dcs.Logger(self.level)
        self.print  = 'Logger(8)'

    def test_get_level(self):
        level = self.logger.get_level()
        self.assertEqual(level, self.level)

    def test_set_level(self):
        level = self.logger.get_level()
        self.assertEqual(level, self.level)
        self.logger.set_level(5)
        self.assertEqual(self.logger.get_level(), 5)

    def test_null_instantiation(self):
        level = self.logger.get_level()
        logger = dcs.Logger()
        self.assertEqual(level, logger.get_level())

    def test_bad_level(self):
        with self.assertRaises(ValueError):
            self.logger.set_level(-1)

    def test_printing(self):
        with dcs.capture_output() as (out, _):
            print(self.logger)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.print)

#%% OptiOpts
class Test_OptiOpts(unittest.TestCase):
    r"""
    Tests the OptiOpts class with the following cases:
        Initialization
    """
    def test_init(self):
        opti_opts = dcs.OptiOpts()
        for this_attr in ['params', 'slope_method']:
            self.assertTrue(hasattr(opti_opts, this_attr))

#%% OptiParams
pass

#%% _calculate_jacobian
pass

#%% validate_opti_opts
@unittest.skip('Code is still being actively developed.')
class Test_validate_opti_opts(unittest.TestCase):
    r"""
    Tests the validate_opti_opts function with the following cases:
        TBD
    """
    def setUp(self):
        self.opti_opts           = dcs.OptiOpts()
        # TODO: set log level
        self.opti_opts.params    = [dcs.OptiParam('param.life.age_calibration', 1.0, 0, 10.)]

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'Validating optimization options.')

    def test_no_logging(self):
        # TODO: set log level self.opti_opts.log_level = 0
        with dcs.capture_output() as (out, _):
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'')

    def test_not_valid(self):
        with dcs.capture_output() as (out, err):
            with self.assertRaises(AssertionError):
                dcs.validate_opti_opts(dcs.OptiOpts())
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'Validating optimization options.')

#%% _analyze_results
pass

#%% get_parameters
pass

#%% set_parameters
pass

#%% run_bpe
pass

#%% plot_bpe_results
pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
