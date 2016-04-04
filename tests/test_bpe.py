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

#%% OptiOpts
class Test_OptiOpts(unittest.TestCase):
    r"""
    Tests the OptiOpts class with the following cases:
        Initialization
    """
    def test_init(self):
        opti_opts = dcs.OptiOpts()
        for this_attr in ['log_level', 'params', 'slope_method']:
            self.assertTrue(hasattr(opti_opts, this_attr))

#%% OptiParams
pass

#%% _calculate_jacobian
pass

#%% _validate_opti_opts
class Test__validate_opti_opts(unittest.TestCase):
    r"""
    Tests the _validate_opti_opts function with the following cases:
        TBD
    """
    def setUp(self):
        self.opti_opts           = dcs.OptiOpts()
        self.opti_opts.log_level = 10
        self.opti_opts.params    = [dcs.OptiParam('param.life.age_calibration', 1.0, 0, 10.)]

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            is_valid = dcs.bpe._validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'Validating optimization options.')

    def test_no_logging(self):
        self.opti_opts.log_level = 0
        with dcs.capture_output() as (out, _):
            is_valid = dcs.bpe._validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'')

    def test_not_valid(self):
        with dcs.capture_output() as (out, err):
            with self.assertRaises(AssertionError):
                dcs.bpe._validate_opti_opts(dcs.OptiOpts())
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
