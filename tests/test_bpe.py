# -*- coding: utf-8 -*-
r"""
Test file for the `bpe` module of the dstauffman code.  It is intented to contain test cases to
demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""

#%% Imports
import numpy as np
import os
import unittest
import dstauffman as dcs

#%% Setup for testing
# Classes - SimParams
class SimParams(dcs.Frozen):
    r"""Simulation model parameters."""
    def __init__(self, time, *, magnitude, frequency, phase):
        self.time      = time
        self.magnitude = magnitude
        self.frequency = frequency
        self.phase     = phase

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for key in vars(self):
            if np.any(getattr(self, key) != getattr(other, key)):
                return False
        return True

# Functions - _get_truth_index
def _get_truth_index(results_time, truth_time):
    r"""
    Finds the indices to the truth data from the results time.
    """
    # Hard-coded values
    precision    = 1e-7
    # find the indices to truth
    ix_truth     = np.nonzero((truth_time >= results_time[0] - precision) & (truth_time <= \
        results_time[-1] + precision))[0]
    # find the indices to results (in case truth isn't long enough)
    ix_results   = np.nonzero(results_time <= truth_time[-1] + precision)[0]
    # return the indices
    return (ix_truth, ix_results)

# Functions - sim_model
def sim_model(sim_params):
    r"""Simple example simulation model."""
    return sim_params.magnitude * np.sin(2*np.pi*sim_params.frequency*sim_params.time/1000 + \
        sim_params.phase*np.pi/180)

# Functions - truth
def truth(time, magnitude=5, frequency=10, phase=90):
    r"""Simple example truth data."""
    return magnitude * np.sin(2*np.pi*frequency*time/1000 + phase*np.pi/180)

# Functions - cost_wrapper
def cost_wrapper(results_data, *, results_time, truth_time, truth_data):
    r"""Example Cost wrapper for the model."""
    # Pull out overlapping time points and indices
    (ix_truth, ix_results) = _get_truth_index(results_time, truth_time)
    sub_truth  = truth_data[ix_truth]
    sub_result = results_data[ix_results]

    # calculate the innovations
    innovs = sub_result - sub_truth
    return innovs

# Functions - get_parameter
def get_parameter(sim_params, *, names):
    r"""Simple example parameter getter."""
    num = len(names)
    values = np.nan * np.ones(num)
    for (ix, name) in enumerate(names):
        if hasattr(sim_params, name):
            values[ix] = getattr(sim_params, name)
        else:
            raise ValueError('Bad parameter name: "{}".'.format(name))
    return values

# Functions - set_parameter
def set_parameter(sim_params, *, names, values):
    r"""Simple example parameter setter."""
    num = len(names)
    assert len(values) == num, 'Names and Values must have the same length.'
    for (ix, name) in enumerate(names):
        if hasattr(sim_params, name):
            setattr(sim_params, name, values[ix])
        else:
            raise ValueError('Bad parameter name: "{}".'.format(name))

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
        with dcs.capture_output() as out:
            print(self.logger)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.print)

#%% OptiOpts
class Test_OptiOpts(unittest.TestCase):
    r"""
    Tests the OptiOpts class with the following cases:
        Initialization
        Equality
        Inequality
    """
    def test_init(self):
        opti_opts = dcs.OptiOpts()
        self.assertTrue(isinstance(opti_opts, dcs.OptiOpts))

    def test_equality(self):
        opti_opts1 = dcs.OptiOpts()
        opti_opts2 = dcs.OptiOpts()
        self.assertEqual(opti_opts1, opti_opts2)

    def test_inequality(self):
        opti_opts1 = dcs.OptiOpts()
        opti_opts2 = dcs.OptiOpts()
        opti_opts2.grow_radius = 5.5
        self.assertNotEqual(opti_opts1, opti_opts2)

    def test_inequality2(self):
        opti_opts = dcs.OptiOpts()
        self.assertNotEqual(opti_opts, 2)

#%% OptiParam
class Test_OptiParam(unittest.TestCase):
    r"""
    Tests the OptiParam class with the following cases:
        Initialization
        Equality
        Inequality
        Get array
        Get names
    """
    def test_init(self):
        opti_param = dcs.OptiParam('test')
        self.assertTrue(isinstance(opti_param, dcs.OptiParam))

    def test_equality(self):
        opti_param1 = dcs.OptiParam('test')
        opti_param2 = dcs.OptiParam('test')
        self.assertEqual(opti_param1, opti_param2)

    def test_inequality(self):
        opti_param1 = dcs.OptiParam('test')
        opti_param2 = dcs.OptiParam('test')
        opti_param2.min_ = 5.5
        self.assertNotEqual(opti_param1, opti_param2)

    def test_inequality2(self):
        opti_param = dcs.OptiParam('test')
        self.assertNotEqual(opti_param, 2)

    def test_get_array(self):
        opti_param = dcs.OptiParam('test')
        params = [opti_param, opti_param]
        best = dcs.OptiParam.get_array(params)
        np.testing.assert_array_equal(best, np.array([np.nan, np.nan]))

    def test_get_names(self):
        opti_param1 = dcs.OptiParam('test1')
        opti_param2 = dcs.OptiParam('test2')
        params = [opti_param1, opti_param2]
        names = dcs.OptiParam.get_names(params)
        self.assertEqual(names, ['test1', 'test2'])

#%% BpeResults
class Test_BpeResults(unittest.TestCase):
    r"""
    Tests the BpeResults class with the following cases:
        Save (HDF5)
        Load (HDF5)
    """
    def setUp(self):
        self.bpe_results = dcs.BpeResults()
        self.bpe_results.num_evals = 5
        self.filename    = os.path.join(dcs.get_tests_dir(), 'test_bpe_results.hdf5')
        self.filename2   = self.filename.replace('hdf5', 'pkl')
        dcs.Logger.set_level(1)

    def test_save(self):
        self.bpe_results.save(self.filename)
        self.assertTrue(os.path.isfile(self.filename))

    def test_save2(self):
        self.bpe_results.save(self.filename, use_hdf5=False)
        self.assertTrue(os.path.isfile(self.filename2))

    def test_load(self):
        self.bpe_results.save(self.filename)
        bpe_results = dcs.BpeResults.load(self.filename)
        self.assertTrue(dcs.compare_two_classes(bpe_results, self.bpe_results, suppress_output=True))

    def test_load2(self):
        self.bpe_results.save(self.filename, use_hdf5=False)
        bpe_results = dcs.BpeResults.load(self.filename, use_hdf5=False)
        self.assertTrue(dcs.compare_two_classes(bpe_results, self.bpe_results, suppress_output=True))

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        if os.path.isfile(self.filename2):
            os.remove(self.filename2)

#%% _print_divider
class Test__print_divider(unittest.TestCase):
    r"""
    Tests the _print_divider function with the following cases:
        TBD
    """
    def setUp(self):
        self.output = '******************************'

    def test_with_new_line(self):
        with dcs.capture_output() as out:
            dcs.bpe._print_divider()
        lines = out.getvalue().split('\n')
        self.assertEqual(lines[0], '')
        self.assertEqual(lines[1], self.output)

    def test_no_new_line(self):
        with dcs.capture_output() as out:
            dcs.bpe._print_divider(new_line=False)
        lines = out.getvalue().split('\n')
        self.assertEqual(lines[0], self.output)

#%% _pprint_args
class Test__pprint_args(unittest.TestCase):
    r"""
    Tests the _pprint_args function with the following cases:
        TBD
    """
    def setUp(self):
        self.names  = ['Name 1', 'Longer name 2', 'Name 42']
        self.values = [0.10000000002, 1999999999, 1e-14]
        self.lines  = ['        Name 1        = 0.1', '        Longer name 2 = 2e+09', '        Name 42       = 1e-14', '']

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.bpe._pprint_args(self.names, self.values)
        lines = out.getvalue().split('\n')
        for i in range(len(self.lines)):
            self.assertEqual(lines[i], self.lines[i])

#%% _function_wrapper
class Test__function_wrapper(unittest.TestCase):
    r"""
    Tests the _function_wrapper function with the following cases:
        Nominal
        Model args
        Cost args
    """
    def setUp(self):
        self.results = np.array([1, 2, np.nan])
        self.innovs  = np.array([1, 2, 0])
        func = lambda *args, **kwargs: np.array([1, 2, np.nan])
        self.opti_opts = type('Class1', (object, ), {'model_args': {}, 'cost_args': {}, 'model_func': func, 'cost_func': func})
        self.bpe_results = type('Class2', (object, ), {'num_evals': 0})

    def test_nominal(self):
        (results, innovs) = dcs.bpe._function_wrapper(self.opti_opts, self.bpe_results)
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_model_args(self):
        (results, innovs) = dcs.bpe._function_wrapper(self.opti_opts, self.bpe_results, model_args={'a': 5})
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_cost_args(self):
        (results, innovs) = dcs.bpe._function_wrapper(self.opti_opts, self.bpe_results, cost_args={'a': 5})
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

#%% _finite_differences
pass

#%% _levenberg_marquardt
class Test__levenberg_marquardt(unittest.TestCase):
    r"""
    Tests the _levenberg_marquardt function with the following cases:
        with lambda_
        without lambda_
    """
    def setUp(self):
        self.jacobian    = np.array([[1, 2], [3, 4], [5, 6]])
        self.innovs      = np.array([7, 8, 9])
        self.lambda_     = 5
        self.delta_param = np.array([-0.46825397, -1.3015873])

    def test_nominal(self):
        delta_param = dcs.bpe._levenberg_marquardt(self.jacobian, self.innovs, self.lambda_)
        np.testing.assert_array_almost_equal(delta_param, self.delta_param)

    def test_lambda_zero(self):
        b = -np.linalg.pinv(self.jacobian).dot(self.innovs)
        delta_param = dcs.bpe._levenberg_marquardt(self.jacobian, self.innovs, 0)
        np.testing.assert_array_almost_equal(delta_param, b)

#%% _predict_func_change
pass

#%% _check_for_convergence
pass

#%% _double_dogleg
pass

#%% _dogleg_search
pass

#%% _analyze_results
pass

#%% validate_opti_opts
@unittest.skip('Code is still being actively developed.')
class Test_validate_opti_opts(unittest.TestCase):
    r"""
    Tests the validate_opti_opts function with the following cases:
        TBD
    """
    def setUp(self):
        self.opti_opts        = dcs.OptiOpts()
        self.opti_opts.params = [dcs.OptiParam("param[:].tb.tb_new_inf['beta']", 0.1375, 0.1, 0.2, typical=0.14)]

    def test_nominal(self):
        with dcs.capture_output() as out:
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'Validating optimization options.')

    def test_no_logging(self):
        # TODO: set log level self.opti_opts.log_level = 0
        with dcs.capture_output() as out:
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'')

    def test_not_valid(self):
        with dcs.capture_output() as out:
            with self.assertRaises(AssertionError):
                dcs.validate_opti_opts(dcs.OptiOpts())
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'Validating optimization options.')

#%% run_bpe
pass

#%% plot_bpe_results
pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
