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

#%% Hard-coded values
plotter = dcs.Plotter(False)

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
def cost_wrapper(results_data, *, results_time, truth_time, truth_data, sim_params):
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
    values = np.full(num, np.nan, dtype=float)
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
        Get array (x5)
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

    def test_get_array2(self):
        opti_param = dcs.OptiParam('test')
        params = [opti_param, opti_param]
        values = dcs.OptiParam.get_array(params, type_='min')
        np.testing.assert_array_equal(values, np.array([-np.inf, -np.inf]))

    def test_get_array3(self):
        opti_param = dcs.OptiParam('test')
        params = [opti_param, opti_param]
        values = dcs.OptiParam.get_array(params, type_='max')
        np.testing.assert_array_equal(values, np.array([np.inf, np.inf]))

    def test_get_array4(self):
        opti_param = dcs.OptiParam('test')
        params = [opti_param, opti_param]
        with self.assertRaises(ValueError):
            dcs.OptiParam.get_array(params, type_='typical')

    def test_get_array5(self):
        opti_param = dcs.OptiParam('test')
        params = [opti_param, opti_param]
        with self.assertRaises(ValueError):
            dcs.OptiParam.get_array(params, type_='bad_name')

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
        str method
        pprint method
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

    def test_str(self):
        with dcs.capture_output() as out:
            print(self.bpe_results)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'BpeResults:')
        self.assertTrue(lines[1].startswith('  begin_params: '))

    def test_pprint(self):
        self.bpe_results.param_names  = ['a'.encode('utf-8')]
        self.bpe_results.begin_params = [1]
        self.bpe_results.final_params = [2]
        with dcs.capture_output() as out:
            self.bpe_results.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Initial parameters:')
        self.assertEqual(lines[1].strip(), 'a = 1')
        self.assertEqual(lines[2], 'Final parameters:')
        self.assertEqual(lines[3].strip(), 'a = 2')

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        if os.path.isfile(self.filename2):
            os.remove(self.filename2)

#%% CurrentResults
class Test_CurrentResults(unittest.TestCase):
    r"""
    Tests the CurrentResults class with the following cases:
        Printing
    """
    def setUp(self):
        self.current_results = dcs.CurrentResults()

    def test_printing(self):
        with dcs.capture_output() as out:
            print(self.current_results)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Current Results:')
        self.assertEqual(lines[1], '  Trust Radius: None')
        self.assertEqual(lines[2], '  Best Cost: None')
        self.assertEqual(lines[3], '  Best Params: None')

#%% _pprint_args
class Test__pprint_args(unittest.TestCase):
    r"""
    Tests the _pprint_args function with the following cases:
        Nominal
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

#%% _print_divider
class Test__print_divider(unittest.TestCase):
    r"""
    Tests the _print_divider function with the following cases:
        With new line
        Without new line
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
class Test__predict_func_change(unittest.TestCase):
    r"""
    Tests the _predict_func_change function with the following cases:
        Nominal
    """
    def setUp(self):
        self.delta_param = np.array([1, 2])
        self.gradient    = np.array([[3], [4]])
        self.hessian     = np.array([[5, 2], [2, 5]])
        self.pred_change = 27.5

    def test_nominal(self):
        delta_func = dcs.bpe._predict_func_change(self.delta_param, self.gradient, self.hessian)
        self.assertEqual(delta_func, self.pred_change)

#%% _check_for_convergence
class Test__check_for_convergence(unittest.TestCase):
    r"""
    Tests the _check_for_convergence function with the following cases:
        TBD
    """
    def setUp(self):
        self.logger           = dcs.Logger(10)
        self.opti_opts        = type('Class1', (object, ), {'tol_cosmax_grad': 1, 'tol_delta_step': 2, \
            'tol_delta_cost': 3})
        self.cosmax           = 10
        self.delta_step_len   = 10
        self.pred_func_change = 10

    def test_not_converged(self):
        convergence = dcs.bpe._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, self.pred_func_change)
        self.assertFalse(convergence)

    def test_convergence1(self):
        with dcs.capture_output() as out:
            convergence = dcs.bpe._check_for_convergence(self.opti_opts, 0.5, self.delta_step_len, self.pred_func_change)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(convergence)
        self.assertEqual(output, 'Declare convergence because cosmax of 0.5 <= options.tol_cosmax_grad of 1')

    def test_convergence2(self):
        with dcs.capture_output() as out:
            convergence = dcs.bpe._check_for_convergence(self.opti_opts, self.cosmax, 1.5, self.pred_func_change)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(convergence)
        self.assertEqual(output, 'Declare convergence because delta_step_len of 1.5 <= options.tol_delta_step of 2')

    def test_convergence3(self):
        with dcs.capture_output() as out:
            convergence = dcs.bpe._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, -2.5)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(convergence)
        self.assertEqual(output, 'Declare convergence because abs(pred_func_change) of 2.5 <= options.tol_delta_cost of 3')

    def test_convergence4(self):
        with dcs.capture_output() as out:
            convergence = dcs.bpe._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertTrue(convergence)
        self.assertEqual(lines[0], 'Declare convergence because cosmax of 0.5 <= options.tol_cosmax_grad of 1')
        self.assertEqual(lines[1], 'Declare convergence because delta_step_len of 1.5 <= options.tol_delta_step of 2')
        self.assertEqual(lines[2], 'Declare convergence because abs(pred_func_change) of 2.5 <= options.tol_delta_cost of 3')

    def test_no_logging(self):
        self.logger.set_level(0)
        with dcs.capture_output() as out:
            convergence = dcs.bpe._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(convergence)
        self.assertEqual(output, '')

#%% _double_dogleg
pass

#%% _dogleg_search
pass

#%% _analyze_results
pass

#%% validate_opti_opts
class Test_validate_opti_opts(unittest.TestCase):
    r"""
    Tests the validate_opti_opts function with the following cases:
        TBD
    """
    def setUp(self):
        self.logger = dcs.Logger(0)
        self.opti_opts = dcs.OptiOpts()
        self.opti_opts.model_func     = str
        self.opti_opts.model_args     = {'a': 1}
        self.opti_opts.cost_func      = str
        self.opti_opts.cost_args      = {'b': 2}
        self.opti_opts.get_param_func = str
        self.opti_opts.set_param_func = repr
        self.opti_opts.output_folder  = ''
        self.opti_opts.output_results = ''
        self.opti_opts.params         = [1, 2]

    def support(self):
        with self.assertRaises(AssertionError):
            dcs.validate_opti_opts(self.opti_opts)

    def test_nominal(self):
        self.logger.set_level(10)
        with dcs.capture_output() as out:
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'******************************\nValidating optimization options.')

    def test_no_logging(self):
        with dcs.capture_output() as out:
            is_valid = dcs.validate_opti_opts(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output,'')

    def test_not_valid1(self):
        self.opti_opts.model_func = None
        self.support()

    def test_not_valid2(self):
        self.opti_opts.model_args = None
        self.support()

    def test_not_valid3(self):
        self.opti_opts.cost_func = None
        self.support()

    def test_not_valid4(self):
        self.opti_opts.cost_args = None
        self.support()

    def test_not_valid5(self):
        self.opti_opts.get_param_func = None
        self.support()

    def test_not_valid6(self):
        self.opti_opts.set_param_func = None
        self.support()

    def test_not_valid7(self):
        self.opti_opts.params = []
        self.support()

    def test_not_valid8(self):
        self.opti_opts.params = None
        self.support()

    def test_not_valid9(self):
        self.opti_opts.slope_method = 'bad_sided'
        self.support()

    def test_not_valid10(self):
        self.opti_opts.search_method = 'wild_ass_guess'
        self.support()

#%% run_bpe
class Test_run_bpe(unittest.TestCase):
    r"""
    Tests the run_bpe function with the following cases:
        TBD
    """
    def setUp(self):
        self.logger = dcs.Logger(10)
        time        = np.arange(251)
        sim_params  = SimParams(time, magnitude=3.5, frequency=12, phase=180)
        truth_time  = np.arange(-10, 201)
        truth_data  = 5 * np.sin(2*np.pi*10*time/1000 + 90*np.pi/180)

        self.opti_opts = dcs.OptiOpts()
        self.opti_opts.model_func     = sim_model
        self.opti_opts.model_args     = {'sim_params': sim_params}
        self.opti_opts.cost_func      = cost_wrapper
        self.opti_opts.cost_args      = {'results_time': time, 'truth_time': truth_time, 'truth_data': truth_data}
        self.opti_opts.get_param_func = get_parameter
        self.opti_opts.set_param_func = set_parameter
        self.opti_opts.output_folder  = ''
        self.opti_opts.output_results = ''
        self.opti_opts.params         = []

        # Parameters to estimate
        self.opti_opts.params.append(dcs.OptiParam('magnitude', best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(dcs.OptiParam('frequency', best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(dcs.OptiParam('phase', best=180, min_=0, max_=360, typical=100, minstep=0.1))

    def test_nominal(self):
        with dcs.capture_output() as out:
            (bpe_results, results) = dcs.run_bpe(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('******************************\nValidating optimization options.'))
        self.assertTrue(isinstance(bpe_results, dcs.BpeResults))
        self.assertTrue(isinstance(results, np.ndarray))

    def test_no_logging(self):
        self.logger.set_level(0)
        with dcs.capture_output() as out:
            dcs.run_bpe(self.opti_opts)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')

    def test_max_likelihood(self):
        self.logger.set_level(0)
        self.opti_opts.is_max_like = True
        dcs.run_bpe(self.opti_opts)

    def test_normalized(self):
        pass # TODO: method not yet coded all the way

    def test_two_sided(self):
        with dcs.capture_output() as out:
            self.opti_opts.slope_method = 'two_sided'
            dcs.run_bpe(self.opti_opts)
        lines = out.getvalue().strip().split('\n')
        out.close()
        for (ix, line) in enumerate(lines):
            if line == 'Running iteration 1.':
                self.assertTrue(lines[ix+1].startswith('  Running model with magnitude'))
                self.assertTrue(lines[ix+2].startswith('  Running model with magnitude'))
                break
        else:
            self.assertTrue(False, 'two sided had issues')
        # rerun at log_level 0
        self.logger.set_level(0)
        dcs.run_bpe(self.opti_opts)

    def test_to_convergence(self):
        self.opti_opts.max_iters = 100
        with dcs.capture_output() as out:
            dcs.run_bpe(self.opti_opts)
        lines = out.getvalue().strip().split('\n')
        out.close()
        for line in lines:
            if line.startswith('Declare convergence'):
                break
        else:
            self.assertTrue(False, "Didn't converge")

    def test_saving(self):
        self.logger.set_level(0)
        self.opti_opts.max_iters = 0
        self.opti_opts.output_folder = dcs.get_tests_dir()
        self.opti_opts.output_results = 'temp_results.hdf5'
        dcs.run_bpe(self.opti_opts)
        # TODO: test with more iterations and files?

    def tearDown(self):
        filename = os.path.join(self.opti_opts.output_folder, self.opti_opts.output_results)
        if os.path.isfile(filename):
            os.remove(filename)

#%% plot_bpe_results
class Test_plot_bpe_results(unittest.TestCase):
    r"""
    Tests the plot_bpe_results function with the following cases:
        TBD
    """
    def setUp(self):
        self.figs = []
        self.logger = dcs.Logger(10)
        self.bpe_results = dcs.BpeResults()
        self.opts = dcs.Opts()
        self.plots = {'innovs': True, 'convergence': True, 'correlation': True, 'info_svd': True, \
        'covariance': True}

    def test_nominal(self):
        # add data
        names = ['a', 'b', 'c', 'd']
        matrix = np.random.rand(4, 4)
        self.bpe_results.param_names  = [x.encode('utf-8') for x in names]
        self.bpe_results.begin_innovs = np.array([1, 2, 3, 4], dtype=float)
        self.bpe_results.final_innovs = np.array([0.5, 0.25, 0.1, 0.05])
        self.bpe_results.costs        = np.array([1, 0.1, 0.05, 0.01])
        self.bpe_results.correlation  = matrix.copy()
        self.bpe_results.info_svd     = matrix.copy()
        self.bpe_results.covariance   = matrix.copy()
        self.figs = dcs.plot_bpe_results(self.bpe_results, plots=self.plots)

    def test_nodata(self):
        with dcs.capture_output() as out:
            self.figs = dcs.plot_bpe_results(self.bpe_results, plots=self.plots)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], "Data isn't available for Innovations plot.")
        self.assertEqual(lines[1], "Data isn't available for convergence plot.")
        self.assertEqual(lines[2], "Data isn't available for correlation plot.")
        self.assertEqual(lines[3], "Data isn't available for information SVD plot.")
        self.assertEqual(lines[4], "Data isn't available for covariance plot.")

    def test_no_logging(self):
        self.logger.set_level(0)
        with dcs.capture_output() as out:
            self.figs = dcs.plot_bpe_results(self.bpe_results, plots=self.plots)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')

    def test_no_plots(self):
        dcs.plot_bpe_results(self.bpe_results, self.opts)

    def test_bad_plot(self):
        with self.assertRaises(ValueError):
            dcs.plot_bpe_results(self.bpe_results, plots={'bad_key': False})

    def test_only_one_key(self):
        dcs.plot_bpe_results(self.bpe_results, plots={'innovs': False})

    def tearDown(self):
        dcs.close_all(self.figs)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
