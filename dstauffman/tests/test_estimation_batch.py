r"""
Test file for the `batch` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Renamed from bpe.py to estimation.py by David C. Stauffer in May 2020.
"""

#%% Imports
import logging
import os
import unittest
from unittest.mock import patch

import numpy as np

from dstauffman import capture_output, compare_two_classes, get_tests_dir, Frozen, LogLevel, rss
from dstauffman.plotting import close_all, Opts, Plotter

import dstauffman.estimation as estm

#%% Hard-coded values
plotter = Plotter(False)

#%% Setup for testing
# Classes - SimParams
class SimParams(Frozen):
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
    ix_truth     = np.flatnonzero((truth_time >= results_time[0] - precision) & (truth_time <= \
        results_time[-1] + precision))
    # find the indices to results (in case truth isn't long enough)
    ix_results   = np.flatnonzero(results_time <= truth_time[-1] + precision)
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

#%% estimation.OptiOpts
class Test_estimation_OptiOpts(unittest.TestCase):
    r"""
    Tests the estimation.OptiOpts class with the following cases:
        Initialization
        Equality
        Inequality
    """
    def test_init(self):
        opti_opts = estm.OptiOpts()
        self.assertTrue(isinstance(opti_opts, estm.OptiOpts))

    def test_equality(self):
        opti_opts1 = estm.OptiOpts()
        opti_opts2 = estm.OptiOpts()
        self.assertEqual(opti_opts1, opti_opts2)

    def test_inequality(self):
        opti_opts1 = estm.OptiOpts()
        opti_opts2 = estm.OptiOpts()
        opti_opts2.grow_radius = 5.5
        self.assertNotEqual(opti_opts1, opti_opts2)

    def test_inequality2(self):
        opti_opts = estm.OptiOpts()
        self.assertNotEqual(opti_opts, 2)

    def test_pprint(self):
        opti_opts = estm.OptiOpts()
        with capture_output() as out:
            opti_opts.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'OptiOpts')
        self.assertEqual(lines[1], ' model_func      = None')
        self.assertEqual(lines[-1],' trust_radius    = 1.0')

#%% estimation.OptiParam
class Test_estimation_OptiParam(unittest.TestCase):
    r"""
    Tests the estimation.OptiParam class with the following cases:
        Initialization
        Equality
        Inequality
        Get array (x5)
        Get names
    """
    def test_init(self):
        opti_param = estm.OptiParam('test')
        self.assertTrue(isinstance(opti_param, estm.OptiParam))

    def test_equality(self):
        opti_param1 = estm.OptiParam('test')
        opti_param2 = estm.OptiParam('test')
        self.assertEqual(opti_param1, opti_param2)

    def test_inequality(self):
        opti_param1 = estm.OptiParam('test')
        opti_param2 = estm.OptiParam('test')
        opti_param2.min_ = 5.5
        self.assertNotEqual(opti_param1, opti_param2)

    def test_inequality2(self):
        opti_param = estm.OptiParam('test')
        self.assertNotEqual(opti_param, 2)

    def test_get_array(self):
        opti_param = estm.OptiParam('test')
        params = [opti_param, opti_param]
        best = estm.OptiParam.get_array(params)
        np.testing.assert_array_equal(best, np.array([np.nan, np.nan]))

    def test_get_array2(self):
        opti_param = estm.OptiParam('test')
        params = [opti_param, opti_param]
        values = estm.OptiParam.get_array(params, type_='min')
        np.testing.assert_array_equal(values, np.array([-np.inf, -np.inf]))

    def test_get_array3(self):
        opti_param = estm.OptiParam('test')
        params = [opti_param, opti_param]
        values = estm.OptiParam.get_array(params, type_='max')
        np.testing.assert_array_equal(values, np.array([np.inf, np.inf]))

    def test_get_array4(self):
        opti_param = estm.OptiParam('test')
        params = [opti_param, opti_param]
        for this_type in ['best', 'min', 'min_', 'max', 'max_', 'minstep', 'typical']:
            # just test that these all execute, but don't worry about values
            estm.OptiParam.get_array(params, type_=this_type)

    def test_get_array5(self):
        opti_param = estm.OptiParam('test')
        params = [opti_param, opti_param]
        with self.assertRaises(ValueError):
            estm.OptiParam.get_array(params, type_='bad_name')

    def test_get_names(self):
        opti_param1 = estm.OptiParam('test1')
        opti_param2 = estm.OptiParam('test2')
        params = [opti_param1, opti_param2]
        names = estm.OptiParam.get_names(params)
        self.assertEqual(names, ['test1', 'test2'])

    def test_pprint(self):
        opti_param = estm.OptiParam('test')
        with capture_output() as out:
            opti_param.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'OptiParam')
        self.assertEqual(lines[1], ' name    = test')
        self.assertEqual(lines[2], ' best    = nan')
        self.assertEqual(lines[3], ' min_    = -inf')
        self.assertEqual(lines[4], ' max_    = inf')
        self.assertEqual(lines[5], ' minstep = 0.0001')
        self.assertEqual(lines[6], ' typical = 1.0')

#%% estimation.BpeResults
class Test_estimation_BpeResults(unittest.TestCase):
    r"""
    Tests the estimation.BpeResults class with the following cases:
        Save (HDF5)
        Load (HDF5)
        str method
        pprint method
    """
    def setUp(self):
        self.bpe_results = estm.BpeResults()
        self.bpe_results.num_evals = 5
        self.filename    = os.path.join(get_tests_dir(), 'test_estimation_results.hdf5')
        self.filename2   = self.filename.replace('hdf5', 'pkl')
        estm.batch.logger.setLevel(LogLevel.L0)

    def test_save(self):
        self.bpe_results.save(self.filename)
        self.assertTrue(os.path.isfile(self.filename))

    def test_save2(self):
        self.bpe_results.save(self.filename, use_hdf5=False)
        self.assertTrue(os.path.isfile(self.filename2))

    def test_load(self):
        self.bpe_results.save(self.filename)
        bpe_results = estm.BpeResults.load(self.filename)
        self.assertTrue(compare_two_classes(bpe_results, self.bpe_results, suppress_output=True))

    def test_load2(self):
        self.bpe_results.save(self.filename, use_hdf5=False)
        bpe_results = estm.BpeResults.load(self.filename, use_hdf5=False)
        self.assertTrue(compare_two_classes(bpe_results, self.bpe_results, suppress_output=True))

    def test_str(self):
        with capture_output() as out:
            print(self.bpe_results)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'BpeResults')
        self.assertTrue(lines[1].startswith('  begin_params = None'))

    def test_pprint(self):
        self.bpe_results.param_names  = ['a'.encode('utf-8')]
        self.bpe_results.begin_params = [1]
        self.bpe_results.final_params = [2]
        with capture_output() as out:
            self.bpe_results.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Initial cost: None')
        self.assertEqual(lines[1], 'Initial parameters:')
        self.assertEqual(lines[2].strip(), 'a = 1')
        self.assertEqual(lines[3], 'Final cost: None')
        self.assertEqual(lines[4], 'Final parameters:')
        self.assertEqual(lines[5].strip(), 'a = 2')

    def test_pprint2(self):
        with capture_output() as out:
            self.bpe_results.pprint()
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        if os.path.isfile(self.filename2):
            os.remove(self.filename2)

#%% estimation.CurrentResults
class Test_estimation_CurrentResults(unittest.TestCase):
    r"""
    Tests the estimation.CurrentResults class with the following cases:
        Printing
    """
    def setUp(self):
        self.current_results = estm.CurrentResults()

    def test_printing(self):
        with capture_output() as out:
            print(self.current_results)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'Current Results:')
        self.assertEqual(lines[1], '  Trust Radius: None')
        self.assertEqual(lines[2], '  Best Cost: None')
        self.assertEqual(lines[3], '  Best Params: None')

#%% estimation._print_divider
@patch('dstauffman.estimation.batch.logger')
class Test_estimation__print_divider(unittest.TestCase):
    r"""
    Tests the estimation._print_divider function with the following cases:
        With new line
        Without new line
    """
    def setUp(self):
        self.output = '******************************'

    def test_with_new_line(self, mock_logger):
        estm.batch.logger.setLevel(LogLevel.L5)
        estm.batch._print_divider()
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L5, ' ')
        mock_logger.log.assert_any_call(LogLevel.L5, '******************************')

    def test_no_new_line(self, mock_logger):
        estm.batch.logger.setLevel(LogLevel.L8)
        estm.batch._print_divider(new_line=False)
        mock_logger.log.assert_called_with(LogLevel.L5, '******************************')

    def test_alternative_level(self, mock_logger):
        estm.batch.logger.setLevel(LogLevel.L2)
        estm.batch._print_divider(level=LogLevel.L0)
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L0, ' ')
        mock_logger.log.assert_any_call(LogLevel.L0, '******************************')

    def test_not_logging(self, mock_logger):
        estm.batch.logger.setLevel(LogLevel.L2)
        estm.batch._print_divider()
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L5, ' ')
        mock_logger.log.assert_any_call(LogLevel.L5, '******************************')
        # TODO: how to test that this wouldn't log anything?

#%% estimation._function_wrapper
class Test_estimation__function_wrapper(unittest.TestCase):
    r"""
    Tests the estimation._function_wrapper function with the following cases:
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
        (innovs, results) = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, return_results=True)
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_model_args(self):
        (innovs, results) = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, model_args={'a': 5}, return_results=True)
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_cost_args(self):
        (innovs, results) = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, cost_args={'a': 5}, return_results=True)
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_innov_only(self):
        innovs = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, cost_args={'a': 5})
        np.testing.assert_array_equal(innovs, self.innovs)

#%% estimation._finite_differences
@patch('dstauffman.estimation.batch.logger')
class Test_estimation__finite_differences(unittest.TestCase):
    r"""
    Tests the estimation._finite_differences function with the following cases:
        Nominal
        Normalized
    """
    def setUp(self):
        estm.batch.logger.setLevel(LogLevel.L5)
        time        = np.arange(251)
        sim_params  = SimParams(time, magnitude=3.5, frequency=12, phase=180)
        truth_time  = np.arange(-10, 201)
        truth_data  = 5 * np.sin(2*np.pi*10*time/1000 + 90*np.pi/180)

        self.opti_opts                = estm.OptiOpts()
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
        self.opti_opts.params.append(estm.OptiParam('magnitude', best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('frequency', best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('phase', best=180, min_=0, max_=360, typical=100, minstep=0.1))

        self.model_args = self.opti_opts.model_args

        self.bpe_results = estm.BpeResults()
        self.cur_results = estm.CurrentResults()

        # initialize current results
        self.cur_results.innovs    = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, self.model_args)
        self.cur_results.trust_rad = self.opti_opts.trust_radius
        self.cur_results.cost      = 0.5 * rss(self.cur_results.innovs, ignore_nans=True)
        names = estm.OptiParam.get_names(self.opti_opts.params)
        self.cur_results.params    = self.opti_opts.get_param_func(names=names, **self.model_args)

        # set relevant results variables
        self.bpe_results.param_names  = [name.encode('utf-8') for name in names]
        self.bpe_results.begin_params = self.cur_results.params.copy()
        self.bpe_results.begin_innovs = self.cur_results.innovs.copy()
        self.bpe_results.begin_cost   = self.cur_results.cost
        self.bpe_results.costs.append(self.cur_results.cost)

        self.two_sided = False
        self.normalized = False

    def test_nominal(self, mock_logger):
        (jacobian, gradient, hessian) = estm.batch._finite_differences(self.opti_opts, self.model_args, self.bpe_results, \
            self.cur_results, two_sided=self.two_sided, normalized=self.normalized)
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3, ))
        self.assertEqual(hessian.shape, (3, 3))

    def test_normalized(self, mock_logger):
        self.normalized = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(self.opti_opts, self.model_args, self.bpe_results, \
            self.cur_results, two_sided=self.two_sided, normalized=self.normalized)
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3, ))
        self.assertEqual(hessian.shape, (3, 3))

    def test_two_sided(self, mock_logger):
        self.two_sided = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(self.opti_opts, self.model_args, self.bpe_results, \
            self.cur_results, two_sided=self.two_sided, normalized=self.normalized)
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3, ))
        self.assertEqual(hessian.shape, (3, 3))

    def test_norm_and_two_sided(self, mock_logger):
        self.normalized = True
        self.two_sided = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(self.opti_opts, self.model_args, self.bpe_results, \
            self.cur_results, two_sided=self.two_sided, normalized=self.normalized)
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3, ))
        self.assertEqual(hessian.shape, (3, 3))

#%% estimation._levenberg_marquardt
class Test_estimation__levenberg_marquardt(unittest.TestCase):
    r"""
    Tests the estimation._levenberg_marquardt function with the following cases:
        with lambda_
        without lambda_
    """
    def setUp(self):
        self.jacobian    = np.array([[1, 2], [3, 4], [5, 6]])
        self.innovs      = np.array([7, 8, 9])
        self.lambda_     = 5
        self.delta_param = np.array([-0.46825397, -1.3015873])

    def test_nominal(self):
        delta_param = estm.batch._levenberg_marquardt(self.jacobian, self.innovs, self.lambda_)
        np.testing.assert_array_almost_equal(delta_param, self.delta_param)

    def test_lambda_zero(self):
        b = -np.linalg.pinv(self.jacobian).dot(self.innovs)
        delta_param = estm.batch._levenberg_marquardt(self.jacobian, self.innovs, 0)
        np.testing.assert_array_almost_equal(delta_param, b)

#%% estimation._predict_func_change
class Test_estimation__predict_func_change(unittest.TestCase):
    r"""
    Tests the estimation._predict_func_change function with the following cases:
        Nominal
    """
    def setUp(self):
        self.delta_param = np.array([1, 2])
        self.gradient    = np.array([3, 4])
        self.hessian     = np.array([[5, 2], [2, 5]])
        self.pred_change = 27.5

    def test_nominal(self):
        delta_func = estm.batch._predict_func_change(self.delta_param, self.gradient, self.hessian)
        self.assertEqual(delta_func, self.pred_change)

#%% estimation._check_for_convergence
@patch('dstauffman.estimation.batch.logger')
class Test_estimation__check_for_convergence(unittest.TestCase):
    r"""
    Tests the estimation._check_for_convergence function with the following cases:
        TBD
    """
    def setUp(self):
        estm.batch.logger.setLevel(LogLevel.L5)
        self.opti_opts        = type('Class1', (object, ), {'tol_cosmax_grad': 1, 'tol_delta_step': 2, \
            'tol_delta_cost': 3})
        self.cosmax           = 10
        self.delta_step_len   = 10
        self.pred_func_change = 10

    def test_not_converged(self, mock_logger):
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, self.pred_func_change)
        self.assertFalse(convergence)

    def test_convergence1(self, mock_logger):
        convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, self.delta_step_len, self.pred_func_change)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L3, 'Declare convergence because cosmax of 0.5 <= options.tol_cosmax_grad of 1')

    def test_convergence2(self, mock_logger):
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, 1.5, self.pred_func_change)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L3, 'Declare convergence because delta_step_len of 1.5 <= options.tol_delta_step of 2')

    def test_convergence3(self, mock_logger):
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, -2.5)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L3, 'Declare convergence because abs(pred_func_change) of 2.5 <= options.tol_delta_cost of 3')

    def test_convergence4(self, mock_logger):
        convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        self.assertTrue(convergence)
        self.assertEqual(mock_logger.log.call_count, 3)
        mock_logger.log.assert_any_call(LogLevel.L3, 'Declare convergence because cosmax of 0.5 <= options.tol_cosmax_grad of 1')
        mock_logger.log.assert_any_call(LogLevel.L3, 'Declare convergence because delta_step_len of 1.5 <= options.tol_delta_step of 2')
        mock_logger.log.assert_any_call(LogLevel.L3, 'Declare convergence because abs(pred_func_change) of 2.5 <= options.tol_delta_cost of 3')

    def test_no_logging(self, mock_logger):
        mock_logger.setLevel(logging.NOTSET) # CRITICAL
        with capture_output('err') as err:
            convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        self.assertTrue(convergence)
        error = err.getvalue().strip()
        self.assertEqual(error, '')

#%% estimation._double_dogleg
class Test_estimation__double_dogleg(unittest.TestCase):
    r"""
    Tests the estimation._double_dogleg function with the following cases:
        TBD
    """
    def setUp(self):
        self.delta_param = np.array([1, 2])
        self.gradient = np.array([3, 4])
        self.grad_hessian_grad = 5
        self.x_bias = 0.1
        self.trust_radius = 2

    def test_large_trust_radius(self):
        # Newton step in trust radius
        self.trust_radius = 10000
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(self.delta_param, \
             self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius)

    def test_small_bias(self):
        # Newton step outside trust_radius
        self.x_bias = 0.01
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(self.delta_param, \
             self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius)

    def test_gradient_step(self):
        # Newton step outside trust_radius
        self.x_bias = 0.001
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(self.delta_param, \
             self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius)

    def test_dogleg1(self):
        # Dogleg step 1
        self.x_bias = 0.001
        self.grad_hessian_grad = 75
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(self.delta_param, \
             self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius)

    def test_dogleg2(self):
        # Dogleg step 2
        self.x_bias = 0.001
        self.grad_hessian_grad = 75
        self.delta_param = 0.001 * np.array([1, 2])
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(self.delta_param, \
             self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius)

#%% estimation._dogleg_search
@patch('dstauffman.estimation.batch.logger')
class Test_estimation__dogleg_search(unittest.TestCase):
    r"""
    Tests the estimation._dogleg_search function with the following cases:
        TBD
    """
    def setUp(self):
        estm.batch.logger.setLevel(LogLevel.L5)
        time        = np.arange(251)
        sim_params  = SimParams(time, magnitude=3.5, frequency=12, phase=180)
        truth_time  = np.arange(-10, 201)
        truth_data  = 5 * np.sin(2*np.pi*10*time/1000 + 90*np.pi/180)

        self.opti_opts                = estm.OptiOpts()
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
        self.opti_opts.params.append(estm.OptiParam('magnitude', best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('frequency', best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('phase', best=180, min_=0, max_=360, typical=100, minstep=0.1))

        self.model_args = self.opti_opts.model_args

        self.bpe_results = estm.BpeResults()
        self.cur_results = estm.CurrentResults()

        # initialize current results
        self.cur_results.innovs    = estm.batch._function_wrapper(self.opti_opts, self.bpe_results, self.model_args)
        self.cur_results.trust_rad = self.opti_opts.trust_radius
        self.cur_results.cost      = 0.5 * rss(self.cur_results.innovs, ignore_nans=True)
        names = estm.OptiParam.get_names(self.opti_opts.params)
        self.cur_results.params    = self.opti_opts.get_param_func(names=names, **self.model_args)

        # set relevant results variables
        self.bpe_results.param_names  = [name.encode('utf-8') for name in names]
        self.bpe_results.begin_params = self.cur_results.params.copy()
        self.bpe_results.begin_innovs = self.cur_results.innovs.copy()
        self.bpe_results.begin_cost   = self.cur_results.cost
        self.bpe_results.costs.append(self.cur_results.cost)

        self.delta_param = np.array([1, 2, 3])
        self.gradient    = np.array([4, 5, 6])
        self.hessian     = np.array([[5, 2, 1], [1, 2, 5], [3, 3, 3]])
        self.jacobian    = np.random.rand(201, 3)
        self.normalized  = False

    def test_nominal(self, mock_logger):
        estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
            self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

    def test_normalized(self, mock_logger):
        self.normalized = True
        estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
            self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

    def test_levenberg_marquardt(self, mock_logger):
        self.opti_opts.search_method = 'levenberg_marquardt'
        estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
            self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

    def test_bad_method(self, mock_logger):
        self.opti_opts.search_method = 'bad_method'
        with self.assertRaises(ValueError):
            estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
                self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

    def test_minimums(self, mock_logger):
        self.opti_opts.params[0].min_ = 10
        estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
            self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

    def test_huge_trust_radius(self, mock_logger):
        # TODO: figure out how to get this to shrink a Newton step.
        self.opti_opts.trust_radius = 1000000
        estm.batch._dogleg_search(self.opti_opts, self.opti_opts.model_args, self.bpe_results, self.cur_results, \
            self.delta_param, self.jacobian, self.gradient, self.hessian, normalized=self.normalized)

#%% estimation._analyze_results
@patch('dstauffman.estimation.batch.logger')
class Test_estimation__analyze_results(unittest.TestCase):
    r"""
    Tests the estimation._analyze_results function with the following cases:
        Nominal
        Normalized
    """
    def setUp(self):
        self.opti_opts = estm.OptiOpts()
        self.opti_opts.params = [estm.OptiParam('a'), estm.OptiParam('b')]
        self.bpe_results = estm.BpeResults()
        self.bpe_results.param_names = [x.encode('utf-8') for x in ['a', 'b']]
        self.jacobian = np.array([[1, 2], [3, 4], [5, 6]])
        self.normalized = False

    def test_nominal(self, mock_logger):
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)

    def test_normalized(self, mock_logger):
        self.normalized = True
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)

    def test_no_iters(self, mock_logger):
        self.opti_opts.max_iters = 0
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)

#%% estimation.validate_opti_opts
@patch('dstauffman.estimation.batch.logger')
class Test_estimation_validate_opti_opts(unittest.TestCase):
    r"""
    Tests the estimation.validate_opti_opts function with the following cases:
        TBD
    """
    def setUp(self):
        estm.batch.logger.setLevel(LogLevel.L5)
        self.opti_opts = estm.OptiOpts()
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
            estm.validate_opti_opts(self.opti_opts)

    def test_nominal(self, mock_logger):
        is_valid = estm.validate_opti_opts(self.opti_opts)
        self.assertTrue(is_valid)
        mock_logger.log.assert_any_call(LogLevel.L5, '******************************')
        mock_logger.log.assert_any_call(LogLevel.L5, 'Validating optimization options.')

    def test_no_logging(self, mock_logger):
        estm.batch.logger.setLevel(LogLevel.L3)
        is_valid = estm.validate_opti_opts(self.opti_opts)
        self.assertTrue(is_valid)
        mock_logger.log.assert_any_call(LogLevel.L5, '******************************')
        mock_logger.log.assert_any_call(LogLevel.L5, 'Validating optimization options.')

    def test_not_valid1(self, mock_logger):
        self.opti_opts.model_func = None
        self.support()

    def test_not_valid2(self, mock_logger):
        self.opti_opts.model_args = None
        self.support()

    def test_not_valid3(self, mock_logger):
        self.opti_opts.cost_func = None
        self.support()

    def test_not_valid4(self, mock_logger):
        self.opti_opts.cost_args = None
        self.support()

    def test_not_valid5(self, mock_logger):
        self.opti_opts.get_param_func = None
        self.support()

    def test_not_valid6(self, mock_logger):
        self.opti_opts.set_param_func = None
        self.support()

    def test_not_valid7(self, mock_logger):
        self.opti_opts.params = []
        self.support()

    def test_not_valid8(self, mock_logger):
        self.opti_opts.params = None
        self.support()

    def test_not_valid9(self, mock_logger):
        self.opti_opts.slope_method = 'bad_sided'
        self.support()

    def test_not_valid10(self, mock_logger):
        self.opti_opts.search_method = 'wild_ass_guess'
        self.support()

#%% estimation.run_bpe
@patch('dstauffman.estimation.batch.logger')
class Test_estimation_run_bpe(unittest.TestCase):
    r"""
    Tests the estimation.run_bpe function with the following cases:
        TBD
    """
    def setUp(self):
        estm.batch.logger.setLevel(LogLevel.L5)
        time        = np.arange(251)
        sim_params  = SimParams(time, magnitude=3.5, frequency=12, phase=180)
        truth_time  = np.arange(-10, 201)
        truth_data  = 5 * np.sin(2*np.pi*10*time/1000 + 90*np.pi/180)

        self.opti_opts                = estm.OptiOpts()
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
        self.opti_opts.params.append(estm.OptiParam('magnitude', best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('frequency', best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam('phase', best=180, min_=0, max_=360, typical=100, minstep=0.1))

    def test_nominal(self, mock_logger):
        mock_logger.level = LogLevel.L5
        (bpe_results, results) = estm.run_bpe(self.opti_opts)
        # TODO: check logging results?
        self.assertTrue(isinstance(bpe_results, estm.BpeResults))
        self.assertTrue(isinstance(results, np.ndarray))

    def test_no_logging(self, mock_logger):
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)
        # TODO: check logging results

    def test_max_likelihood(self, mock_logger):
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        self.opti_opts.is_max_like = True
        estm.run_bpe(self.opti_opts)

    def test_normalized(self, mock_logger):
        pass # TODO: method not yet coded all the way

    def test_two_sided(self, mock_logger):
        mock_logger.level = LogLevel.L5
        self.opti_opts.slope_method = 'two_sided'
        estm.run_bpe(self.opti_opts)
#        for (ix, line) in enumerate(lines):
#            if line == 'Running iteration 1.':
#                self.assertTrue(lines[ix+1].startswith('  Running model with magnitude'))
#                self.assertTrue(lines[ix+2].startswith('  Running model with magnitude'))
#                break
#        else:
#            self.assertTrue(False, 'two sided had issues')
        # rerun with no logging
        estm.batch.logger.setLevel(logging.CRITICAL)
        estm.run_bpe(self.opti_opts)

    def test_to_convergence(self, mock_logger):
        self.opti_opts.max_iters = 100
        mock_logger.level = LogLevel.L5
        estm.run_bpe(self.opti_opts)
#        for line in lines:
#            if line.startswith('Declare convergence'):
#                break
#        else:
#            self.assertTrue(False, "Didn't converge")

    def test_saving(self, mock_logger):
        mock_logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        self.opti_opts.max_iters = 1
        self.opti_opts.output_folder = get_tests_dir()
        self.opti_opts.output_results = 'temp_results.hdf5'
        estm.run_bpe(self.opti_opts)
        # TODO: test with more iterations and files?

    def test_startup_finish_funcs(self, mock_logger):
        self.opti_opts.start_func = lambda sim_params: dict()
        self.opti_opts.final_func = lambda sim_params, settings: None
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)

    def test_failed(self, mock_logger):
        # TODO: this case doesn't fail yet.  Make it do so.
        self.opti_opts.max_iters = 1
        self.opti_opts.tol_delta_step = 100
        self.opti_opts.step_limit = 1
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)

    def tearDown(self):
        if self.opti_opts.output_results:
            files = [self.opti_opts.output_results, 'bpe_results_iter_1.hdf5', 'cur_results_iter_1.hdf5']
            for this_file in files:
                filename = os.path.join(self.opti_opts.output_folder, this_file)
                if os.path.isfile(filename):
                    os.remove(filename)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
