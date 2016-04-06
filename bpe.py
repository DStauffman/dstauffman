# -*- coding: utf-8 -*-
r"""
BPE module file for the dstauffman code.  It defines the code necessary for doing batch parameter
estimation analysis, and is arbitrary enough to be called with many different modules.

Notes
-----
#.  Written by David C. Stauffer in May 2015 and continued in April 2016.  This work is based loosely
    on prior experience at Lockheed Martin using the GOLF/BPE code with GARSE, but all the numeric
    algorithms are re-coded from external sources to avoid any potential proprietary issues.
"""
# pylint: disable=E1101, C0301, C0326

#%% Imports
from copy import deepcopy
import doctest
import numpy as np
from scipy.linalg import norm
import unittest
from dstauffman.classes import Frozen
from dstauffman.utils import rms

#%% Logger
class Logger(Frozen):
    r"""
    Class that keeps track of the logger options.
    """
    # Logging level
    level = 10

    def __init__(self, **kwargs):
        r"""Creates options instance with ability to override defaults."""
        # override attributes
        for key in kwargs:
            if hasattr(Logger, key):
                setattr(Logger, key, kwargs[key])
            else:
                raise ValueError('Unexpected attribute: {}'.format(key))

    def get_level(self):
        r"""Gets the logging level."""
        return Logger.level

    def set_level(self, level):
        r"""Sets the logging level."""
        if level < 0 or level > 10:
            raise ValueError('Invalid logging level: "{}"'.format(level))
        Logger.level = level

#%% OptiOpts
class OptiOpts(Frozen):
    r"""
    Optimization options for the batch parameter estimator.
    """
    def __init__(self):
        self.model_func     = None
        self.model_args     = None # {}
        self.cost_func      = None
        self.cost_args      = None # {}
        self.get_param_func = None
        self.set_param_func = None
        self.slope_method   = 'one_sided' # or 'two_sided'
        self.max_iters      = 10
        self.params         = None # []

#%% OptiParam
class OptiParam(Frozen):
    r"""
    Optimization parameter to be estimated by the batch parameters estimator.
    """
    def __init__(self, name, best=0., min_=-np.inf, max_=np.inf):
        self.name = name
        self.best = best
        self.min_ = min_
        self.max_ = max_

#%% global logger
logger = Logger()

#%% _function_wrapper
def _function_wrapper(opti_opts, model_args=None, cost_args=None):
    r"""
    Wraps the call to the model function, and returns the results from the model, plus the
    innovations as defined by the given cost function.
    """
    # pull inputs from opti_opts if necessary
    if model_args is None:
        model_args = opti_opts.model_args
    if cost_args is None:
        cost_args = opti_opts.cost_args

    # Run the model to get the results
    results = opti_opts.model_func(**model_args)

    # Run the cost function to get the innovations
    innovs = opti_opts.cost_func(results, **cost_args)

    return (results, innovs)

#%% _calculate_jacobian
def _calculate_jacobian(func, old_param, old_innovs, opti_opts):
    r"""
    Perturbs the state by a litte bit and calculates the numerical slope (i.e. Jacobian approximation)

    Has options for first or second order approximations.
    """
    jacobian = np.array([[]])
    return jacobian

#%% _levenberg_marquardt
def _levenberg_marquardt(jacobian, innovs, lambda_=0):
    r"""
    Classical Levenberg-Marquardt parameter search step

    Parameters
    ----------
    jacobian :
    lambda_  :
    innovs   :

    Returns
    -------
    delta_param : ndarray
        Small step parameter vector

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015 based on Matlab levenberg_marquardt.m function.

    Examples
    --------

    >>> from dstauffman.bpe import _levenberg_marquardt
    >>> import numpy as np
    >>> jacobian = np.array([[1], [2]])
    >>> innovs = np.array([3, 4])
    >>> lamba_ = 0.1
    >>> delta_param = _levenberg_marquardt(jacobian, innovs, lambda_)

    """
    if lambda_ <= 0:
        # calculate this simplified version directly
        delta_param = -np.linalg.inv(jacobian).dot(innovs) # in Matlab: -jacobian\innovs
    else:
        # get the number of parameters
        num_params = jacobian.shape[1]
        # augment the jacobian
        jacobian_aug = np.hstack((jacobian, np.sqrt(lambda_)*np.eye(num_params)))
        # augment the innovations
        innovs_aug = np.hstack((innovs, np.zeros((num_params, 1))))
        # calucalte based on augmented forms
        delta_param = -np.linalg.inv(jacobian_aug).dot(innovs_aug)
    return delta_param

#%% _double_dogleg
def _double_dogleg(delta_param, jacobian, grad_hessian_grad, x_bias, trust_radius):
    r"""
    Compute a double dog-leg parameter search.

    Parameters
    ----------
    delta_param :
        Small change in parameter values
    jacobian :
        Gradient of cost function with respect to parameters
    grad_hessian_grad : float
        g'*Hess*g, units of (cost_func)**3/param**4
    x_bias : float
        shapes the parameter search between Cauchy point and Newton point
    trust_radius : float
        Radius of trusted parameter change region

    Returns
    -------
    new_delta_param
    step_len
    step_scale
    step_type

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015 based on Matlab dbldog.m function.

    Examples
    --------

    >>> from dstauffman.bpe import _double_dogleg
    >>> import numpy as np
    >>> delta_param = np.array([1, 2])
    >>> jacobian = np.array([[3], [4]])
    >>> grad_hessian_grad = 5
    >>> x_bias = 0.1
    >>> trust_radius = 2
    >>> (new_delta_param, step_len, step_scale, step_type) = _double_dogleg(delta_param, \
    ...     jacobian, grad_hessian_grad, x_bias, trust_radius)

    """
    #%% TODO: replace all matrix mults with dots and see about divisions?
    # Calculate some norms
    newton_len   = norm(delta_param)
    gradient_len = norm(jacobian)
    cauchy_len   = gradient_len**3/grad_hessian_grad
    # relaxed_Newton_point is between the initial point and the Newton point
    # If x_bias = 0, the relaxed point is at the Newton point
    relaxed_newton_len = 1 - x_bias*(1 + cauchy_len*gradient_len/(jacobian.T.dot(delta_param)))

    # Compute the minimizing point on the dogleg path
    # This point is inside the trust radius and minimizes the linearized least square function
    if newton_len <= trust_radius:
        # Newton step is inside the trust region so take it
        new_delta_param = delta_param
        step_type       = 'Newton'
        step_scale      = 1
    else:
        # Compute a step somewhere on the dog leg
        if trust_radius / newton_len >= relaxed_newton_len:
            # Step is along the Newton direction and has length equal to trust radius
            step_scale      = trust_radius / newton_len
            step_type       = 'restrained Newton'
            new_delta_param = step_scale*delta_param
        elif cauchy_len > trust_radius:
            # Cauchy step is outside trust region so take gradient step
            step_scale      = trust_radius / cauchy_len
            step_type       = 'gradient'
            new_delta_param = -(trust_radius / gradient_len)*jacobian
        else:
            # Take a dogleg step between relaxed Newton and Cauchy steps
            # This will be on a line between the Cauchy point and the relaxed
            # Newton point such that the distance from the initial point
            # and the restrained point is equal to the trust radius.

            # Cauchy point is at the predicted minimum of the function along the
            # gradient search direction
            cauchy_pt             = -cauchy_len / gradient_len*jacobian
            new_minus_cau         = relaxed_newton_len * delta_param - cauchy_pt
            cau_dot_new_minus_cau = cauchy_pt.T*new_minus_cau
            cau_len_sq            = cauchy_pt.T*cauchy_pt
            new_minus_cau_len_sq  = new_minus_cau.T*new_minus_cau
            tr_sq_minus_cau_sq    = trust_radius**2 - cau_len_sq
            discr                 = np.sqrt(cau_dot_new_minus_cau**2 + \
                new_minus_cau_len_sq*tr_sq_minus_cau_sq)
            # Compute weighting between Newton and Cauchy steps
            # Weighting = 1 gives Newton step
            # Weighting = 0 gives Cauchy step
            if cau_dot_new_minus_cau < 0:
                # angle between Cauchy and Newton-Cauchy is greater than 90 deg
                new_cau_weighting = (discr - cau_dot_new_minus_cau) / new_minus_cau_len_sq;
            else:
                # angle between Cauchy and Newton-Cauchy is less than 90 deg
                new_cau_weighting = tr_sq_minus_cau_sq / (cau_dot_new_minus_cau + discr)
            # save these results
            step_scale      = new_cau_weighting
            step_type       = 'Newton-Cauchy'
            new_delta_param = cauchy_pt + new_cau_weighting*new_minus_cau

    # calculate the new step length for output
    step_len = norm(new_delta_param)

    return (new_delta_param, step_len, step_scale, step_type)

#%% _predict_func_change
def _predict_func_change(delta_param, gradient, hessian):
    r"""
    Predicts change in sum of squared errors function.

    Parameters
    ----------
    delta_param : ndarray (N,)
    gradient : ndarray (1, N)
    hessian : ndarray (N, N)

    Returns
    -------
    delta_func : float
        Predicted change in the squared error function

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015 based on Matlab predict_function_change.m function.

    References
    ----------
    If g is the gradient and H is the Hessian, and d_theta is a set of parameter changes, and the functions
    are linear in parameters theta, then the change in the function is just:

        delta_f = g*d_theta + d_theta'*H*d_theta/2

    However, this is an approximation because the actual functions are usually nonlinear.

    Examples
    --------

    >>> from dstauffman.bpe import _predict_func_change
    >>> import numpy as np
    >>> delta_param = np.array([1, 2])
    >>> gradient = np.array([[3], [4]])
    >>> hessian = np.array([[5, 2], [2, 5]])
    >>> delta_func = _predict_func_change(delta_param, gradient, hessian)

    """
    # expand to 2D matrix if necessary
    if delta_param.ndim == 1:
        delta_param = np.expand_dims(delta_param, axis=1)
    # calculate and return the predicted change
    delta_func = gradient.T.dot(delta_param) + np.dot(delta_param.T.dot(hessian), delta_param)/2
    assert len(delta_func) == 1
    return delta_func.flatten()[0]

#%% _analyze_results
def _analyze_results(innovs_final, log_level=10):
    if log_level >= 5:
        print('Analyzing final results.')
    bpe_results = dict()
    return bpe_results

#%% validate_opti_opts
def validate_opti_opts(opti_opts):
    r"""
    Validates that the optimization options are valid.

    Parameters
    ----------
    opti_opts : class OptiOpts
        Optimization options

    Examples
    --------

    >>> from dstauffman import OptiOpts, OptiParams, validate_opti_opts
    >>> opti_opts = OptiOpts()
    >>> opti_opts.params = [OptiParam('param.life.age_calibration', 1.0, 0., 10.)]
    >>> is_valid = validate_opti_opts(opti_opts)
    Validating optimization options.

    >>> print(is_valid)
    True

    """
    # get the logging level
    log_level = Logger().get_level()
    # display some information
    if log_level >= 5:
        print('Validating optimization options.')
    # Must have specified all parameters
    assert callable(opti_opts.model_func)
    assert isinstance(opti_opts.model_args, dict)
    assert callable(opti_opts.cost_func)
    assert isinstance(opti_opts.cost_args, dict)
    assert callable(opti_opts.get_param_func)
    assert callable(opti_opts.set_param_func)
    # Must be one of these two slope methods
    assert opti_opts.slope_method in {'one_sided', 'two_sided'}
    # Must estimate at least one parameter (TODO: make work with zero?)
    assert isinstance(opti_opts.params, list) and len(opti_opts.params) > 0
    # Return True to signify that everything validated correctly
    return True

#%% run_bpe
def run_bpe(opti_opts):
    r"""
    Runs the batch parameter estimator with the given model optimization options.

    Parameters
    ----------
    opti_opts : class OptiOpts
        estimation options

    Returns
    -------
    bpe_results : dict
        BPE results as a dictionary with the following keys:
            TBD
    results : class Results
        Results of the model using the final set of parameters as determined by BPE

    Notes
    -----
    #.   Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman import run_bpe, Parameters, OptiOpts #TODO: finish this

    """
    # alias the log level
    log_level = Logger().get_level()

    # check for valid parameters
    validate_opti_opts(opti_opts)

    # create a working copy of the model inputs
    model_args = deepcopy(opti_opts.model_args)

    # alias the parameter getter and setters
    getter = opti_opts.get_param_func
    setter = opti_opts.set_param_func

    # initialize counters
    iter_count = 0

    # run the model
    if log_level >= 2:
        print('Running initial simulation.')
    (_, innovs_start) = _function_wrapper(opti_opts, model_args)
    iter_count += 1

    # initialize costs
    first_cost = 0.5 * rms(innovs_start**2, ignore_nans=True)
    prev_cost  = first_cost
    best_cost  = first_cost

    # Do some stuff
    while iter_count <= opti_opts.max_iters:
        # update status
        if log_level >= 2:
            print('Running iteration {}.'.format(iter_count))

        # run finite differences code to numerically approximate the Jacobian matrix
        jacobian = _calculate_jacobian(func, old_param, old_innovs, opti_opts)

        # calculate the numerical gradient with respect to the estimated parameters
        grad_innovs = jacobian.T.dot(innovs)

        # calculate the hessian matrix
        hessian = jacobian.T.dot(jacobian)

        # calculate the delta parameter step to try on the next iteration
        delta_param = _levenberg_marquardt(jacobian, old_innovs, lambda_=0)

        # check for convergence conditions
        pass # TODO: do this

        # search for parameter set that is better than the current set
        pass # TODO: do this
        # new_param = dogleg_search(old_param)

        # increment counter
        iter_count += 1

    # Run for final time
    if log_level >= 2:
        print('Running final simulation.')
    (results, innovs_final) = _function_wrapper(opti_opts, model_args)

    # analyze BPE results
    bpe_results = _analyze_results(innovs_final, log_level=log_level)

    return (bpe_results, results)

#%% plot_bpe_results
def plot_bpe_results(batch, param, opts):
    pass

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_bpe', exit=False)
    doctest.testmod(verbose=False)
