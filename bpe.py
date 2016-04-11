# -*- coding: utf-8 -*-
r"""
BPE module file for the dstauffman code.  It defines the code necessary for doing batch parameter
estimation analysis, and is arbitrary enough to be called with many different modules.

Notes
-----
#.  Written by David C. Stauffer in May 2015 and continued in April 2016.  This work is based
    loosely on prior experience at Lockheed Martin using the GOLF/BPE code with GARSE, but all the
    numeric algorithms are re-coded from external sources to avoid any potential proprietary issues.
"""
# pylint: disable=E1101, C0301, C0326

#%% Imports
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

    Parameters
    ----------
    level : int
        Level of logger

    Examples
    --------
    >>> from dstauffman import Logger
    >>> logger = Logger(5)
    >>> print(logger)
    Logger(5)

    """
    # class attribute for Logging level
    level = 10

    def __init__(self, level=None):
        r"""Creates options instance with ability to override defaults."""
        if level is not None:
            type(self).level = self._check_level(level)

    def __str__(self):
        r"""Prints the current level."""
        return '{}({})'.format(type(self).__name__, self.level)

    @staticmethod
    def _check_level(level):
        r"""Checks for a valid logging level."""
        if level < 0 or level > 10:
            raise ValueError('Invalid logging level: "{}"'.format(level))
        return level

    @classmethod
    def get_level(cls):
        r"""Gets the logging level."""
        return cls.level

    @ classmethod
    def set_level(cls, level):
        r"""Sets the logging level."""
        cls.level = cls._check_level(level)

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
        self.tol_cosmax_gradient = 1e-4
        self.tol_delta_step = 1e-20
        self.tol_delta_cost = 1e-20
        self.params         = None # []

#%% OptiParam
class OptiParam(Frozen):
    r"""
    Optimization parameter to be estimated by the batch parameters estimator.
    """
    def __init__(self, name, best=0., min_=-np.inf, max_=np.inf, typical=1.):
        self.name = name
        self.best = best
        self.min_ = min_
        self.max_ = max_
        self.typical = typical

    @staticmethod
    def get_array(opti_param, type_='best'):
        r"""
        Gets a numpy vector of all the optimization parameters for the desired type.
        """
        # check for valid types
        if type_ in ['best', 'min_', 'max_', 'typical']:
            key = type_
        elif type_ in ['min', 'max']:
            key = type_[:-1]
        else:
            raise ValueError('Unexpected type of "{}"'.format(type_))
        # pull out the data
        out = np.array([getattr(x, key) for x in opti_param])
        return out

    @staticmethod
    def get_names(opti_param):
        r"""
        Gets the names of the optimization parameters as a list.
        """
        names = [x.name for x in opti_param]
        return names

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

    # Set any NaNs to zero so that they are ignored
    innovs[np.isnan(innovs)] = 0

    return (results, innovs)

#%% _calculate_jacobian
def _calculate_jacobian(opti_opts, old_params, old_innovs, *, two_sided=False, normalized=False, perturb_fact=None):
    r"""
    Perturbs the state by a litte bit and calculates the numerical slope (i.e. Jacobian approximation)

    Has options for first or second order approximations.

    References
    ----------
    #.  Conn, Andrew R., Gould, Nicholas, Toint, Philippe, "Trust-Region Methods," MPS-SIAM Series
        on Optimization, 2000.

    """
    # alias useful values
    sqrt_eps      = np.sqrt(np.finfo(float).eps)
    num_param     = old_params.size
    num_innov     = old_innovs.size
    param_typical = OptiParam.get_array(opti_opts.params, type_='typical')
    param_signs   = np.sign(old_params)
    param_signs[param_signs == 0] = 1
    names         = OptiParam.get_names(opti_opts.params)

    # initialize output
    jacobian = np.zeros((num_innov, num_param), dtype=float)

    # initialize loop variables
    # set parameter pertubation (Reference 1, section 8.4.3)
    if normalized:
        param_perturb = perturb_fact * sqrt_eps * param_signs # TODO: how to get perturb_fact?
    else:
        param_perturb = sqrt_eps * param_signs * np.maximum(np.abs(old_params), np.abs(param_typical))

    param_current_plus  = old_params.copy()
    param_current_minus = old_params.copy()

    for i_param in range(num_param):
        # update the parameters for this run
        param_current_plus[i_param]  = old_params[i_param] + param_perturb[i_param]
        param_current_minus[i_param] = old_params[i_param] - param_perturb[i_param]

        # get the new parameters for this run
        if normalized:
            new_params = param_current_plus * param_typical
        else:
            new_params = param_current_plus.copy()

        # call model with new parameters
        opti_opts.set_param_func(names, new_params, **opti_opts.model_args)
        (_, new_innovs) = _function_wrapper(opti_opts)

        if two_sided:
            if normalized:
                new_params = param_current_minus * param_typical
            else:
                new_params = param_current_minus.copy()
            opti_opts.set_param_func(names, new_params, **opti_opts.model_args)
            (_, new_innovs_minus) = _function_wrapper(opti_opts)

        # compute the jacobian
        if two_sided:
            jacobian[:, i_param] = 0.5 * (new_innovs - new_innovs_minus) / param_perturb[i_param]
            #grad_log_det_b[i_param] = 0.25 *
        else:
            jacobian[:, i_param] = (new_innovs - old_innovs) / param_perturb[i_param]

        # reset the parameter to the original value for next loop
        param_current_plus[i_param]  = old_params[i_param]
        param_current_minus[i_param] = old_params[i_param]

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
    >>> jacobian    = np.array([[1, 2], [3, 4], [5, 6]])
    >>> innovs      = np.array([7, 8, 9])
    >>> lambda_     = 5
    >>> delta_param = _levenberg_marquardt(jacobian, innovs, lambda_)
    >>> print(delta_param)
    [-0.46825397 -1.3015873 ]

    """
    if lambda_ <= 0:
        # calculate this simplified version directly
        delta_param = -np.linalg.lstsq(jacobian, innovs)[0] # in Matlab: -jacobian\innovs
    else:
        # get the number of parameters
        num_params = jacobian.shape[1]
        # augment the jacobian
        jacobian_aug = np.vstack((jacobian, np.sqrt(lambda_)*np.eye(num_params)))
        # augment the innovations
        innovs_aug = np.hstack((innovs, np.zeros(num_params)))
        # calucalte based on augmented forms
        delta_param = -np.linalg.lstsq(jacobian_aug, innovs_aug)[0]
    return delta_param

#%% _predict_func_change
def _predict_func_change(delta_param, gradient, hessian):
    r"""
    Predicts change in sum of squared errors function.

    Parameters
    ----------
    delta_param : ndarray (N,)
        Change in parameters
    gradient : ndarray (1, N)
        gradient (1st derivatives) of the cost function
    hessian : ndarray (N, N)
        hessian (2nd derivatives) of the cost function

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
    >>> print(delta_func)
    27.5

    """
    # calculate and return the predicted change
    delta_func = gradient.T @ delta_param + 0.5*delta_param.T @ hessian @ delta_param
    # check that this is a scalar result and return the result
    assert delta_func.size == 1 and delta_func.ndim <= 1
    if delta_func.ndim == 1:
        delta_func = delta_func[0]
    return delta_func

#%% _check_for_convergence
def _check_for_convergence(opti_opts, cosmax, delta_step_len, pred_func_change):
    r"""Check for convergence."""
    # alias the log level
    log_level = Logger().get_level()

    # initialize the output and assume not converged
    convergence = False

    # check for and optionally display the reasons for convergence
    if cosmax <= opti_opts.tol_cosmax_gradient:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because cosmax <= options.tol_cosmax_gradient')
    if delta_step_len <= opti_opts.tol_delta_step:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because delta_step_len <= options.tol_delta_step')
    if abs(pred_func_change) <= opti_opts.tol_delta_cost:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because abs(pred_func_change) <= options.tol_delta_cost')
    return convergence

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
    # Calculate some norms
    newton_len   = norm(delta_param)
    gradient_len = norm(jacobian)
    cauchy_len   = gradient_len**3/grad_hessian_grad
    # relaxed_Newton_point is between the initial point and the Newton point
    # If x_bias = 0, the relaxed point is at the Newton point
    relaxed_newton_len = 1 - x_bias*(1 + cauchy_len*gradient_len/(jacobian.T @ delta_param))

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
                new_cau_weighting = (discr - cau_dot_new_minus_cau) / new_minus_cau_len_sq
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

#%% _dogleg_search
def _dogleg_search(opti_opts, old_params, delta_param, old_innovs, jacobian, gradient, hessian, \
    trust_radius, old_cost, *, \
    search_method='trust_region', normalized=False, is_max_like=False):
    r"""
    Searchs for improved parameters for nonlinear least square or maximum likelihood function, using
    a trust radius search path.

    """
    # process inputs
    search_method = search_method.lower().replace(' ', '_')

    # alias the log level
    log_level = Logger().get_level()

    # hard-coded values # TODO: put in opti_opts?
    step_limit = 5
    x_bias = 0.8
    grow_radius = 2
    shrink_radius = 0.5
    log_det_B = 0 # TODO: get this elsewhere for max_likelihood mode

    # initialize status flags and counters
    try_again = True
    tried_expanding = False
    tried_shrinking = False
    died_on_step_cuts = False

    num_shrinks = 0
    step_number = 0
    num_evals   = 0

    new_params = old_params.copy()

    best_cost = old_cost

    grad_hessian_grad = gradient.T @ hessian @ gradient
    param_typical = OptiParam.get_array(opti_opts.params, type_='typical')

    while (num_shrinks < step_limit) and try_again:
        # increment step number
        step_number += 1

        # compute restrained trial parameter step
        if search_method == 'trust_region':
            (new_delta_param, step_len, step_scale, step_type) = _double_dogleg(delta_param, \
                jacobian, grad_hessian_grad, x_bias, trust_radius)

        elif search_method == 'levenberg_marquardt':
            new_delta_param = _levenberg_marquardt(jacobian, old_innovs, lambda_=1/trust_radius)
            step_type       = 'Levenberg-Marquardt'
            step_len        = norm(new_delta_param)
            step_scale      = step_len/norm(new_delta_param)

        else:
            raise ValueError('Unexpected value for search_method of "{}".'.format(search_method))

        # predict function change based on linearized model
        pred_func_change = _predict_func_change(new_delta_param, gradient, hessian)

        # set new parameter values
        new_params += delta_param

        # evaluate the cost function at the new parameter values
        if normalized:
            new_params *= param_typical # TODO: get param_typical

        # Run model
        (_, new_innovs) = _function_wrapper(opti_opts)

        sum_sq_innov = np.sum(new_innovs**2)
        if is_max_like:
            trial_cost = 0.5*(sum_sq_innov + log_det_B)
        else:
            trial_cost = 0.5*sum_sq_innov

        num_evals += 1

        # check if this step actually an improvement
        is_improvement = trial_cost < best_cost

        # decide what to do with this step
        if is_improvement:
            best_cost = trial_cost
            best_param = new_params.copy()
            if step_type == 'Newton':
                # this was a Newton step.
                trust_radius = step_len
                # No point in trying anything more. This is probably the best we can do.
                try_again       = False
                step_resolution = 'Accept the Newton step.'
            elif tried_shrinking:
                try_again       = False
                step_resolution = 'Improvement after shrinking, so accept this restrained step.'
            else:
                # Constrained step yielded some improvement and there is a possibility that a still
                # larger step might do better.
                trust_radius    = grow_radius * step_len
                tried_expanding = True
                try_again       = True
                step_resolution = 'Constrained step yielded some improvement, so try still longer step.'

        else:
            # Candidate step yielded no improvement
            if tried_expanding:
                # Give up the search
                try_again       = False
                trust_radius /= grow_radius
                params          = best_param
                step_resolution = 'Worse result after expanding, so accept previous restrained step.'
                if normalized:
                    new_params = old_params * param_typical
                else:
                    new_params = old_params.copy()
                # Run model
                (_, innovs_new) = _function_wrapper(opti_opts)
                num_evals += 1

            else:
                # There is still hope. Reduce step size.
                tried_shrinking = True
                if step_type == 'Newton':
                    # A Newton step failed.
                    trust_radius = shrink_radius*step_len
                else:
                    # Some other type of step failed.
                    trust_radius *= shrink_radius
                step_resolution = 'Bad step rejected, try still smaller step.'
                num_shrinks += 1
                try_again = True

        if log_level >= 5:
            # Show the user what is going on.
            pass # TODO: write this

        if try_again:
            # Reset params to get ready to try new delta_params
            new_params = old_params.copy()

    # Display status message
    if num_shrinks >= step_limit:
        print(' Died on step cuts.')
        print('Failed to find any step on the dogleg path that was actually an improvement')
        print(' before exceeding the step cut limit, which was {}  steps.'.format(step_limit))
        died_on_step_cuts = True

    return (new_params, delta_param, new_innovs, best_cost, num_evals, died_on_step_cuts)

#%% _analyze_results
def _analyze_results(innovs_final, log_level=10):
    r"""Analyze the results."""
    # alias the log level
    log_level = Logger().get_level()
    # update the status
    if log_level >= 5:
        print('Analyzing final results.')
    # build the results
    bpe_results = {}
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

    >>> from dstauffman import OptiOpts, OptiParam, validate_opti_opts
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

    >>> from dstauffman import run_bpe, OptiOpts #TODO: finish this

    """
    # alias the log level
    log_level = Logger().get_level()

    # check for valid parameters
    validate_opti_opts(opti_opts)

    # alias some stuff
    names = OptiParam.get_names(opti_opts.params)
    is_max_like = False # TODO: worry about this later
    trust_radius = 1.0 # TODO: put in opti_opts later
    grad_log_det_B = 0 # TODO: calculate somewhere later
    hessian_log_det_b = 0 # TODO: calculate somewhere later
    cosmax = 1 # TODO: calculate somewhere later
    grow_radius = 2 # TODO: put in opti_opts

    # initialize counters
    iter_count = 0

    # run the model
    if log_level >= 2:
        print('Running initial simulation.')
    (_, innovs_start) = _function_wrapper(opti_opts)
    iter_count += 1

    # initialize costs
    first_cost = 0.5 * rms(innovs_start**2, ignore_nans=True)
    prev_cost  = first_cost
    best_cost  = first_cost

    # initialize loop variables
    old_innovs  = innovs_start.copy()
    old_params  = opti_opts.get_param_func(names, **opti_opts.model_args)
    delta_param = np.zeros(len(names))

    # Do some stuff
    while iter_count <= opti_opts.max_iters:
        # update status
        if log_level >= 2:
            print('Running iteration {}.'.format(iter_count))

        # run finite differences code to numerically approximate the Jacobian matrix
        jacobian = _calculate_jacobian(opti_opts, old_params, old_innovs)

        # calculate the numerical gradient with respect to the estimated parameters
        gradient = jacobian.T @ old_innovs
        if is_max_like:
            gradient += grad_log_det_B

        # calculate the hessian matrix
        hessian = jacobian.T @ jacobian

        # Check direction of the last step and the gradient. If the old step and the negative new
        # gradient are in the same general direction, then increase the trust radius.
        grad_dot_step = gradient.T @ delta_param
        if grad_dot_step > 0 and iter_count > 1:
            trust_radius += grow_radius
            if log_level >= 8:
                print('Old step still in descent direction, so expand current trust_radius to {}.'.format(trust_radius))

        # calculate the delta parameter step to try on the next iteration
        if is_max_like:
            delta_param = -np.linalg.lstsq(hessian + hessian_log_det_b, gradient)[0]
        else:
            delta_param = _levenberg_marquardt(jacobian, old_innovs, lambda_=0)

        # find the step length
        delta_step_len = norm(delta_param)
        pred_func_change = _predict_func_change(delta_param, gradient, hessian)

        # check for convergence conditions
        convergence = _check_for_convergence(opti_opts, cosmax, delta_step_len, pred_func_change)
        if convergence:
            break

        # search for parameter set that is better than the current set
        (new_params, delta_param, new_innovs, best_cost, num_evals, died_on_step_cuts) = \
            _dogleg_search(opti_opts, old_params, delta_param, old_innovs, jacobian, \
            gradient, hessian, trust_radius, best_cost)

        # increment counter
        iter_count += 1

    # Run for final time
    if log_level >= 2:
        print('Running final simulation.')
    (results, innovs_final) = _function_wrapper(opti_opts)

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
