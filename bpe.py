# -*- coding: utf-8 -*-
r"""
BPE module file for the dstauffman code.  It defines the code necessary for doing batch parameter
estimation analysis, and is arbitrary enough to be called with many different modules.

Notes
-----
#.  Written by David C. Stauffer in May 2015 and continued in April 2016.  This work is based
    loosely on prior experience at Lockheed Martin using the GOLF/BPE code with GARSE, but all the
    numeric algorithms are re-coded from external sources to avoid any potential proprietary issues.
#.  Uses the Python @ infix operator, so you must be on v3.5 or newer
"""
# pylint: disable=E1101, C0301, C0326

#%% Imports
from copy import deepcopy
import doctest
import numpy as np
import os
from scipy.linalg import norm
import time
import unittest
from dstauffman.classes  import Frozen, SaveAndLoad
from dstauffman.plotting import Opts, plot_correlation_matrix, plot_multiline_history, \
                                    plot_bpe_convergence
from dstauffman.utils    import rss, setup_dir

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

    @classmethod
    def set_level(cls, level):
        r"""Sets the logging level."""
        cls.level = cls._check_level(level)

#%% OptiOpts
class OptiOpts(Frozen):
    r"""
    Optimization options for the batch parameter estimator.
    """
    def __init__(self):
        # specifically required settings
        self.model_func      = None
        self.model_args      = None # {}
        self.cost_func       = None
        self.cost_args       = None # {} # TODO: add note, these are additional cost args, plus model_args
        self.get_param_func  = None
        self.set_param_func  = None
        self.output_folder   = ''
        self.output_results  = 'bpe_results.hdf5'
        self.params          = None # []
        self.start_func      = None
        self.final_func      = None

        # less common optimization settings
        self.slope_method    = 'one_sided' # from {'one_sided', 'two_sided'}
        self.is_max_like     = False
        self.search_method   = 'trust_region' # from {'trust_region', 'levenberg_marquardt'}
        self.max_iters       = 10
        self.tol_cosmax_grad = 1e-4
        self.tol_delta_step  = 1e-20
        self.tol_delta_cost  = 1e-20
        self.step_limit      = 5
        self.x_bias          = 0.8
        self.grow_radius     = 2
        self.shrink_radius   = 0.5
        self.trust_radius    = 1.0

    def __eq__(self, other):
        r"""
        Checks for equality based on the values of the fields.
        """
        # if not of the same type, then they are not equal
        if type(other) != type(self):
            return False
        # loop through the fields, and if any are not equal, then it's not equal
        for key in vars(self):
            if getattr(self, key) != getattr(other, key):
                return False
        # if it made it all the way through the fields, then things must be equal
        return True

#%% OptiParam
class OptiParam(Frozen):
    r"""
    Optimization parameter to be estimated by the batch parameters estimator.
    """
    def __init__(self, name, *, best=np.nan, min_=-np.inf, max_=np.inf, minstep=1e-4, typical=1.):
        self.name = name
        self.best = best
        self.min_ = min_
        self.max_ = max_
        self.minstep = minstep
        self.typical = typical

    def __eq__(self, other):
        r"""
        Checks for equality between two OptiParam instances.
        """
        # if not of the same type, then they are not equal
        if type(other) != type(self):
            return False
        # loop through the fields, and if any are not equal (or both NaNs), then it's not equal
        for key in vars(self):
            v1 = getattr(self, key)
            v2 = getattr(other, key)
            if v1 != v2 and (not np.isnan(v1) or not np.isnan(v2)):
                return False
        # if it made it all the way through the fields, then things must be equal
        return True

    @staticmethod
    def get_array(opti_param, type_='best'):
        r"""
        Gets a numpy vector of all the optimization parameters for the desired type.
        """
        # check for valid types
        if type_ in {'best', 'min_', 'max_', 'minstep'}:
            key = type_
        elif type_ in {'min', 'max'}:
            key = type_ + '_'
        elif type_ in {'typical'}:
            raise ValueError('"{}" is a placeholder for later on.'.format(type_))
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

#%% BpeResults
class BpeResults(Frozen, metaclass=SaveAndLoad):
    r"""
    Results of the Batch Parameter Estimator.
    """
    def __init__(self):
        self.param_names  = None
        self.begin_params = None
        self.begin_innovs = None
        self.begin_cost   = None
        self.num_evals    = 0
        self.num_iters    = 0
        self.costs        = []
        self.correlation  = None
        self.info_svd     = None
        self.covariance   = None
        self.final_params = None
        self.final_innovs = None
        self.final_cost   = None

    def __str__(self):
        r"""
        Print all the fields of the Results.
        """
        # fields to print
        keys = ['begin_params', 'begin_cost', 'num_evals', 'num_iters', 'final_params', \
            'final_cost', 'correlation', 'info_svd', 'covariance', 'costs']
        # initialize output text
        text = [' BpeResults:']
        # loop through fields
        for key in keys:
            text.append('  {}: {}'.format(key, getattr(self, key)))
        # return everything as a single string
        return '\n'.join(text)

    def pprint(self):
        names = [name.decode('utf-8') for name in self.param_names]
        print('Initial parameters:')
        _pprint_args(names, self.begin_params)
        print('Final parameters:')
        _pprint_args(names, self.final_params)

#%% CurrentResults
class CurrentResults(Frozen, metaclass=SaveAndLoad):
    r"""
    Current results used as temporary values through the analysis.
    """
    def __init__(self):
        r"""
        Initialize the fields.
        """
        self.trust_rad = None
        self.params    = None
        self.innovs    = None
        self.cost      = None

    def __str__(self):
        r"""
        Print a useful summary of results.
        """
        text = [' Current Results:']
        text.append('  Trust Radius: {}'.format(self.trust_rad))
        text.append('  Best Cost: {}'.format(self.cost))
        text.append('  Best Params: {}'.format(self.params))
        return '\n'.join(text)

#%% _pprint_args
def _pprint_args(names, values):
    r"""
    Prints the current name and value pairs for all the estimated parameters.
    """
    # get the maximum length of any parameter name
    max_len = max(len(x) for x in names)
    # initialize the lines
    lines = []
    # loop through all the names
    for (ix, this_name) in enumerate(names):
        # add the .ix part into param
        this_name = this_name.replace('param.','param.ix(c).')
        # get this value
        this_value = values[ix]
        # find how much to pad this particular name
        pad = max_len - len(this_name)
        # append this line
        lines.append('        ' + this_name + ' ' + ' '*pad + '= {:g}'.format(this_value))
    # combine all the lines into one string
    text = '\n'.join(lines)
    # print the resulting string and return it
    print(text)
    return text

#%% _print_divider
def _print_divider(new_line=True):
    r"""
    Prints some characters to the std out to break apart the different stpes within the model.

    Parameters
    ----------
    new_line : bool, optional
        Whether to include a newline in the print statement

    Examples
    --------

    >>> from dstauffman.bpe import _print_divider
    >>> _print_divider()
    <BLANKLINE>
    ******************************

    """
    # hard-coded text
    text = '\n******************************'
    # print with or without newline
    print(text) if new_line else print(text[1:])

#%% _function_wrapper
def _function_wrapper(opti_opts, bpe_results, model_args=None, cost_args=None):
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
    bpe_results.num_evals += 1

    # Run the cost function to get the innovations
    innovs = opti_opts.cost_func(results, **model_args, **cost_args)

    # Set any NaNs to zero so that they are ignored
    innovs[np.isnan(innovs)] = 0

    return (results, innovs)

#%% _finite_differences
def _finite_differences(opti_opts, model_args, bpe_results, cur_results, *, two_sided=False, normalized=False):
    r"""
    Perturbs the state by a litte bit and calculates the numerical slope (i.e. Jacobian approximation)

    Has options for first or second order approximations.

    Notes
    -----
    #.  No input variables are modified by this function.

    References
    ----------
    #.  Conn, Andrew R., Gould, Nicholas, Toint, Philippe, "Trust-Region Methods," MPS-SIAM Series
        on Optimization, 2000.

    """
    # hard-coded values
    sqrt_eps = np.sqrt(np.finfo(float).eps)
    step_sf  = 0.1

    # alias useful values
    log_level     = Logger().get_level()
    names         = [name.decode('utf-8') for name in bpe_results.param_names]
    num_param     = cur_results.params.size
    num_innov     = cur_results.innovs.size
    param_signs   = np.sign(cur_results.params)
    param_signs[param_signs == 0] = 1
    param_minstep = OptiParam.get_array(opti_opts.params, type_='minstep')
    params_min    = OptiParam.get_array(opti_opts.params, type_='min')
    params_max    = OptiParam.get_array(opti_opts.params, type_='max')
    if normalized:
        param_typical = OptiParam.get_array(opti_opts.params, type_='typical')

    # initialize output
    jacobian = np.zeros((num_innov, num_param), dtype=float)

    # initialize loop variables
    grad_log_det_B = 0 # TODO: calculate somewhere later
    # set parameter pertubation (Reference 1, section 8.4.3)
    if normalized:
        perturb_fact  = 1 # TODO: how to get perturb_fact?
        param_perturb = perturb_fact * sqrt_eps * param_signs
    else:
        temp_step     = np.abs(cur_results.params)*step_sf * 1/cur_results.trust_rad
        param_perturb = param_signs * np.maximum(temp_step, param_minstep)

    temp_params_plus  = cur_results.params.copy()
    temp_params_minus = cur_results.params.copy()

    for i_param in range(num_param):
        # update the parameters for this run
        temp_params_plus[i_param]  = np.minimum(cur_results.params[i_param] + param_perturb[i_param], params_max[i_param])
        temp_params_minus[i_param] = np.maximum(cur_results.params[i_param] - param_perturb[i_param], params_min[i_param])

        # get the new parameters for this run
        if normalized:
            temp_params = temp_params_plus * param_typical
        else:
            temp_params = temp_params_plus.copy()

        # call model with new parameters
        opti_opts.set_param_func(names=names, values=temp_params, **model_args)
        if log_level >= 8:
            print('  Running model with {} = {}'.format(names[i_param], temp_params[i_param]))
        (_, new_innovs) = _function_wrapper(opti_opts, bpe_results, model_args)

        if two_sided:
            if normalized:
                temp_params = temp_params_minus * param_typical
            else:
                temp_params = temp_params_minus.copy()
            opti_opts.set_param_func(names=names, values=temp_params, **model_args)
            if log_level >= 8:
                print('  Running model with {} = {}'.format(names[i_param], temp_params[i_param]))
            (_, new_innovs_minus) = _function_wrapper(opti_opts, bpe_results, model_args)

        # compute the jacobian
        if two_sided:
            jacobian[:, i_param] = 0.5 * (new_innovs - new_innovs_minus) / param_perturb[i_param]
            #grad_log_det_b[i_param] = 0.25 *
        else:
            jacobian[:, i_param] = (new_innovs - cur_results.innovs) / param_perturb[i_param]

        # reset the parameter to the original value for next loop
        temp_params_plus[i_param]  = cur_results.params[i_param]
        temp_params_minus[i_param] = cur_results.params[i_param]

    # calculate the numerical gradient with respect to the estimated parameters
    gradient = jacobian.T @ cur_results.innovs
    if opti_opts.is_max_like:
        gradient += grad_log_det_B

    # calculate the hessian matrix
    hessian = jacobian.T @ jacobian

    return (jacobian, gradient, hessian)

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
    if cosmax <= opti_opts.tol_cosmax_grad:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because cosmax of {} <= options.tol_cosmax_grad of {}'.format(\
                cosmax, opti_opts.tol_cosmax_grad))
    if delta_step_len <= opti_opts.tol_delta_step:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because delta_step_len of {} <= options.tol_delta_step of {}'.format(\
                delta_step_len, opti_opts.tol_delta_step))
    if abs(pred_func_change) <= opti_opts.tol_delta_cost:
        convergence = True
        if log_level >= 5:
            print('Declare convergence because abs(pred_func_change) of {} <= options.tol_delta_cost of {}'.format(\
                abs(pred_func_change), opti_opts.tol_delta_cost))
    return convergence

#%% _double_dogleg
def _double_dogleg(delta_param, gradient, grad_hessian_grad, x_bias, trust_radius):
    r"""
    Compute a double dog-leg parameter search.

    Parameters
    ----------
    delta_param :
        Small change in parameter values
    gradient :
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
    >>> gradient = np.array([[3], [4]])
    >>> grad_hessian_grad = 5
    >>> x_bias = 0.1
    >>> trust_radius = 2
    >>> (new_delta_param, step_len, step_scale, step_type) = _double_dogleg(delta_param, \
    ...     gradient, grad_hessian_grad, x_bias, trust_radius)

    """
    # Calculate some norms
    newton_len   = norm(delta_param)
    gradient_len = norm(gradient)
    cauchy_len   = gradient_len**3/grad_hessian_grad
    # relaxed_Newton_point is between the initial point and the Newton point
    # If x_bias = 0, the relaxed point is at the Newton point
    relaxed_newton_len = 1 - x_bias*(1 + cauchy_len*gradient_len/(gradient.T @ delta_param))

    # Compute the minimizing point on the dogleg path
    # This point is inside the trust radius and minimizes the linearized least square function
    if newton_len <= trust_radius:
        # Newton step is inside the trust region so take it
        new_delta_param = delta_param.copy()
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
            new_delta_param = -(trust_radius / gradient_len)*gradient
        else:
            # Take a dogleg step between relaxed Newton and Cauchy steps
            # This will be on a line between the Cauchy point and the relaxed
            # Newton point such that the distance from the initial point
            # and the restrained point is equal to the trust radius.

            # Cauchy point is at the predicted minimum of the function along the
            # gradient search direction
            cauchy_pt             = -cauchy_len / gradient_len * gradient
            new_minus_cau         = relaxed_newton_len * delta_param - cauchy_pt
            cau_dot_new_minus_cau = cauchy_pt.T @ new_minus_cau
            cau_len_sq            = cauchy_pt.T @ cauchy_pt
            new_minus_cau_len_sq  = new_minus_cau.T @ new_minus_cau
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
def _dogleg_search(opti_opts, model_args, bpe_results, cur_results, delta_param, jacobian, gradient, \
        hessian, *, normalized=False):
    r"""
    Searchs for improved parameters for nonlinear least square or maximum likelihood function, using
    a trust radius search path.
    """
    # process inputs
    search_method = opti_opts.search_method.lower().replace(' ', '_')
    if normalized:
        param_typical = OptiParam.get_array(opti_opts.params, type_='typical')

    # alias the log level, trust radius, parameters names and bounds
    log_level = Logger().get_level()
    trust_radius = cur_results.trust_rad
    names = [name.decode('utf-8') for name in bpe_results.param_names]
    params_min = OptiParam.get_array(opti_opts.params, type_='min')
    params_max = OptiParam.get_array(opti_opts.params, type_='max')

    # save a copy of the original param values
    orig_params = cur_results.params.copy()

    # do some calculations for things constant within the loop
    grad_hessian_grad = gradient.T @ hessian @ gradient
    log_det_B = 0 # TODO: get this elsewhere for max_likelihood mode

    # initialize status flags and counters
    try_again       = True
    tried_expanding = False
    tried_shrinking = False
    num_shrinks     = 0
    step_number     = 0
    failed          = False
    was_limited     = False

    # try a step
    while (num_shrinks < opti_opts.step_limit) and try_again and not was_limited:
        # increment step number
        step_number += 1

        # compute restrained trial parameter step
        if search_method == 'trust_region':
            (new_delta_param, step_len, step_scale, step_type) = _double_dogleg(delta_param, \
                gradient, grad_hessian_grad, opti_opts.x_bias, trust_radius)

        elif search_method == 'levenberg_marquardt':
            new_delta_param = _levenberg_marquardt(jacobian, cur_results.innovs, lambda_=1/trust_radius)
            step_type       = 'Levenberg-Marquardt'
            step_len        = norm(new_delta_param)
            step_scale      = step_len/norm(new_delta_param)

        else:
            raise ValueError('Unexpected value for search_method of "{}".'.format(search_method))

        # predict function change based on linearized model
        pred_func_change = _predict_func_change(new_delta_param, gradient, hessian)

        # set new parameter values
        params = orig_params + new_delta_param
        if normalized:
            params *= param_typical

        # enforce min/max bounds
        if np.any(params > params_max):
            was_limited = True
            params = np.minimum(params, params_max)
        if np.any(params < params_min):
            was_limited = True
            params = np.maximum(params, params_min)

        # Run model
        if log_level >= 8:
            print('  Running model with new trial parameters.')
        opti_opts.set_param_func(names=names, values=params, **model_args)
        (_, innovs) = _function_wrapper(opti_opts, bpe_results, model_args)

        # evaluate the cost function at the new parameter values
        sum_sq_innov = rss(innovs, ignore_nans=True)
        if opti_opts.is_max_like:
            trial_cost = 0.5*(sum_sq_innov + log_det_B)
        else:
            trial_cost = 0.5 * sum_sq_innov

        # check if this step actually an improvement
        is_improvement = trial_cost < cur_results.cost

        # decide what to do with this step
        if is_improvement:
            # update the new best values
            cur_results.params = params.copy()
            cur_results.innovs = innovs.copy()
            cur_results.cost   = trial_cost
            # Determine what to do next
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
                trust_radius    = opti_opts.grow_radius * step_len
                tried_expanding = True
                try_again       = True
                step_resolution = 'Constrained step yielded some improvement, so try still longer step.'

        else:
            # Candidate step yielded no improvement
            if tried_expanding:
                # Give up the search
                try_again       = False
                trust_radius    /= opti_opts.grow_radius
                step_resolution = 'Worse result after expanding, so accept previous restrained step.'
            else:
                # There is still hope. Reduce step size.
                tried_shrinking = True
                if step_type == 'Newton':
                    # A Newton step failed.
                    trust_radius = opti_opts.shrink_radius*step_len
                else:
                    # Some other type of step failed.
                    trust_radius *= opti_opts.shrink_radius
                step_resolution = 'Bad step rejected, try still smaller step.'
                num_shrinks += 1
                try_again = True

        if log_level >= 8:
            print(' Tried a {} step of length: {}, (with scale: {}).'.format(step_type, step_len, step_scale))
            print(' New trial cost: {}'.format(trial_cost))
            print(' With result: {}'.format(step_resolution))
            if was_limited:
                print(' Caution, the step length was limited by the given bounds.')

    # Display status message
    if log_level >= 8 and num_shrinks >= opti_opts.step_limit:
        print('Died on step cuts.')
        print(' Failed to find any step on the dogleg path that was actually an improvement')
        print(' before exceeding the step cut limit, which was {}  steps.'.format(opti_opts.step_limit))
        failed = True
    if log_level >= 5:
        print(' New parameters are: {}'.format(cur_results.params))
    return failed

#%% _analyze_results
def _analyze_results(opti_opts, bpe_results, jacobian, normalized=False):
    r"""Analyze the results."""
    # hard-coded values
    min_eig = 1e-14 # minimum allowed eigenvalue

    # alias the log level
    log_level = Logger().get_level()

    # get the names and number of parameters
    num_params    = len(bpe_results.param_names)

    # update the status
    if log_level >= 5:
        print('Analyzing final results.')
    if log_level >= 8:
        print('There were a total of {} function model evaluations.'.format(bpe_results.num_evals))

    # exit if nothing else to analyze
    if opti_opts.max_iters == 0:
        return

    # Compute values of un-normalized parameters.
    if normalized:
        param_typical = OptiParam.get_array(opti_opts.params, type_='typical')
        normalize_matrix  = np.eye(num_params) @ (1 / param_typical)
    else:
        normalize_matrix  = np.eye(num_params)

    # Make information, covariance matrix, compute Singular Value Decomposition (SVD).
    try:
        # note, python has x = U*S*Vh instead of U*S*V', when V = Vh'
        (_, S_jacobian, Vh_jacobian) = np.linalg.svd(jacobian @ normalize_matrix, full_matrices=False)
        V_jacobian = Vh_jacobian.T
        temp = np.power(S_jacobian, -2, out=np.zeros(S_jacobian.shape), where=S_jacobian > min_eig)
        covariance = V_jacobian @ np.diag(temp) @ Vh_jacobian
    except MemoryError:
        if log_level >= 6:
            print('Singular value decomposition of Jacobian failed.')
        V_jacobian = np.nan * np.ones((num_params, num_params))
        covariance = np.inv(jacobian.T @ jacobian)

    param_one_sigmas = np.sqrt(np.diag(covariance))
    param_one_sigmas[param_one_sigmas < min_eig] = np.nan
    correlation    = covariance / (param_one_sigmas[:, np.newaxis] @ param_one_sigmas[np.newaxis, :])
    covariance[np.isnan(correlation)] = np.nan

    # Update SVD and covariance for the normalized parameters (but correlation remains as calculated above)
    if normalized:
        try:
            (_, S_jacobian, Vh_jacobian) = np.linalg.svd(jacobian, full_matrices=False)
            V_jacobian = Vh_jacobian.T
            covariance = V_jacobian @ np.diag(S_jacobian**-2) @ Vh_jacobian
        except MemoryError:
            pass # caught in earlier exception (hopefully?)

    # update the results
    bpe_results.correlation  = correlation
    bpe_results.info_svd     = V_jacobian.T
    bpe_results.covariance   = covariance

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

    >>> from dstauffman import Logger, OptiOpts, validate_opti_opts
    >>> logger = Logger(10)
    >>> opti_opts = OptiOpts()
    >>> opti_opts.model_func     = str
    >>> opti_opts.model_args     = {'a': 1}
    >>> opti_opts.cost_func      = str
    >>> opti_opts.cost_args      = {'b': 2}
    >>> opti_opts.get_param_func = str
    >>> opti_opts.set_param_func = repr
    >>> opti_opts.output_folder  = ''
    >>> opti_opts.output_results = ''
    >>> opti_opts.params         = [1, 2]

    >>> is_valid = validate_opti_opts(opti_opts)
    ******************************
    Validating optimization options.

    >>> print(is_valid)
    True

    """
    # get the logging level
    log_level = Logger().get_level()
    # display some information
    if log_level >= 5:
        _print_divider(new_line=False)
        print('Validating optimization options.')
    # Must have specified all parameters
    assert callable(opti_opts.model_func)
    assert isinstance(opti_opts.model_args, dict)
    assert callable(opti_opts.cost_func)
    assert isinstance(opti_opts.cost_args, dict)
    assert callable(opti_opts.get_param_func)
    assert callable(opti_opts.set_param_func)
    # Must estimate at least one parameter (TODO: make work with zero?)
    assert isinstance(opti_opts.params, list) and len(opti_opts.params) > 0
    # Must be one of these two slope methods
    assert opti_opts.slope_method in {'one_sided', 'two_sided'}
    # Must be one of these two seach methods
    assert opti_opts.search_method in {'trust_region', 'levenberg_marquardt'}
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
    # check for valid parameters
    validate_opti_opts(opti_opts)

    # start timer
    start_model = time.time()

    # alias some stuff
    log_level = Logger().get_level()
    names     = OptiParam.get_names(opti_opts.params)
    two_sided = True if opti_opts.slope_method == 'two_sided' else False

    # determine if saving data
    filename  = os.path.join(opti_opts.output_folder, opti_opts.output_results)
    is_saving = bool(opti_opts.output_folder) and bool(opti_opts.output_results)

    # TODO: write ability to resume from previously saved iteration results
    # initialize the output and current results instances
    bpe_results = BpeResults()
    cur_results = CurrentResults()

    # save the parameter names
    bpe_results.param_names = [name.encode('utf-8') for name in names]

    # create a working copy of the model arguments that can be modified in place while running
    model_args = deepcopy(opti_opts.model_args)

    # run an optional initialization function
    if opti_opts.start_func is not None:
        init_saves = opti_opts.start_func(**model_args)
    else:
        init_saves = {}

    # future calculations
    hessian_log_det_b = 0 # TODO: calculate somewhere later
    cosmax = 1 # TODO: calculate somewhere later

    # run the initial model
    if log_level >= 2:
        new_line = log_level > 5
        _print_divider(new_line)
        print('Running initial simulation.')
    (_, cur_results.innovs) = _function_wrapper(opti_opts, bpe_results, model_args)

    # initialize loop variables
    iter_count   = 1
    delta_param  = np.zeros(len(names))

    # initialize current results
    cur_results.trust_rad = opti_opts.trust_radius
    cur_results.cost      = 0.5 * rss(cur_results.innovs, ignore_nans=True)
    cur_results.params    = opti_opts.get_param_func(names=names, **model_args)

    # set relevant results variables
    bpe_results.begin_params = cur_results.params.copy()
    bpe_results.begin_innovs = cur_results.innovs.copy()
    bpe_results.begin_cost   = cur_results.cost
    bpe_results.costs.append(cur_results.cost)

    # display initial status
    if log_level >= 6:
        print(' Initial parameters: {}'.format(cur_results.params))
        print(' Initial cost: {}'.format(cur_results.cost))

    # Set-up saving: check that the folder exists
    if is_saving:
        if opti_opts.output_folder and not os.path.isdir(opti_opts.output_folder):
            # if the folder doesn't exist, then create it
            setup_dir(opti_opts.output_folder) # pragma: no cover

    # Do some stuff
    convergence = False
    jacobian = 0
    while iter_count <= opti_opts.max_iters:
        # update status
        if log_level >= 2:
            _print_divider()
            print('Running iteration {}.'.format(iter_count))

        # run finite differences code to numerically approximate the Jacobian, gradient and Hessian
        (jacobian, gradient, hessian) = _finite_differences(opti_opts, model_args, bpe_results, \
            cur_results, two_sided=two_sided)

        # Check direction of the last step and the gradient. If the old step and the negative new
        # gradient are in the same general direction, then increase the trust radius.
        grad_dot_step = gradient.T @ delta_param
        if grad_dot_step > 0 and iter_count > 1:
            cur_results.trust_rad += opti_opts.grow_radius
            if log_level >= 8:
                print('Old step still in descent direction, so expand current trust_radius to {}.'.format(\
                    cur_results.trust_rad))

        # calculate the delta parameter step to try on the next iteration
        if opti_opts.is_max_like:
            delta_param = -np.linalg.lstsq(hessian + hessian_log_det_b, gradient)[0]
        else:
            delta_param = _levenberg_marquardt(jacobian, cur_results.innovs, lambda_=0)

        # find the step length
        delta_step_len = norm(delta_param)
        pred_func_change = _predict_func_change(delta_param, gradient, hessian)

        # check for convergence conditions
        convergence = _check_for_convergence(opti_opts, cosmax, delta_step_len, pred_func_change)
        if convergence:
            break

        # search for parameter set that is better than the current set
        failed = _dogleg_search(opti_opts, model_args, bpe_results, cur_results, delta_param, jacobian, \
            gradient, hessian)
        bpe_results.costs.append(cur_results.cost)

        # save results from this iteration
        if is_saving:
            bpe_results.save(os.path.join(opti_opts.output_folder, 'bpe_results_iter_{}.hdf5'.format(iter_count)))
            cur_results.save(os.path.join(opti_opts.output_folder, 'cur_results_iter_{}.hdf5'.format(iter_count)))

        # increment counter
        iter_count += 1

        if failed:
            break

    # display if this converged out timed out on iteration steps
    if log_level >= 6 and not convergence and not failed:
        print('Stopped iterating due to hitting the max number of iterations: {}.'.format(opti_opts.max_iters))

    # run an optional final function before doing the final simulation
    if opti_opts.final_func is not None:
        opti_opts.final_func(**model_args, settings=init_saves)

    # Run for final time
    if log_level >= 2:
        _print_divider()
        print('Running final simulation.')
    opti_opts.set_param_func(names=names, values=cur_results.params, **model_args)
    (results, cur_results.innovs) = _function_wrapper(opti_opts, bpe_results, model_args)
    cur_results.cost = 0.5 * rss(cur_results.innovs, ignore_nans=True)
    bpe_results.final_innovs = cur_results.innovs.copy()
    bpe_results.final_params = cur_results.params.copy()
    bpe_results.final_cost   = cur_results.cost
    bpe_results.costs.append(cur_results.cost)

    # display final status
    if log_level >= 6:
        print(' Final parameters: {}'.format(bpe_results.final_params))
        print(' Final cost: {}'.format(bpe_results.final_cost))
    if log_level >= 8:
        print(' Final individual parameters:')
        _pprint_args(names, bpe_results.final_params)

    # analyze BPE results
    _analyze_results(opti_opts, bpe_results, jacobian)

    # show status and save results
    if log_level > 2 and is_saving:
        print('Saving results to: "{}".'.format(filename))
    if is_saving:
        bpe_results.save(filename)

    # display total elapsed time
    if log_level >= 3:
        print('BPE Model completed: ' + time.strftime('%H:%M:%S', time.gmtime(time.time()-start_model)))

    return (bpe_results, results)

#%% plot_bpe_results
def plot_bpe_results(bpe_results, opts=None, *, plots=None):
    r"""
    Plots the results of estimation.
    """
    # hard-coded options
    label_values = False

    # get the logging level
    log_level = Logger.get_level()

    # alias the names
    if bpe_results.param_names is not None:
        names = [name.decode('utf-8') for name in bpe_results.param_names]
    else:
        names = []

    # defaults for which plots to make
    default_plots = {'innovs': False, 'convergence': False, 'correlation': False, 'info_svd': False, \
        'covariance': False}

    # check for optional variables
    if opts is None:
        opts = Opts()
    if plots is None:
        plots = default_plots
    else:
        # check for unexpected keys
        for key in plots:
            if key not in default_plots:
                raise ValueError('Unexpected plotting option: "{}".'.format(key))
        # start with the defaults, and then overlay any specified inputs
        for key in default_plots:
            if key not in plots:
                plots[key] = default_plots[key]

    # preallocate output
    figs = []

    # time based plots
    if plots['innovs']:
        if bpe_results.begin_innovs is not None and bpe_results.final_innovs is not None:
            time = np.arange(len(bpe_results.begin_innovs))
            data = np.vstack((bpe_results.begin_innovs, bpe_results.final_innovs)).T
            fig = plot_multiline_history(time, data, label='Innovs Before and After', opts=opts, colormap='bwr_r', \
                legend=['Before', 'After'])
            figs.append(fig)
        elif log_level >= 2:
            print("Data isn't available for Innovations plot.")
    if plots['convergence']:
        if len(bpe_results.costs) != 0:
            fig = plot_bpe_convergence(bpe_results.costs, opts=opts)
            figs.append(fig)
        elif log_level >= 2:
            print("Data isn't available for convergence plot.")

    if plots['correlation']:
        if bpe_results.correlation is not None:
            fig = plot_correlation_matrix(bpe_results.correlation, labels=names, opts=opts, \
                matrix_name='Correlation Matrix', cmin=-1, colormap='bwr', plot_lower_only=True, \
                label_values=label_values)
            figs.append(fig)
        elif log_level >= 2:
            print("Data isn't available for correlation plot.")

    if plots['info_svd']:
        if bpe_results.info_svd is not None:
            fig = plot_correlation_matrix(np.abs(bpe_results.info_svd), opts=opts, cmin=0, \
                matrix_name='Information SVD Matrix', colormap='cool', label_values=label_values, \
                labels=[['{}'.format(i+1) for i in range(len(names))], names])
            figs.append(fig)
        elif log_level >= 2:
            print("Data isn't available for information SVD plot.")

    if plots['covariance']:
        if bpe_results.covariance is not None:
            max_mag = np.nanmax(np.abs(bpe_results.covariance))
            fig = plot_correlation_matrix(bpe_results.covariance, labels=names, opts=opts, \
                matrix_name='Covariance Matrix', cmin=-max_mag, cmax=max_mag, colormap='bwr', \
                plot_lower_only=True, label_values=label_values)
            figs.append(fig)
        elif log_level >= 2:
            print("Data isn't available for covariance plot.")
    return figs

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_bpe', exit=False)
    doctest.testmod(verbose=False)
