# -*- coding: utf-8 -*-
r"""
Classes related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.

"""

#%% Imports
import doctest
import unittest

from matplotlib.colors import ListedColormap
import numpy as np

from dstauffman.classes import Frozen
from dstauffman.plot_generic import make_difference_plot, make_quaternion_plot
from dstauffman.plot_support import get_color_lists, setup_plots
from dstauffman.plotting import Opts
from dstauffman.units import get_factors

#%% KfInnov
class KfInnov(Frozen):
    r"""
    A class for Kalman Filter innovations outputs.

    Attributes
    ----------
    name : str
        Name of the innovation structure, often specifying which sensor it comes from, like GPS
    chan : [M, ] list of str
        Names of the different axes found within the innovations
    units : str
        Units for the innovations
    time : (N, ) ndarray
        Time vector
    innov : (N, M) ndarray
        Time history of the raw innovations
    norm : (N, M) ndarray
        Time history of the normalized innovations
    status : (N,)
        Status of the innovation, such as applied, or reason for rejection

    Examples
    --------
    >>> from dstauffman import KfInnov
    >>> innov = KfInnov()

    """
    def __init__(self, *, name='', units='', num_innovs=0, num_axes=0, time_dtype=float):
        r"""Initializes a new KfInnov instance."""
        self.name   = name
        self.chan   = ['' for i in range(num_axes)] if num_axes > 0 else None
        self.units  = units
        if num_innovs > 0:
            self.time   = np.empty(num_innovs, dtype=time_dtype)
            innov_shape = (num_axes, num_innovs) if num_axes > 1 else (num_innovs, )
            self.innov  = np.full(innov_shape, np.nan, dtype=float)
            self.norm   = np.full(innov_shape, np.nan, dtype=float)
            self.status = np.empty(num_innovs, dtype=int)
        else:
            self.time   = None
            self.innov  = None
            self.norm   = None
            self.status = None

#%% Kf
class Kf(Frozen):
    r"""
    A class for doing Kalman Filter analysis.

    Attributes
    ----------
    name : str
        Name of the structure, used when comparing multiple sources or runs
    chan : list of str
        Name of the states in the state and covar fields
    time : (N, ) ndarray
        Time vector
    att : (4, N) ndarray
        Attitude quaternion history
    pos : (3, N) ndarray
        Position history
    vel : (3, N) ndarray
        Velocity history
    innov : class KfInnov
        Innovation history for GPS measurements
    state : (N, M) ndarray
        State history
    covar : (N, M) ndarray
        Covariance history

    Examples
    --------
    >>> from dstauffman import Kf
    >>> kf = Kf()

    """
    def __init__(self, *, name='', num_points=0, num_states=0, time_dtype=float, active_states=None, **kwargs):
        r"""Initializes a new Kf instance."""
        self.name = name
        self.chan = ['' for i in range(num_states)] if num_states > 0 else None
        if num_points > 0:
            num_active  = num_states if active_states is None else len(active_states)
            state_shape = (num_active, num_points) if num_active > 1 else (num_points, )
            self.time   = np.empty(num_points, dtype=time_dtype)
            self.att    = np.empty((4, num_points), dtype=float)
            self.pos    = None # TODO: flag to enable?
            self.vel    = None # TODO: flag to enable?
            self.active = active_states if active_states is not None else np.arange(num_states)
            self.state  = np.empty(state_shape, dtype=float)
            self.istate = np.empty(num_states, dtype=float)
            self.covar  = np.empty(state_shape, dtype=float)
            self.innov  = KfInnov(time_dtype=time_dtype, **kwargs)
        else:
            self.time   = None
            self.att    = None
            self.pos    = None
            self.vel    = None
            self.active = None
            self.state  = None
            self.istate = None
            self.covar  = None
            self.innov  = KfInnov(time_dtype=time_dtype, **kwargs)

#%% Functions - calc_kalman_gain
def calc_kalman_gain(P, H, R, use_inverse=False):
    r"""
    Calculates K, the Kalman Gain matrix.

    Parameters
    ----------
    P : (N, N) ndarray
        Covariance Matrix
    H : (A, B) ndarray
        Measurement Update Matrix
    R : () ndarray
        Measurement Noise Matrix
    use_inverse : bool, optional
        Whether to explicitly calculate the inverse or not, default is False

    Returns
    -------
    K : (N, ) ndarray
        Kalman Gain Matrix

    Notes
    -----
    #.  Written by David C Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman import calc_kalman_gain
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(5)
    >>> H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
    >>> R = 0.5 * np.eye(3)
    >>> K = calc_kalman_gain(P, H, R)

    """
    if use_inverse:
        # explicit version with inverse
        K = (P @ H.T) @ np.linalg.inv(H @ P @ H.T + R)
    else:
        # implicit solver
        K = np.linalg.lstsq((H @ P @ H.T + R).T, (P @ H.T).T, rcond=None)[0].T
    return K

#%% Functions - propagate_covariance
def propagate_covariance(P, phi, Q, *, gamma=None, inplace=True):
    r"""
    Propagates the covariance forward in time.

    Parameters
    ----------
    P :
        Covariance matrix
    phi :
        State transition matrix
    Q :
        Process noise matrix
    gamma :
        Shaping matrix?
    inplace : bool, optional, default is True
        Whether to update the value inplace or as a new output

    Returns
    -------
    (N, N) ndarray
        Updated covariance matrix

    Notes
    -----
    #.  Written by David C. Stauffer in December 2018.
    #.  Updated by David C. Stauffer in July 2020 to have inplace option.

    Examples
    --------
    >>> from dstauffman import propagate_covariance
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(6)
    >>> phi = np.diag([1., 1, 1, -1, -1, -1])
    >>> Q = np.diag([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7])
    >>> propagate_covariance(P, phi, Q)
    >>> print(P[0, 0])
    0.002

    """
    if gamma is None:
        out = phi @ P @ phi.T + Q
    else:
        out = phi @ P @ phi.T + gamma @ Q @ gamma.T
    if inplace:
        P[:] = out
    else:
        return out

#%% Functions - update_covariance
def update_covariance(P, K, H, inplace=True):
    r"""
    Updates the covariance for a given measurement.

    Parameters
    ----------
    P : (N, N) ndarray
        Covariance Matrix
    K : (N, ) ndarray
        Kalman Gain Matrix
    H : (A, N) ndarray
        Measurement Update Matrix
    inplace : bool, optional, default is True
        Whether to update the value inplace or as a new output

    Returns
    -------
    P_out : (N, N) ndarray
        Updated Covariance Matrix

    Notes
    -----
    #.  Written by David C Stauffer in December 2018.
    #.  Updated by David C. Stauffer in July 2020 to have inplace option.

    Examples
    --------
    >>> from dstauffman import update_covariance
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(6)
    >>> P[0, -1] = 5e-2
    >>> K = np.ones((6, 3))
    >>> H = np.hstack((np.eye(3), np.eye(3)))
    >>> update_covariance(P, K, H)
    >>> print(P[-1, -1])
    -0.05

    """
    out = (np.eye(*P.shape) - K @ H) @ P
    if inplace:
        P[:] = out
    else:
        return out

#%% plot_attitude
def plot_attitude(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""
    Plots the attitude quaternion history.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import Kf, Opts, plot_attitude, quat_from_euler, quat_mult, quat_norm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> q1 = quat_norm(np.array([0.1, -0.2, 0.3, 0.4]))
    >>> dq = quat_from_euler(1e-6*np.array([-300, 100, 200]), [3, 1, 2])
    >>> q2 = quat_mult(dq, q1)

    >>> kf1      = Kf()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.att  = np.tile(q1[:, np.newaxis], (1, kf1.time.size))

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.att  = np.tile(q2[:, np.newaxis], (1, kf2.time.size))
    >>> kf2.att[3,4] += 50e-6
    >>> kf2.att = quat_norm(kf2.att)

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.vert_fact = 'micro'
    >>> opts.quat_comp = True
    >>> opts.sub_plots = True

    >>> fig_hand = plot_attitude(kf1, kf2, opts=opts)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        truth = Kf()
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'att': 'Attitude Quaternion'}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    plot_comps   = kwargs.pop('plot_components', this_opts.quat_comp)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # initialize outputs
    figs = []
    err  = dict()

    # call wrapper function for most of the details
    for (field, description) in fields.items():
        (this_figs, this_err) = make_quaternion_plot(description, kf1.time, kf2.time, getattr(kf1, field), getattr(kf2, field), \
            name_one=kf1.name, name_two=kf2.name, time_units=time_units, start_date=start_date, \
            rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, plot_components=plot_comps, single_lines=single_lines, \
            use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
            truth_name=truth.name, truth_time=truth.time, truth_data=truth.att, return_err=True, **kwargs)
        figs += this_figs
        err[field] = this_err

    # Setup plots
    setup_plots(figs, opts)
    if return_err:
        return (figs, err)
    return figs

#%% plot_los
def plot_los(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {'los': 'LOS'}
    out = plot_attitude(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% plot_position
def plot_position(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""
    Plots the position and velocity history.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import Kf, plot_position
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> kf1      = Kf()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.pos  = 1e6 * np.random.rand(3, 11)
    >>> kf1.vel  = 1e3 * np.random.rand(3, 11)

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.pos  = kf1.pos[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e5
    >>> kf2.vel  = kf1.vel[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 100

    >>> fig_hand = plot_position(kf1, kf2)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        truth = Kf()
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'pos': 'Position'}

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # hard-coded defaults
    elements       = kwargs.pop('elements', ['x', 'y', 'z'])
    default_units  = 'm' if 'pos' in fields else 'm/s' if 'vel' in fields else ''
    units          = kwargs.pop('units', default_units)
    leg_scale      = kwargs.pop('leg_scale', 'kilo')
    (fact, name)   = get_factors(leg_scale)
    second_yscale  = kwargs.pop('second_yscale', {name + units: 1/fact})
    color_lists    = get_color_lists()
    colormap       = ListedColormap(color_lists['vec_diff'].colors + color_lists['vec'].colors)

    # initialize outputs
    figs = []
    err  = dict()

    # call wrapper function for most of the details
    for (field, description) in fields.items():
        (this_figs, this_err) = make_difference_plot(description, kf1.time, kf2.time, getattr(kf1, field), getattr(kf2, field), \
            name_one=kf1.name, name_two=kf2.name, elements=elements, time_units=time_units, units=units, \
            start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, single_lines=single_lines, \
            show_rms=show_rms, leg_scale=leg_scale, legend_loc=legend_loc, second_yscale=second_yscale, \
            return_err=True, **kwargs)
        figs += this_figs
        err[field] = this_err

    # Setup plots
    setup_plots(figs, opts)
    if return_err:
        return (figs, err)
    return figs

#%% plot_velocity
def plot_velocity(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Line of Sight histories."""
    if fields is None:
        fields = {'vel': 'Velocity'}
    out = plot_position(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% plot_innovations
def plot_innovations(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""
    Plots the Kalman Filter innovation histories.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import KfInnov, Opts, plot_innovations
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> num_axes   = 2
    >>> num_innovs = 11

    >>> kf1       = KfInnov()
    >>> kf1.units = 'm'
    >>> kf1.time  = np.arange(num_innovs, dtype=float)
    >>> kf1.innov = 1e-6 * np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
    >>> kf1.norm  = np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)

    >>> ix        = np.hstack((np.arange(7), np.arange(8, num_innovs)))
    >>> kf2       = KfInnov()
    >>> kf2.time  = kf1.time[ix]
    >>> kf2.innov = kf1.innov[:, ix] + 1e-8 * np.random.rand(num_axes, ix.size)
    >>> kf2.norm  = kf1.norm[:, ix] + 0.1 * np.random.rand(num_axes, ix.size)

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.vert_fact = 'micro'
    >>> opts.sub_plots = True

    >>> fig_hand = plot_innovations(kf1, kf2, opts=opts)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = KfInnov()
    if kf2 is None:
        kf2 = KfInnov()
    if truth is None:
        pass # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'innov': 'Innovations', 'norm': 'Normalized Innovations'}

    # aliases and defaults
    description   = kf1.name + ' ' if kf1.name else kf2.name + ' ' if kf2.name else ''
    num_chan      = kf1.innov.shape[0] if kf1.innov is not None else kf2.innov.shape[0] if kf2.innov is not None else 0
    elements      = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f'Channel {i+1}' for i in range(num_chan)]
    elements      = kwargs.pop('elements', elements)
    units         = kwargs.pop('units', kf1.units)
    leg_scale     = kwargs.pop('leg_scale', 'micro')
    (fact, name)  = get_factors(leg_scale)
    second_yscale = kwargs.pop('second_yscale', {name + units: 1/fact})

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # TODO: incorporate status information

    # Initialize outputs
    figs = []
    err  = dict()

    #% call wrapper functions for most of the details
    for (field, sub_description) in fields.items():
        if 'Normalized' in sub_description:
            units = u'σ'
            second_yscale=None
        (this_figs, this_err) = make_difference_plot(description+sub_description, kf1.time, kf2.time, getattr(kf1, field), getattr(kf2, field), \
            name_one=kf1.name, name_two=kf2.name, elements=elements, units=units, time_units=time_units, \
            start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
            make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, single_lines=single_lines, \
            legend_loc=legend_loc, leg_scale=leg_scale, second_yscale=second_yscale, return_err=True, **kwargs)
        figs += this_figs
        err[field] = this_err
    # Setup plots
    setup_plots(figs, opts)
    if return_err:
        return (figs, err)
    return figs

#%% plot_covariance
def plot_covariance(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, groups=None, fields=None, **kwargs):
    r"""
    Plots the Kalman Filter square root diagonal variance value.

    Parameters
    ----------
    kf1 : class Kf
        Kalman filter output
    kf2 : class Kf, optional
        Second filter output for potential comparison
    truth : class Kf, optional
        Third filter output that is considered truth
    opts : class Opts, optional
        Plotting options
    return_err : bool, optional, default is False
        Whether the function should return the error differences in addition to the figure handles

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import Kf, Opts, plot_covariance
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> num_points = 11
    >>> num_states = 6

    >>> kf1        = Kf()
    >>> kf1.name   = 'KF1'
    >>> kf1.time   = np.arange(num_points, dtype=float)
    >>> kf1.covar  = 1e-6 * np.tile(np.arange(1, num_states+1, dtype=float)[:, np.newaxis], (1, num_points))
    >>> kf1.active = np.array([1, 2, 3, 4, 8, 12])

    >>> kf2        = Kf(name='KF2')
    >>> kf2.time   = kf1.time
    >>> kf2.covar  = kf1.covar + 1e-9 * np.random.rand(*kf1.covar.shape)
    >>> kf2.active = kf1.active

    >>> opts = Opts()
    >>> opts.case_name = 'test_plot'
    >>> opts.sub_plots = True

    >>> fig_hand = plot_covariance(kf1, kf2, opts=opts)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = Kf()
    if kf2 is None:
        kf2 = Kf()
    if truth is None:
        pass # Note: truth is not used within this function, but kept for argument consistency
    if opts is None:
        opts = Opts()
    if fields is None:
        fields = {'covar': 'Covariance'}

    # TODO: allow different sets of states in the different structures

    # aliases and defaults
    num_chan = 0
    for key in fields.keys():
        num_chan = max(num_chan, getattr(kf1, key).shape[0] if getattr(kf1, key) is not None else getattr(kf2, key).shape[0] if getattr(kf2, key) is not None else 0)
    elements      = kf1.chan if kf1.chan else kf2.chan if kf2.chan else [f'Channel {i+1}' for i in range(num_chan)]
    elements      = kwargs.pop('elements', elements)
    units         = kwargs.pop('units', 'mixed')
    leg_scale     = kwargs.pop('leg_scale', 'micro')
    (fact, name)  = get_factors(leg_scale)
    second_yscale = kwargs.pop('second_yscale', {name + units: 1/fact})
    if groups is None:
        groups = [i for i in range(num_chan)]

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop('save_plot', this_opts.save_plot)

    # alias opts
    time_units   = kwargs.pop('time_units', this_opts.time_base)
    start_date   = kwargs.pop('start_date', this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop('rms_xmin', this_opts.rms_xmin)
    rms_xmax     = kwargs.pop('rms_xmax', this_opts.rms_xmax)
    disp_xmin    = kwargs.pop('disp_xmin', this_opts.disp_xmin)
    disp_xmax    = kwargs.pop('disp_xmax', this_opts.disp_xmax)
    sub_plots    = kwargs.pop('make_subplots', this_opts.sub_plots)
    single_lines = kwargs.pop('single_lines', this_opts.sing_line)
    use_mean     = kwargs.pop('use_mean', this_opts.use_mean)
    plot_zero    = kwargs.pop('plot_zero', this_opts.show_zero)
    show_rms     = kwargs.pop('show_rms', this_opts.show_rms)
    legend_loc   = kwargs.pop('legend_loc', this_opts.leg_spot)

    # initialize output
    figs = []
    err  = dict()

    #% call wrapper functions for most of the details
    for (field, description) in fields.items():
        err[field] = {}
        for (ix, states) in enumerate(groups):
            states     = np.atleast_1d(states)
            data_one   = np.atleast_2d(getattr(kf1, field)[states, :]) if getattr(kf1, field) is not None else None
            data_two   = np.atleast_2d(getattr(kf2, field)[states, :]) if getattr(kf2, field) is not None else None
            have_data1 = data_one is not None and np.any(~np.isnan(data_one))
            have_data2 = data_two is not None and np.any(~np.isnan(data_two))
            if have_data1 or have_data2:
                this_elements = [elements[state] for state in states]
                (this_figs, this_err) = make_difference_plot(description, kf1.time, kf2.time, data_one, data_two, \
                    name_one=kf1.name, name_two=kf2.name, elements=this_elements, units=units, time_units=time_units, \
                    start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, disp_xmax=disp_xmax, \
                    make_subplots=sub_plots, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, single_lines=single_lines, \
                    legend_loc=legend_loc, leg_scale=leg_scale, second_yscale=second_yscale, return_err=True, **kwargs)
                figs += this_figs
                err[field][f'Group {ix}'] = this_err
    # Setup plots
    setup_plots(figs, opts)
    if not figs:
        print('No covariance data was provided, so no plots were generated.')
    if return_err:
        return (figs, err)
    return figs

#%% plot_states
def plot_states(kf1=None, kf2=None, *, truth=None, opts=None, return_err=False, fields=None, **kwargs):
    r"""Plots the Kalman Filter state histories."""
    if fields is None:
        fields = {'state': 'State Estimates'}
    out = plot_covariance(kf1, kf2, truth=truth, opts=opts, return_err=return_err, fields=fields, **kwargs)
    return out

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_kalman', exit=False)
    doctest.testmod(verbose=False)
