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

#%% KfInnov
class KfInnov(Frozen):
    r"""
    A class for Kalman Filter innovations from a sensor.

    Attributes
    ----------
    name : str
        Name of the innovation structure, often specifying which sensor it comes from, like GPS
    chan : list of str
        Names of the different axes found within the innovations
    time : (N, ) ndarray
        Time vector
    innov : (N, M) ndarray
        Time history of the raw innovations
    norm : (N, M) ndarray
        Time history of the normalized innovations

    Examples
    --------
    >>> from dstauffman import KfInnov
    >>> innov = KfInnov()

    """
    def __init__(self):
        r"""Initializes a new KfInnov instance."""
        self.name   = ''
        self.chan   = None
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
    def __init__(self):
        r"""Initializes a new Kf instance."""
        self.name  = ''
        self.chan  = None
        self.time  = None
        self.att   = None
        self.pos   = None
        self.vel   = None
        self.innov = KfInnov()
        self.state = None
        self.covar = None

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
def propagate_covariance(phi, P, Q, gamma=None):
    r"""
    Propagates the covariance forward in time.

    Parameters
    ----------
    phi :
        State transition matrix
    P :
        Covariance matrix
    Q :
        Process noise matrix
    gamma :
        Shaping matrix?

    Returns
    -------
    (N, N) ndarray
        Updated covariance matrix

    Notes
    -----
    #.  Written by David C. Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman import propagate_covariance
    >>> import numpy as np
    >>> phi = np.diag([1., 1, 1, -1, -1, -1])
    >>> P = 1e-3 * np.eye(6)
    >>> Q = np.diag([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7])
    >>> cov_out = propagate_covariance(phi, P, Q)
    >>> print(cov_out[0, 0])
    0.002

    """
    if gamma is None:
        return phi @ P @ phi.T + Q
    return phi @ P @ phi.T + gamma @ Q @ gamma.T

#%% Functions - update_covariance
def update_covariance(P, K, H):
    r"""
    Updates the covariance for a given measurement.

    Parameters
    ----------
    P : (N, N) ndarray
        Covariance Matrix
    K : (N, ) ndarray
        Kalman Gain Matrix
    H : (A, N) Measurement Update Matrix

    Returns
    -------
    P_out : (N, N) ndarray
        Updated Covariance Matrix

    Notes
    -----
    #.  Written by David C Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman import update_covariance
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(6)
    >>> K = np.array([])
    >>> H = np.array([])
    >>> P_out = update_covariance(P, K, H)
    >>> print(P_out[0, 0])
    0.001

    """
    return (np.eye(*P.shape) - K @ H) @ P

#%% plot_attitude
def plot_attitude(kf1=None, kf2=None, truth=None, *, config=None, opts=Opts(), return_err=False):
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
    config : dict, optional
        Configuration information
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
    >>> from dstauffman import Kf, plot_attitude, quat_norm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> kf1      = Kf()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.att  = quat_norm(np.random.rand(4, 11))

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.att  = quat_norm(np.random.rand(4, 11))

    >>> fig_hand = plot_attitude(kf1, kf2)

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

    # call wrapper function for most of the details
    (figs, err) = make_quaternion_plot('Attitude Quaternion', kf1.time, kf2.time, kf1.att, kf2.att,
        name_one=kf1.name, name_two=kf2.name, time_units=opts.time_base, start_date=opts.get_date_zero_str(), \
        rms_xmin=opts.rms_xmin, rms_xmax=opts.rms_xmax, disp_xmin=opts.disp_xmin, disp_xmax=opts.disp_xmax, \
        make_subplots=opts.sub_plots, plot_components=opts.quat_comp, \
        use_mean=opts.use_mean, plot_zero=opts.show_zero, show_rms=opts.show_rms, legend_loc=opts.leg_spot, \
        truth_name=truth.name, truth_time=truth.time, truth_data=truth.att, return_err=True)

    # Setup plots
    setup_plots(figs, opts)
    if return_err:
        return (figs, err)
    return figs

#%% plot_position
def plot_position(kf1=None, kf2=None, truth=None, *, config=None, opts=Opts(), return_err=False):
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
    config : dict, optional
        Configuration information
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
    >>> kf1.vel  = None

    >>> kf2      = Kf()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.pos  = kf1.pos[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e5
    >>> kf2.vel  = None

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

    # hard-coded values
    elements      = ['x', 'y', 'z']
    units         = 'm'
    leg_scale     = 'kilo'
    second_yscale = {'km': 1e-3}
    color_lists   = get_color_lists()
    colormap      = ListedColormap(color_lists['vec_diff'].colors + color_lists['vec'].colors)

    # call wrapper function for most of the details
    (figs, err) = make_difference_plot('Position', kf1.time, kf2.time, kf1.pos, kf2.pos,
        name_one=kf1.name, name_two=kf2.name, elements=elements, units=units, leg_scale=leg_scale, \
        start_date=opts.get_date_zero_str(), rms_xmin=opts.rms_xmin, rms_xmax=opts.rms_xmax, \
        disp_xmin=opts.disp_xmin, disp_xmax=opts.disp_xmax, make_subplots=opts.sub_plots, \
        colormap=colormap, use_mean=opts.use_mean, plot_zero=opts.show_zero, \
        show_rms=opts.show_rms, legend_loc=opts.leg_spot, second_yscale=second_yscale, return_err=True)
    # Setup plots
    setup_plots(figs, opts)
    if return_err:
        return (figs, err)
    return figs

#%% plot_innovation
def plot_innovation(kf1=None, kf2=None, opts=Opts(), return_err=False):
    pass # TODO: write this

#%% plot_covariance
def plot_covariance(kf1=None, kf2=None, opts=Opts(), return_err=False):
    pass # TODO: write this

#%% plot_states
def plot_states(kf1=None, kf2=None, opts=Opts(), return_err=False):
    pass # TODO: write this

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_kalman', exit=False)
    doctest.testmod(verbose=False)
