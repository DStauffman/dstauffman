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

from dstauffman.classes import Frozen
from dstauffman.plot_support import get_color_lists, setup_plots
from dstauffman.plotting import general_difference_plot, general_quaternion_plot, Opts

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
        r"""
        Initializes a new KfInnov instance.
        """
        self.name  = ''
        self.chan  = None
        self.time  = None
        self.innov = None
        self.norm  = None

#%% KfOut
class KfOut(Frozen):
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
    >>> from dstauffman import KfOut
    >>> kf = KfOut()

    """
    def __init__(self):
        r"""
        Initializes a new KfOut instance.
        """
        self.name  = ''
        self.chan  = None
        self.time  = None
        self.att   = None
        self.pos   = None
        self.vel   = None
        self.innov = KfInnov()
        self.state = None
        self.covar = None

#%% plot_attitude
def plot_attitude(kf1=None, kf2=None, truth=None, *, config=None, opts=Opts()):
    r"""
    Plots the attitude quaternion history.

    Parameters
    ----------
    kf1 : class KfOut
        Kalman filter output
    kf2 : class KfOut, optional
        Second filter output for potential comparison
    truth : class KfOut, optional
        Third filter output that is considered truth
    config : dict, optional
        Configuration information
    opts : class Opts, optional
        Plotting options

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import KfOut, plot_attitude, quat_norm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> kf1      = KfOut()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.att  = quat_norm(np.random.rand(4, 11))

    >>> kf2      = KfOut()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.att  = quat_norm(np.random.rand(4, 11))

    >>> (fig_hand, err) = plot_attitude(kf1, kf2)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = KfOut()
    if kf2 is None:
        kf2 = KfOut()
    if truth is None:
        truth = KfOut()
    if opts is None:
        opts = Opts()

    # call wrapper function for most of the details
    (figs, err) = general_quaternion_plot('Attitude Quaternion', kf1.time, kf2.time, kf1.att, kf2.att,
        name_one=kf1.name, name_two=kf2.name, time_units=opts.time_base, start_date=opts.get_date_zero_str(), \
        rms_xmin=opts.rms_xmin, rms_xmax=opts.rms_xmax, disp_xmin=opts.disp_xmin, disp_xmax=opts.disp_xmax, \
        fig_visible=opts.show_plot, make_subplots=opts.sub_plots, plot_components=opts.quat_comp, \
        use_mean=opts.use_mean, plot_zero=opts.show_zero, show_rms=opts.show_rms, legend_loc=opts.leg_spot, \
        truth_name=truth.name, truth_time=truth.time, truth_data=truth.att)

    # Setup plots
    setup_plots(figs, opts, 'time')
    return (figs, err)

#%% plot_position
def plot_position(kf1=None, kf2=None, truth=None, *, config=None, opts=Opts()):
    r"""
    Plots the position and velocity history.

    Parameters
    ----------
    kf1 : class KfOut
        Kalman filter output
    kf2 : class KfOut, optional
        Second filter output for potential comparison
    truth : class KfOut, optional
        Third filter output that is considered truth
    config : dict, optional
        Configuration information
    opts : class Opts, optional
        Plotting options

    Returns
    -------
    fig_hand : list of class matplotlib.figure.Figure
        Figure handles
    err : dict
        Numerical outputs of comparison

    Examples
    --------
    >>> from dstauffman import KfOut, plot_position
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> kf1      = KfOut()
    >>> kf1.name = 'KF1'
    >>> kf1.time = np.arange(11)
    >>> kf1.pos  = 1e6 * np.random.rand(3, 11)
    >>> kf1.vel  = None

    >>> kf2      = KfOut()
    >>> kf2.name = 'KF2'
    >>> kf2.time = np.arange(2, 13)
    >>> kf2.pos  = kf1.pos[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e5
    >>> kf2.vel  = None

    >>> (fig_hand, err) = plot_position(kf1, kf2)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # check optional inputs
    if kf1 is None:
        kf1 = KfOut()
    if kf2 is None:
        kf2 = KfOut()
    if truth is None:
        truth = KfOut()
    if opts is None:
        opts = Opts()

    # hard-coded values
    elements       = ['x', 'y', 'z']
    units          = 'm'
    leg_scale      = 'kilo'
    second_y_scale = {'m': 1e3}
    color_lists     = get_color_lists()
    colormap        = color_lists['vec']

    # call wrapper function for most of the details
    (figs, err) = general_difference_plot('Position', kf1.time, kf2.time, kf1.pos, kf2.pos,
        name_one=kf1.name, name_two=kf2.name, elements=elements, units=units, leg_scale=leg_scale, \
        start_date=opts.get_date_zero_str(), rms_xmin=opts.rms_xmin, rms_xmax=opts.rms_xmax, \
        disp_xmin=opts.disp_xmin, disp_xmax=opts.disp_xmax, fig_visible=opts.show_plot, \
        make_subplots=opts.sub_plots, colormap=colormap, use_mean=opts.use_mean, \
        plot_zero=opts.show_zero, show_rms=opts.show_rms, legend_loc=opts.leg_spot, \
        second_y_scale=second_y_scale)
    # Setup plots
    setup_plots(figs, opts, 'time')
    return (figs, err)

#%% plot_innovation
def plot_innovation(kf1=None, kf2=None, opts=Opts()):
    pass # TODO: write this

#%% plot_covariance
def plot_covariance(kf1=None, kf2=None, opts=Opts()):
    pass # TODO: write this

#%% plot_states
def plot_states(kf1=None, kf2=None, opts=Opts()):
    pass # TODO: write this

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_kalman', exit=False)
    doctest.testmod(verbose=False)
