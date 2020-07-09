r"""
Classes related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
import doctest
import unittest

import numpy as np

from dstauffman import Frozen

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
    >>> from dstauffman.aerospace import KfInnov
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
    >>> from dstauffman.aerospace import Kf
    >>> kf = Kf()

    """
    def __init__(self, *, name='', num_points=0, num_states=0, time_dtype=float, active_states=None, innov_class=None, **kwargs):
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
        else:
            self.time   = None
            self.att    = None
            self.pos    = None
            self.vel    = None
            self.active = None
            self.state  = None
            self.istate = None
            self.covar  = None
        if innov_class is None:
            self.innov  = KfInnov(time_dtype=time_dtype, **kwargs)
        else:
            self.innov  = innov_class(time_dtype=time_dtype, **kwargs)

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_classes', exit=False)
    doctest.testmod(verbose=False)
