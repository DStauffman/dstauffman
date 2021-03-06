r"""
Test file for the `smoother` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
from dstauffman.aerospace import KfRecord

import dstauffman.estimation as estm

if HAVE_NUMPY:
    import numpy as np

#%% estimation._update_information
class Test_estimation__update_information(unittest.TestCase):
    r"""
    Tests the estimation._update_information function with the following cases:
        TBD
    """
    def setUp(self):
        num_states = 6
        num_axes = 2
        self.H = np.ones((num_axes, num_states), order='F')
        self.Pz = np.eye(num_axes, num_axes, order='F')
        self.K = np.ones((num_states, num_axes), order='F')
        self.z = np.ones(num_axes)
        self.lambda_bar = np.ones(num_states)
        self.LAMBDA_bar = np.ones((num_states, num_states))

    def test_nominal(self):
        (lambda_hat, LAMBDA_hat) = estm.smoother._update_information(self.H, self.Pz, self.z, \
            self.K, self.lambda_bar, self.LAMBDA_bar)
        # TODO: assert something

#%% estimation.bf_smoother
class Test_estimation_bf_smoother(unittest.TestCase):
    r"""
    Tests the estimation.bf_smoother function with the following cases:
        TBD
    """
    def setUp(self):
        num_points = 5
        num_states = 6
        num_axes = 2
        stm = np.eye(num_states, order='F')
        P = np.eye(num_states, order='F')
        H = np.ones((num_axes, num_states), order='F')
        Pz = np.eye(num_axes, num_axes, order='F')
        K = np.ones((num_states, num_axes), order='F')
        z = np.ones(num_axes)
        self.lambda_bar_final = np.ones(num_states)
        self.kf_record = KfRecord(num_points=num_points, num_active=num_states, num_states=num_states, num_axes=num_axes)
        for i in range(num_points):
            self.kf_record.time[i] = float(num_points)
            self.kf_record.stm[:, :, i] = stm
            self.kf_record.P[:, :, i] = P
            self.kf_record.H[:, :, i] = H
            self.kf_record.Pz[:, :, i] = Pz
            self.kf_record.K[:, :, i] = K
            self.kf_record.z[:, i] = z

    def test_nominal(self):
        (x_delta, lambda_bar_initial, LAMBDA_bar_initial) = estm.bf_smoother(self.kf_record)
        # TODO: assert something

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
