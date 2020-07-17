r"""
dstauffman code related to Kalman filters, including quaternions and vector math.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
"""

#%% Imports
from .classes  import KfInnov, Kf, KfRecord
from .kalman   import calc_kalman_gain, propagate_covariance, update_covariance
from .plotting import make_quaternion_plot, plot_attitude, plot_los, plot_position, plot_velocity, \
                          plot_innovations, plot_covariance, plot_states
from .quat     import QUAT_SIZE, USE_ASSERTIONS, quat_assertions, qrot, quat_angle_diff, quat_from_euler, \
                          quat_interp, quat_inv, quat_mult, quat_norm, quat_prop, \
                          quat_times_vector, quat_to_dcm, quat_to_euler
from .vectors  import rot, vec_cross

#%% Unittest
if __name__ == '__main__':
    pass
