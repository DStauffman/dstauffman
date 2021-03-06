r"""
dstauffman code related to Kalman filters, including quaternions and vector math.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
#.  Updated by David C. Stauffer in February 2021 to include optimized versions for single
        quaternions.
"""

#%% Imports
from .classes    import KfInnov, Kf, KfRecord
from .quat       import QUAT_SIZE, suppress_quat_checks, unsuppress_quat_checks, quat_assertions, \
                            qrot, quat_angle_diff, quat_from_euler, quat_interp, quat_inv, \
                            quat_mult, quat_norm, quat_prop, quat_times_vector, quat_to_euler
from .quat_opt   import qrot_single, quat_interp_single, quat_inv_single, quat_mult_single, \
                            quat_norm_single, quat_prop_single, quat_times_vector_single, \
                            quat_to_dcm
from .vectors    import rot, drot, vec_cross, vec_angle, cart2sph, sph2cart, rv2dcm

#%% Unittest
if __name__ == '__main__':
    pass
