r"""
dstauffman code related to Kalman filters, including quaternions and vector math.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
#.  Updated by David C. Stauffer in February 2021 to include optimized versions for single
        quaternions.
"""

# %% Imports
# fmt: off
from .classes     import KfInnov, Kf, KfRecord
from .earth       import geod2ecf, ecf2geod, find_earth_intersect, find_earth_intersect_wrapper
from .gps         import GPS_DATE_ZERO, ONE_DAY, DAYS_PER_WEEK, WEEK_ROLLOVER, NP_GPS_DATE_ZERO, \
                             bsl, bsr, prn_01_to_m11, get_prn_bits, correlate_prn, generate_prn, \
                             gps_to_datetime, get_gps_to_utc_offset, gps_to_utc_datetime
from .orbit_const import PI, TAU, G, SS_MASSES, SIDEREAL_DAY, SIDEREAL_YEAR, AU, MU_SUN, \
                             MU_EARTH, JULIAN, SPEED_OF_LIGHT, ECLIPTIC, EARTH, PALO_ALTO
from .orbit_conv  import anomaly_eccentric_2_mean, anomaly_eccentric_2_true, \
                             anomaly_hyperbolic_2_mean, anomaly_hyperbolic_2_true, \
                             anomaly_mean_2_eccentric, anomaly_mean_2_true, \
                             anomaly_true_2_eccentric, anomaly_true_2_hyperbolic, \
                             anomaly_true_2_mean, mean_motion_2_semimajor, period_2_semimajor, \
                             semimajor_2_mean_motion, semimajor_2_period, sidereal_2_long, \
                             raan_2_mltan, jd_2_sidereal, quat_eci_2_ecf_approx, quat_eci_2_ecf
from .orbit_support import d_2_r, r_2_d, norm, dot, cross, jd_to_numpy, numpy_to_jd, jd_2_century, \
                             mjd_to_numpy, numpy_to_mjd, d_2_dms, dms_2_d, hms_2_r, r_2_hms, \
                             aer_2_rdr, aer_2_sez, geo_loc_2_ijk, ijk_2_rdr, ijk_2_sez, \
                             long_2_sidereal, rdr_2_aer, rdr_2_ijk, sez_2_aer, sez_2_ijk, \
                             rv_aer_2_ijk, rv_aer_2_sez, rv_ijk_2_aer, rv_ijk_2_sez, rv_sez_2_aer, \
                             rv_sez_2_ijk, get_sun_radec_approx, get_sun_radec, get_sun_distance, \
                             beta_from_oe, eclipse_fraction, earth_radius_by_latitude
from .orbits      import OrbitType, Elements, two_line_elements, rv_2_oe, oe_2_rv, advance_elements
from .quat        import QUAT_SIZE, QuatAssertionError, suppress_quat_checks, \
                             unsuppress_quat_checks, quat_assertions, enforce_pos_scalar, qrot, \
                             quat_from_axis_angle, quat_from_rotation_vector, quat_angle_diff, \
                             quat_from_euler, quat_interp, quat_inv, quat_mult, quat_norm, \
                             quat_prop, quat_times_vector, quat_to_euler, \
                             convert_att_quat_to_body_rate, quat_standards
from .quat_keras  import enforce_pos_scalar_keras, qrot_keras, quat_inv_keras, quat_norm_keras, \
                             quat_mult_keras, quat_prop_keras, quat_times_vector_keras, \
                             quat_angle_diff_keras
from .quat_opt    import qrot_single, quat_from_axis_angle_single, \
                             quat_from_rotation_vector_single, quat_angle_diff_single, \
                             quat_interp_single, quat_inv_single, quat_mult_single, \
                             quat_norm_single, quat_prop_single, quat_times_vector_single, \
                             quat_to_dcm
from .vectors     import rot, drot, vec_cross, vec_angle, cart2sph, sph2cart, rv2dcm, interp_vector
from .weather     import read_tci_data, read_kp_ap_etc_data, read_kp_ap_nowcast, read_solar_cycles
# fmt: on

# %% Unit test
if __name__ == "__main__":
    pass
