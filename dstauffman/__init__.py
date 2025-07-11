r"""
The "dstauffman" module is a generic Python code library of useful functions.

At least they are functions that I (David C. Stauffer) have found useful.  Your results may vary!
"""

# %% Imports
# fmt: off
from .binary    import int2bin, int2hex, split_bits, read_bit_stream
from .classes   import save_hdf5, load_hdf5, save_method, load_method, save_convert_hdf5, \
                           save_restore_hdf5, pprint, pprint_dict, chop_time, subsample_class, \
                           Frozen, SaveAndLoad
from .constants import HAVE_COVERAGE, HAVE_H5PY, HAVE_MPL, HAVE_DS, HAVE_NUMPY, HAVE_KERAS, \
                           HAVE_PANDAS, HAVE_PYTEST, HAVE_SCIPY, INT_TOKEN, IS_WINDOWS, \
                           NP_DATETIME_UNITS, NP_DATETIME_FORM, NP_TIMEDELTA_FORM, \
                           NP_INT64_PER_SEC, NP_ONE_SECOND, NP_ONE_MINUTE, NP_ONE_HOUR, \
                           NP_ONE_DAY, NP_NAT, NP_DATETIME_MIN, NP_DATETIME_MAX
from .matlab    import load_matlab, orth, subspace, ecdf, mat_divide, find_first, find_last, \
                           prepend, postpend
from .multipass import MultipassExceptionWrapper, parfor_wrapper
from .parser    import main, parse_wrapper, parse_commands, execute_command, \
                           process_command_line_options
from .paths     import get_root_dir, get_tests_dir, get_data_dir, get_images_dir, get_output_dir
from .repos     import run_docstrings, run_unittests, run_pytests, run_coverage
from .stats     import convert_annual_to_monthly_probability, \
                           convert_monthly_to_annual_probability, ca2mp, cm2ap, prob_to_rate, \
                           rate_to_prob, annual_rate_to_monthly_probability, \
                           monthly_probability_to_annual_rate, ar2mp, mp2ar, rand_draw, \
                           apply_prob_to_mask
from .time      import get_np_time_units, get_ymd_from_np, round_datetime, round_np_datetime, \
                           round_num_datetime, round_time, convert_date, convert_time_units, \
                           convert_datetime_to_np, convert_duration_to_np, convert_num_dt_to_np, \
                           get_delta_time_str
from .units     import ONE_MINUTE, ONE_HOUR, ONE_DAY, MONTHS_PER_YEAR, RAD2DEG, DEG2RAD, \
                           ARCSEC2RAD, RAD2ARCSEC, FT2M, M2FT, IN2CM, CM2IN, DEGREE_SIGN, \
                           MICRO_SIGN, get_factors, get_time_factor, get_unit_conversion
from .utils     import find_in_range, rms, rss, compare_two_classes, compare_two_dicts, \
                           magnitude, unit, modd, is_np_int, np_digitize, histcounts, full_print, \
                           combine_per_year, execute, execute_wrapper, get_env_var, get_username, \
                           is_datetime, intersect, issorted, zero_order_hold, linear_interp, \
                           linear_lowpass_interp, drop_following_time
from .utils_log import fix_rollover, remove_outliers
from .version   import version_info
# fmt: on

# %% Constants
__version__ = ".".join(str(x) for x in version_info)

# %% Unit test
if __name__ == "__main__":
    pass
