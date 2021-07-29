r"""
The "dstauffman" module is a generic Python code library of useful functions.

At least they are functions that I (David C. Stauffer) have found useful.  Your results may vary!

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Updated by David C. Stauffer in December 2015 to no longer support Python v2.7.  Too many of the
    newer language features were appealing and couldn't be used if compatibility was a concern.
#.  Updated by David C. Stauffer in July 2020 to put aerospace, estimation, and health into
    submodules.
#.  Updated by David C. Stauffer in July 2020 to split the plotting portions into a separate
    submodule, which allows for delayed importing of matplotlib.
#.  Updated by David C. Stauffer in November 2020 to work with just core Python, although with very
    limited functionality.
"""

#%% Imports
from .classes   import save_hdf5, load_hdf5, save_pickle, load_pickle, save_method, load_method, \
                           pprint_dict, chop_time, subsample_class, Frozen, SaveAndLoad, \
                           SaveAndLoadPickle, Counter, FixedDict
from .constants import HAVE_COVERAGE, HAVE_H5PY, HAVE_MPL, HAVE_DS, HAVE_NUMPY, HAVE_PANDAS, \
                           HAVE_PYTEST, HAVE_SCIPY, INT_TOKEN, IS_WINDOWS, NP_DATETIME_UNITS, \
                           NP_DATETIME_FORM, NP_TIMEDELTA_FORM, NP_INT64_PER_SEC, NP_ONE_SECOND, \
                           NP_ONE_MINUTE, NP_ONE_HOUR, NP_ONE_DAY
from .enums     import IntEnumPlus, consecutive, ReturnCodes, LogLevel
from .fortran   import create_fortran_unit_tests, create_fortran_makefile
from .logs      import activate_logging, deactivate_logging, flush_logging, log_multiline
from .matlab    import load_matlab
from .multipass import MultipassExceptionWrapper, parfor_wrapper
from .parser    import main, parse_wrapper, parse_commands, execute_command, \
                           process_command_line_options
from .paths     import is_dunder, get_root_dir, get_tests_dir, get_data_dir, get_images_dir, \
                           get_output_dir, list_python_files
from .repos     import run_docstrings, run_unittests, run_pytests, run_coverage, find_repo_issues, \
                           delete_pyc, get_python_definitions, make_python_init, \
                           write_unit_test_templates
from .time      import get_np_time_units, round_datetime, round_np_datetime, round_num_datetime, \
                           round_time, convert_date, convert_time_units, convert_datetime_to_np, \
                           convert_duration_to_np, convert_num_dt_to_np, get_delta_time_str
from .units     import ONE_MINUTE, ONE_HOUR, ONE_DAY, MONTHS_PER_YEAR, RAD2DEG, DEG2RAD, \
                           ARCSEC2RAD, RAD2ARCSEC, FT2M, M2FT, IN2CM, CM2IN, DEGREE_SIGN, \
                           MICRO_SIGN, get_factors, get_time_factor, get_unit_conversion
from .utils     import find_in_range, rms, rss, compare_two_classes, compare_two_dicts, \
                           read_text_file, write_text_file, capture_output, magnitude, unit, \
                           modd, is_np_int, np_digitize, histcounts, full_print, line_wrap, \
                           combine_per_year, execute, execute_wrapper, get_env_var, get_username, \
                           is_datetime, intersect, issorted, zero_order_hold, drop_following_time
from .utils_log import setup_dir, fix_rollover, remove_outliers
from .version   import version_info

#%% Constants
__version__ = '.'.join(str(x) for x in version_info)

#%% Unit test
if __name__ == '__main__':
    pass
