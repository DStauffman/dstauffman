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
"""

#%% Imports
from .classes      import save_hdf5, load_hdf5, save_pickle, load_pickle, save_method, \
                              load_method, pprint_dict, Frozen, SaveAndLoad, SaveAndLoadPickle, \
                              Counter, FixedDict
from .constants    import DEFAULT_COLORMAP, INT_TOKEN, IS_WINDOWS, DEFAULT_CLASSIFICATION
from .enums        import IntEnumPlus, consecutive, ReturnCodes
from .fortran      import create_fortran_unit_tests, create_fortran_makefile
from .logs         import activate_logging, deactivate_logging
from .matlab       import load_matlab
from .optimized    import np_any, np_all
from .parser       import main, parse_wrapper, parse_commands, execute_command
from .paths        import get_root_dir, get_tests_dir, get_data_dir, get_images_dir, \
                              get_output_dir, list_python_files
from .plot_generic import make_time_plot, make_error_bar_plot, make_difference_plot
from .plot_support import Plotter, TruthPlotter, MyCustomToolbar, ColorMap, close_all, \
                              get_color_lists, ignore_plot_data, whitten, resolve_name, storefig, \
                              titleprefix, disp_xlimits, zoom_ylim, setup_plots, figmenu, \
                              rgb_ints_to_hex, get_screen_resolution, show_zero_ylim, \
                              plot_second_units_wrapper, plot_second_yunits, get_rms_indices, \
                              plot_vert_lines, plot_phases, get_classification, \
                              plot_classification, align_plots
from .plotting     import Opts, plot_time_history, plot_correlation_matrix, plot_bar_breakdown
from .repos        import run_docstrings, run_unittests, run_pytests, run_coverage, \
                              find_repo_issues, delete_pyc, get_python_definitions, make_python_init
from .time         import convert_time_units, get_np_time_units, round_datetime, \
                              round_np_datetime, round_num_datetime, convert_date
from .units        import ONE_MINUTE, ONE_HOUR, ONE_DAY, MONTHS_PER_YEAR, RAD2DEG, DEG2RAD, \
                              ARCSEC2RAD, RAD2ARCSEC, FT2M, M2FT, IN2CM, CM2IN, get_factors, \
                              get_time_factor
from .utils        import rms, rss, compare_two_classes, compare_two_dicts, read_text_file, \
                              write_text_file, capture_output, unit, modd, is_np_int, np_digitize, \
                              histcounts, full_print, line_wrap, combine_per_year, \
                              execute, execute_wrapper, get_env_var, is_datetime, intersect, \
                              issorted, zero_order_hold
from .utils_log    import setup_dir, fix_rollover

#%% Unit test
if __name__ == '__main__':
    pass
