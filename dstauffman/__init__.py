# -*- coding: utf-8 -*-
r"""
The "dstauffman" module is a generic Python code library of useful functions.

At least they are functions that I (David C. Stauffer) have found useful.  Your results may vary!

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Updated by David C. Stauffer in December 2015 to no longer support Python v2.7.  Too many of the
    newer language features were appealing and couldn't be used if compatibility was a concern.

"""

#%% Imports
from .classes      import Frozen, SaveAndLoad, SaveAndLoadPickle, Counter, FixedDict
from .constants    import DEFAULT_COLORMAP, INT_TOKEN, IS_WINDOWS, MONTHS_PER_YEAR, QUAT_SIZE, \
                              DEFAULT_CLASSIFICATION
from .enums        import IntEnumPlus, consecutive, ReturnCodes
from .estimation   import OptiOpts, OptiParam, BpeResults, CurrentResults, validate_opti_opts, \
                              run_bpe, plot_bpe_convergence, plot_bpe_results
from .fortran      import create_fortran_unit_tests, create_fortran_makefile
from .health       import dist_enum_and_mons, icer, plot_icer, plot_population_pyramid
from .kalman       import KfInnov, KfOut, plot_attitude, plot_position, plot_innovation, \
                              plot_covariance, plot_states
from .latex        import make_preamble, make_conclusion, bins_to_str_ranges, latex_str
from .linalg       import orth, subspace
from .logs         import activate_logging, deactivate_logging
from .parser       import main, parse_wrapper, parse_commands, execute_command
from .paths        import get_root_dir, get_tests_dir, get_data_dir, get_images_dir, \
                              get_output_dir, list_python_files
from .plot_generic import make_time_plot, make_error_bar_plot, make_quaternion_plot, \
                              make_difference_plot
from .plot_health  import plot_health_time_history, plot_health_monte_carlo
from .plot_support import Plotter, TruthPlotter, MyCustomToolbar, ColorMap, close_all, \
                              get_color_lists, ignore_plot_data, whitten, resolve_name, storefig, \
                              titleprefix, disp_xlimits, zoom_ylim, setup_plots, figmenu, \
                              rgb_ints_to_hex, get_screen_resolution, show_zero_ylim, \
                              plot_second_units_wrapper, plot_second_yunits, get_rms_indices, \
                              plot_vert_lines, plot_phases, get_classification, \
                              plot_classification, is_datetime, align_plots
from .plotting     import Opts, plot_time_history, plot_correlation_matrix, plot_bar_breakdown
from .quat         import USE_ASSERTIONS, qrot, quat_angle_diff, quat_from_euler, quat_interp, \
                              quat_inv, quat_mult, quat_norm, quat_prop, quat_times_vector, \
                              quat_to_dcm, quat_to_euler
from .repos        import run_docstrings, run_unittests, run_pytests, run_coverage, \
                              find_repo_issues, delete_pyc, get_python_definitions, make_python_init
from .stats        import convert_annual_to_monthly_probability, \
                              convert_monthly_to_annual_probability, ca2mp, cm2ap, prob_to_rate, \
                              rate_to_prob, annual_rate_to_monthly_probability, \
                              monthly_probability_to_annual_rate, ar2mp, mp2ar, combine_sets, \
                              bounded_normal_draw, z_from_ci, rand_draw, intersect
from .time         import get_np_time_units, round_datetime, round_np_datetime
from .units        import get_factors
from .utils        import rms, rss, setup_dir, compare_two_classes, compare_two_dicts, \
                              read_text_file, write_text_file, capture_output, unit, modd, \
                              is_np_int, np_digitize, histcounts, full_print, pprint_dict, \
                              line_wrap, combine_per_year, execute, execute_wrapper, get_env_var, \
                              fix_rollover
from .vectors      import rot

#%% Unit test
if __name__ == '__main__':
    pass
