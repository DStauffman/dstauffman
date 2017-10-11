# -*- coding: utf-8 -*-
r"""
The "dstauffman" module is a generic Python code library of useful functions.

At least they are function that I (David C. Stauffer) have found useful.  Your results may vary!

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Updated by David C. Stauffer in December 2015 to no longer support Python v2.7.  Too many of the
    newer language features were appealing and couldn't be used if compatibility was a concern.

"""

#%% Imports
from .bpe       import OptiOpts, OptiParam, BpeResults, CurrentResults, \
                           validate_opti_opts, run_bpe, plot_bpe_results
from .classes   import Frozen, SaveAndLoad, SaveAndLoadPickle, Counter, FixedDict
from .constants import MONTHS_PER_YEAR, INT_TOKEN, DEFAULT_COLORMAP, QUAT_SIZE
from .enums     import IntEnumPlus, consecutive, dist_enum_and_mons
from .latex     import make_preamble, make_conclusion, bins_to_str_ranges
from .linalg    import orth, subspace
from .plotting  import Plotter, Opts, TruthPlotter, MyCustomToolbar, ColorMap, close_all, \
                           ignore_plot_data, get_axes_scales, whitten, plot_time_history, \
                           plot_correlation_matrix, plot_multiline_history, plot_bar_breakdown, \
                           plot_bpe_convergence, plot_population_pyramid, storefig, titleprefix, \
                           disp_xlimits, setup_plots, figmenu, rgb_ints_to_hex
from .quat      import qrot, quat_angle_diff, quat_from_euler, quat_interp, quat_inv, quat_mult, \
                           quat_norm, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler
from .stats     import convert_annual_to_monthly_probability, \
                           convert_monthly_to_annual_probability, ca2mp, cm2ap, prob_to_rate, \
                           rate_to_prob, month_prob_mult_ratio, \
                           annual_rate_to_monthly_probability, monthly_probability_to_annual_rate, \
                           ar2mp, mp2ar, combine_sets, icer, bounded_normal_draw, z_from_ci
from .units     import Units, get_factors
from .utils     import rms, rss, setup_dir, compare_two_classes, compare_two_dicts, round_time, \
                           make_python_init, get_python_definitions, read_text_file, \
                           write_text_file, get_root_dir, get_tests_dir, get_data_dir, \
                           get_images_dir, get_output_dir, capture_output, unit, reload_package, \
                           delete_pyc, rename_module, modd, find_tabs, np_digitize, full_print, \
                           pprint_dict, line_wrap, combine_per_year, activate_logging, \
                           deactivate_logging

#%% Unit test
if __name__ == '__main__':
    pass
