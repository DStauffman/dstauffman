# -*- coding: utf-8 -*-
r"""
The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer)
have found useful.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from .classes   import frozen, Frozen
from .constants import MONTHS_PER_YEAR, INT_TOKEN, DEFAULT_COLORMAP, spyder_custom_colors
from .enums     import IntEnumPlus, dist_enum_and_mons
from .plotting  import Opts, MyCustomToolbar, ColorMap, plot_time_history, \
                           plot_correlation_matrix, storefig, titleprefix, disp_xlimits, \
                           setup_plots, figmenu
from .photos    import find_missing_nums, find_unexpected_ext, rename_old_picasa_files, \
                           rename_upper_ext, find_long_filenames, batch_resize, convert_tif_to_jpg
from .quat      import qrot, quat_angle_diff, quat_from_euler, quat_interp, quat_inv, quat_mult, \
                       quat_norm, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler, \
                       concat_vectors
from .utils     import rms, setup_dir, compare_two_classes, compare_two_dicts, round_time, \
                           make_python_init, get_python_definitions, read_text_file, \
                           write_text_file, disp, convert_annual_to_monthly_probability, \
                           convert_monthly_to_annual_probability, ca2mp, cm2ap, get_root_dir, \
                           get_tests_dir, get_data_dir, get_images_dir, capture_output, unit, \
                           nonzero_indices, combine_sets, reload_package, delete_pyc

#%% Unit test
if __name__ == '__main__':
    pass
