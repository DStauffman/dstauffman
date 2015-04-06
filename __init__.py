# -*- coding: utf-8 -*-
r"""
The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer)
have found useful.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imports
from .classes   import frozen, Frozen
from .constants import MONTHS_PER_YEAR
from .plotting  import Opts, storefig, titleprefix
from .photos    import find_missing_nums, find_unexpected_ext, rename_old_picasa_files, \
                           rename_upper_ext, find_long_filenames, batch_resize
from .utils     import rms, setup_dir, compare_two_classes, compare_two_dicts, round_time, \
                           make_python_init, get_python_definitions, read_text_file, \
                           write_text_file, disp, convert_annual_to_monthly_probability, \
                           get_root_dir, get_tests_dir, get_data_dir, capture_output

#%% Unit test
if __name__ == '__main__':
    pass
