# -*- coding: utf-8 -*-
r"""
The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer)
have found useful.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imports
from .classes      import frozen, Frozen
from .constants    import MONTHS_PER_YEAR
from .plotting     import Opts, storefig, titleprefix
from .utils        import rms, setup_dir, compare_two_structures, round_time, make_python_init, \
                              get_python_definitions, read_text_file, write_text_file, \
                              disp, convert_annual_to_monthly_probability

#%% Unit test
if __name__ == '__main__':
    pass
