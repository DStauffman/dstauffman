r"""
Functions and decorators for dealing with and without having numba.

Notes
-----
#.  Written by David C. Stauffer in February 2021.
"""

#%% Imports
# fmt: off
from .numpy_mods  import issorted_ascend, issorted_descend, np_all_axis0, np_all_axis1, \
                             np_any_axis0, np_any_axis1
from .optimized   import np_any, np_all, issorted_opt, prob_to_rate_opt, rate_to_prob_opt, \
                             zero_divide
from .passthrough import fake_jit, HAVE_NUMBA, ncjit, TARGET
# fmt: on

#%% Unit test
if __name__ == "__main__":
    pass
