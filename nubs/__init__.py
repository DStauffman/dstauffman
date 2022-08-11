r"""
Functions and decorators for dealing with and without having numba.

Notes
-----
#.  Written by David C. Stauffer in February 2021.
"""

#%% Imports
from .numpy_mods import (
    issorted_ascend as issorted_ascend,
    issorted_descend as issorted_descend,
    np_all_axis0 as np_all_axis0,
    np_all_axis1 as np_all_axis1,
    np_any_axis0 as np_any_axis0,
    np_any_axis1 as np_any_axis1,
)
from .optimized import (
    np_any as np_any,
    np_all as np_all,
    issorted_opt as issorted_opt,
    prob_to_rate_opt as prob_to_rate_opt,
    rate_to_prob_opt as rate_to_prob_opt,
    zero_divide as zero_divide,
)
from .passthrough import (
    fake_jit as fake_jit,
    HAVE_NUMBA as HAVE_NUMBA,
    HAVE_NUMPY as HAVE_NUMPY,
    ncjit as ncjit,
    TARGET as TARGET,
)

#%% Unit test
if __name__ == "__main__":
    pass
