r"""
dstauffman code related to estimation.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
"""

#%% Imports
from .batch    import OptiOpts, OptiParam, BpeResults, CurrentResults, validate_opti_opts, \
                          run_bpe
from .kalman   import calculate_kalman_gain, propagate_covariance, update_covariance
from .linalg   import orth, subspace, mat_divide
from .smoother import bf_smoother
from .support  import get_parameter, set_parameter

#%% Unittest
if __name__ == '__main__':
    pass
