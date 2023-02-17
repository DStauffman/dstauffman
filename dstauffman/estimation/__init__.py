r"""
dstauffman code related to estimation.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
"""

# %% Imports
# fmt: off
from .batch    import OptiOpts, OptiParam, BpeResults, CurrentResults, validate_opti_opts, run_bpe
from .kalman   import calculate_kalman_gain, calculate_kalman_gain_opt, calculate_prediction, \
                          calculate_innovation, calculate_normalized_innovation, \
                          calculate_delta_state, propagate_covariance, propagate_covariance_opt, \
                          update_covariance, update_covariance_opt
from .smoother import bf_smoother
from .support  import get_parameter, set_parameter
# fmt: on

# %% Unit test
if __name__ == "__main__":
    pass
