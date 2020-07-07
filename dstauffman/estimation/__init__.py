r"""
dstauffman code related to estimation.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Moved into subfolder by David C. Stauffer in July 2020.
"""

#%% Imports
from .batch  import OptiOpts, OptiParam, BpeResults, CurrentResults, validate_opti_opts, \
                        run_bpe, plot_bpe_convergence, plot_bpe_results
from .linalg import orth, subspace

#%% Unittest
if __name__ == '__main__':
    pass
