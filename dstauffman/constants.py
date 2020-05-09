# -*- coding: utf-8 -*-
r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

#%% Set error state for module
np.seterr(invalid='ignore', divide='ignore')

#%% Set NumPy printing options
np.set_printoptions(threshold=1000)

#%% Set Matplotlib global settings
plt.rcParams['figure.dpi']     = 160 # 160 for 4K monitors, 100 otherwise
plt.rcParams['figure.figsize'] = [11., 8.5] # makes figures the same size as the paper, keeping aspect ratios even

#%% Constants
# Default colormap to use on certain plots
DEFAULT_COLORMAP = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'

# A specified integer token value to use when you need one
INT_TOKEN = -1

# Whether we are currently on Windows or not
IS_WINDOWS = os.name == 'nt'

# Number of months in a year
MONTHS_PER_YEAR = 12

# Number of elements that should be in a quaternion
QUAT_SIZE = 4

# Whether to include a classification on any generated plots
DEFAULT_CLASSIFICATION = ''

#%% Functions
# None

#%% Script
if __name__ == '__main__':
    # print all the constants (anything in ALL_CAPS)
    fields = [k for k in sorted(dir()) if k.isupper()]
    max_len = max(map(len, fields))
    for field in fields:
        pad = ' ' * (max_len - len(field))
        value = locals()[field]
        print('{}{} = {}'.format(field, pad, value))

    # run unittests
    unittest.main(module='dstauffman.tests.test_constants', exit=False)
