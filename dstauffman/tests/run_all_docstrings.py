# -*- coding: utf-8 -*-
r"""
Test file to execute all the docstrings within the dstauffman code.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import doctest
import os

import dstauffman as dcs

#%% Execution
if __name__ == '__main__':
    folders = [dcs.get_root_dir(), os.path.join(dcs.get_root_dir(), 'commands')]
    dcs.run_doctests(folders, verbose=False)
