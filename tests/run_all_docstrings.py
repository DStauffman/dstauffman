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

#%% Locals
verbose = False

#%% Execution
if __name__ == '__main__':
    folder = dcs.get_root_dir()
    all_files = os.listdir(folder)
    pyfiles = [f for f in all_files if f.endswith('.py') and not f.startswith('__init__')]
    for file in pyfiles:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing ' + file + ':')
        doctest.testfile(os.path.join(folder, file), report=True, verbose=verbose, module_relative=True)
