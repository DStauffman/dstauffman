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
    files  = ['bpe', 'classes', 'constants', 'enums', 'linalg', 'photos', 'plotting', 'quat', 'stats', 'units', 'utils']
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing ' + file + '.py:')
        doctest.testfile(os.path.join(folder, file+'.py'), report=True, verbose=verbose, module_relative=True)
