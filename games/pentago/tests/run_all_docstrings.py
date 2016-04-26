# -*- coding: utf-8 -*-
r"""
Test file to execute all the docstrings within the pentago code.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import os
import dstauffman.games.pentago as pentago

#%% Locals
verbose = False

#%% Execution
if __name__ == '__main__':
    folder = pentago.get_root_dir()
    files  = [f for f in os.listdir(folder) if f.endswith('.py') and not f.startswith('__')]
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing "{}":'.format(file))
        doctest.testfile(os.path.join(folder, file), report=True, verbose=verbose, module_relative=True)
