# -*- coding: utf-8 -*-
r"""
Test file to execute all the docstrings within the tictactoe code.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import os
import dstauffman.games.tictactoe as ttt

#%% Locals
verbose = False

#%% Execution
if __name__ == '__main__':
    folder = ttt.get_root_dir()
    files  = ['classes', 'constants', 'gui', 'plotting', 'utils']
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing ' + file + '.py:')
        doctest.testfile(os.path.join(folder, file+'.py'), report=True, verbose=verbose, module_relative=True)
