# -*- coding: utf-8 -*-
r"""
Tic Tac Toe board game __main__ function that runs on model execution.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
#.  This model has three methods of running based on command arguments, 'null', 'test' or 'run'.
    The 'run' is the default.  'test' executes the unit tests, and 'null' does nothing.
"""

#%% Imports
import doctest
import os
import sys
import unittest
from PyQt5.QtWidgets import QApplication
import dstauffman.games.tictactoe as ttt

#%% Argument parsing
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = 'run'

#%% Execution
if mode == 'run':
    # Runs the GUI application
    qapp = QApplication(sys.argv)
    # instatiates the GUI
    gui = ttt.TicTacToeGui()
    gui.show()
    sys.exit(qapp.exec_())
elif mode == 'test':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # find the test cases
    test_suite = unittest.TestLoader().discover('dstauffman.games.tictactoe.tests')
    # run the tests
    unittest.TextTestRunner(verbosity=1).run(test_suite)
    # run the docstrings
    verbose = False
    folder = ttt.get_root_dir()
    files = [f for f in os.listdir(folder) if f.endswith('.py') and not f.startswith('__')]
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing "{}":'.format(file))
        doctest.testfile(os.path.join(folder, file), report=True, verbose=verbose, module_relative=True)
    # close the qapp
    qapp.closeAllWindows()
    qapp.exit()
elif mode == 'null':
    pass
else:
    raise ValueError('Unexpected mode of "{}".'.format(mode))
