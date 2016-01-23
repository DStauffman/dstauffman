# -*- coding: utf-8 -*-
r"""
Pentago board game __main__ function that runs on model execution.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
#.  This model has three methods of running based on command arguments, 'null', 'test' or 'run'.
    The 'run' is the default.  'test' executes the unit tests, and 'null' does nothing.
"""

#%% Imports
import doctest
import sys
import unittest
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication
from dstauffman.games.pentago import PentagoGui

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
    gui = PentagoGui()
    gui.show()
    sys.exit(qapp.exec_())
elif mode == 'test':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(module='dstauffman.games.pentago.tests.run_all_tests', exit=False)
    doctest.testmod(verbose=False)
    # close the qapp
    qapp.closeAllWindows()
    qapp.exit()
elif mode == 'null':
    pass
else:
    raise ValueError('Unexpected mode of "{}".'.format(mode))
