# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the unittest library within the tictactoe code using nose.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import nose
import sys
import dstauffman.games.tictactoe as ttt
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    nose.run(ttt)
    # close the qapp
    qapp.closeAllWindows()
