# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the unittest library within the dstauffman code using nose.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import nose
import dstauffman as dcs
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # TODO: temporary

#%% Tests
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    nose.run(dcs)
    # close the qapp
    qapp.closeAllWindows()
