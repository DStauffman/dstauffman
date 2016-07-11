# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the unittest library within the dstauffman code using nose.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import sys
import unittest
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

#%% Options
test_everything = True

#%% Tests
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # get a loader
    loader = unittest.TestLoader()
    # find all the test cases
    if test_everything:
        test_suite = loader.discover('dstauffman.apps')
        test_suite.addTests(loader.discover('dstauffman.archery'))
        test_suite.addTests(loader.discover('dstauffman.games'))
        test_suite.addTests(loader.discover('dstauffman.imageproc'))
        test_suite.addTests(loader.discover('dstauffman.tests'))
    else:
        test_suite = loader.discover('dstauffman.tests')
    # run the tests
    unittest.TextTestRunner(verbosity=1).run(test_suite)
    # close the qapp
    qapp.closeAllWindows()
