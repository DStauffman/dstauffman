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

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

#%% Tests
if __name__ == '__main__':
    # turn interactive plotting off
    plt.ioff()
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # get a loader
    loader = unittest.TestLoader()
    # find all the test cases
    test_suite = loader.discover('dstauffman.tests')
    # run the tests
    unittest.TextTestRunner(verbosity=1).run(test_suite)
    # close the qapp
    qapp.closeAllWindows()
