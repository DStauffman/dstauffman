# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the dstauffman library using pytest.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import pytest
import sys

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
    pytest.main(['-k', 'tests'])
    # close the qapp
    qapp.closeAllWindows()
