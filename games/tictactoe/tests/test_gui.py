# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.gui` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import copy
import sys
import unittest
try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtTest import QTest
except ImportError:
    from PyQt4 import QtCore
    from PyQt4.QtGui import QApplication
    from PyQt4.QtTest import QTest
from dstauffman import compare_two_classes
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']

#%% Flags
skip = True

#%% TicTacToeGui
@unittest.skipIf(skip, 'Skipping GUI tests.')
class Test_TicTacToeGui(unittest.TestCase):
    r"""
    Tests the TicTacToeGui with the following cases:
        Press Reset button
        Press None button
        Press All button
        TODO: put in many more
    """
    def _default(self):
        # assert default starting conditions
        self.assertTrue(compare_two_classes(self.gui.state, self.state, suppress_output=True))

    def _reset(self):
        # press the reset button
        QTest.mouseClick(self.gui.btn_reset, QtCore.Qt.LeftButton) # TODO: need equivalent option
        self.assertTrue(compare_two_classes(self.gui.state, self.state, suppress_output=True))

    @classmethod
    def setUpClass(cls):
        cls.gui = ttt.TicTacToeGui()
        cls.state = copy.deepcopy(cls.gui.state)

    def test_undo_button(self):
        self._default()
        # press the reset button
        QTest.mouseClick(self.gui.btn_undo, QtCore.Qt.LeftButton)
        # TODO: assert something

#%% Unit test execution
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(exit=False)
    # close the qapp
    qapp.closeAllWindows()
