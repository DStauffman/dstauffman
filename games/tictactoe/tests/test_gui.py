# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.gui` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
# normal imports
import copy
import numpy as np
import sys
import unittest
# Qt imports
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtTest import QTest
# model imports
from dstauffman import Counter
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']
n = ttt.PLAYER['none']

#%% Flags
skip = True

#%% TicTacToeGui
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
        self.assertEqual(self.gui.state.cur_game, self.state.cur_game)
        self.assertEqual(self.gui.state.cur_move, self.state.cur_move)
        np.testing.assert_array_equal(self.gui.state.board, self.state.board)

    def _reset(self):
        # press the reset button
        self.gui.state.cur_game = Counter(0)
        self.gui.state.cur_move = Counter(0)
        self.gui.state.board    = n * np.ones((3, 3), dtype=int)
        self._default()

    def _place_piece(self, row, col):
        board_width = self.gui.board_canvas.size().width()
        board_height = self.gui.board_canvas.size().height()
        pos = QtCore.QPoint((2*col + 1) * (board_width // 6), (2*row + 1) * (board_height // 6))
        QTest.mouseClick(self.gui.board_canvas, QtCore.Qt.LeftButton, pos=pos)

    def test_sequence(self):
        # set Options
        ttt.Options.o_is_computer = False
        ttt.Options.x_is_computer = False
        ttt.Options.load_previous_game = 'No'
        # instantiate GUI
        self.gui = ttt.TicTacToeGui()
        # copy the state for reference
        self.state = copy.deepcopy(self.gui.state)
        # establish defaults
        self._default()
        new_board = self.state.board.copy()
        # place piece 1
        self._place_piece(0, 0)
        new_board[0, 0] = o
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 1)
        self.assertEqual(self.gui.state.cur_game, 0)
        # place piece 2
        self._place_piece(1, 0)
        new_board[1, 0] = x
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 2)
        # undo second move
        QTest.mouseClick(self.gui.btn_undo, QtCore.Qt.LeftButton)
        new_board[1, 0] = n
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 1)
        # undo first move
        QTest.mouseClick(self.gui.btn_undo, QtCore.Qt.LeftButton)
        new_board[0, 0] = n
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 0)
        # redo first move
        QTest.mouseClick(self.gui.btn_redo, QtCore.Qt.LeftButton)
        new_board[0, 0] = o
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 1)
        # redo second move
        QTest.mouseClick(self.gui.btn_redo, QtCore.Qt.LeftButton)
        new_board[1, 0] = x
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 2)
        # complete game
        self._place_piece(0, 1)
        new_board[0, 1] = o
        self._place_piece(1, 1)
        new_board[1, 1] = x
        self._place_piece(0, 2)
        new_board[0, 2] = o
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 5)
        # make extra click
        self._place_piece(2, 2)
        np.testing.assert_array_equal(self.gui.state.board, new_board)
        self.assertEqual(self.gui.state.cur_move, 5)
        # start new game
        QTest.mouseClick(self.gui.btn_new, QtCore.Qt.LeftButton)
        new_board = self.state.board.copy()
        self.assertEqual(self.gui.state.cur_move, 0)
        self.assertEqual(self.gui.state.cur_game, 1)
        # place one piece
        self._place_piece(1, 2)
        new_board[1, 2] = x
        self.assertEqual(self.gui.state.cur_move, 1)
        # click on existing piece
        self._place_piece(1, 2)
        self.assertEqual(self.gui.state.cur_move, 1)
        # click off the board
        self._place_piece(-1, 2)
        self.assertEqual(self.gui.state.cur_move, 1)

    def test_ai_game(self):
        # set Options
        ttt.Options.o_is_computer = True
        ttt.Options.x_is_computer = True
        ttt.Options.load_previous_game = 'No'
        ttt.Options.plot_best_moves = True
        ttt.Options.plot_move_power = True
        # instantiate GUI
        self.gui = ttt.TicTacToeGui()
        # confirm that the game completed
        self.assertEqual(self.gui.state.cur_move, 9)

    def test_load_game(self):
        ttt.Options.o_is_computer = True
        ttt.Options.x_is_computer = True
        ttt.Options.load_previous_game = 'Yes'
        self.gui = ttt.TicTacToeGui()

    @unittest.skip('Don''t know how to implement this one yet.')
    def test_ask_game_load(self):
        ttt.Options.o_is_computer = True
        ttt.Options.x_is_computer = True
        ttt.Options.load_previous_game = 'Ask'
        self.gui = ttt.TicTacToeGui()
        # TODO: move focus to new window
        self.gui.load_widget.setFocus()
        buttons = self.gui.load_widget.findChildren(QPushButton)
        print(buttons)
        # click on Yes button
        QTest.mouseClick(self.load_widget, QtCore.Qt.LeftButton)

        self._place_piece(1, 1)
        self.assertTrue(self.gui.state.board[1, 1] != n)

    def test_ask_bad_option(self):
        ttt.Options.load_previous_game = 'Bad Option'
        with self.assertRaises(ValueError):
            self.gui = ttt.TicTacToeGui()

    def tearDown(self):
        QApplication.instance().closeAllWindows()

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
