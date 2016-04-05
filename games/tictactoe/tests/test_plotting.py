# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.classes` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import unittest
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']
n = ttt.PLAYER['none']

#%% Private Functions
def _make_board():
    r"""Makes a board and returns the figure and axis for use in testing."""
    plt.ioff()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()
    return (fig, ax)

#%% plot_cur_move
class Test_plot_cur_move(unittest.TestCase):
    r"""
    Tests the plot_cur_move function with the following cases:
        Nominal
    """
    def setUp(self):
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)

    def test_x(self):
        ttt.plot_cur_move(self.ax, x)

    def test_o(self):
        ttt.plot_cur_move(self.ax, o)

    def test_none(self):
        ttt.plot_cur_move(self.ax, n)

    def test_bad_value(self):
        with self.assertRaises(ValueError):
            ttt.plot_cur_move(self.ax, 999)

    def tearDown(self):
        plt.close(self.fig)

#%% plot_piece
class Test_plot_piece(unittest.TestCase):
    r"""
    Tests the plot_piece function with the following cases:
        Nominal
    """
    def setUp(self):
        (self.fig, self.ax) = _make_board()

    def test_x(self):
        ttt.plot_piece(self.ax, 1, 1, 0.9, (0, 0, 1), x)

    def test_o(self):
        ttt.plot_piece(self.ax, 1, 1, 0.9, (0, 0, 1), o, thick=False)

    def test_draw(self):
        ttt.plot_piece(self.ax, 1, 1, 0.9, (0, 0, 1), ttt.PLAYER['draw'])

    def test_bad_player(self):
        with self.assertRaises(ValueError):
            ttt.plot_piece(self.ax, 1, 1, 0.9, (0, 0, 1), 999)

    def tearDown(self):
        plt.close(self.fig)

#%% plot_board
class Test_plot_board(unittest.TestCase):
    r"""
    Tests the plot_board function with the following cases:
        Nominal
    """
    def setUp(self):
        (self.fig, self.ax) = _make_board()

    def test_nominal(self):
        board = n * np.ones((3, 3), dtype=int)
        board[0, 0:2] = x
        board[1, 1] = o
        ttt.plot_board(self.ax, board)

    def test_bad_board_position(self):
        board = n * np.ones((3, 3), dtype=int)
        board[1, 1] = 999
        with self.assertRaises(ValueError):
            ttt.plot_board(self.ax, board)

    def tearDown(self):
        plt.close(self.fig)

#%% plot_win
class Test_plot_win(unittest.TestCase):
    r"""
    Tests the plot_win function with the following cases:
        Nominal
    """
    def setUp(self):
        (self.fig, self.ax) = _make_board()

    def test_nominal(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0:2] = True
        board = n * np.ones((3, 3), dtype=int)
        board[0, 0:2] = x
        ttt.plot_win(self.ax, mask, board)

    def tearDown(self):
        plt.close(self.fig)

#%% plot_possible_win
class Test_plot_possible_win(unittest.TestCase):
    r"""
    Tests the plot_possible_win function with the following cases:
        Nominal
    """
    def setUp(self):
        (self.fig, self.ax) = _make_board()

    def test_all_out_wins(self):
        board = np.array([[x, x, n], [n, n, n], [o, o, n]], dtype=int)
        (o_moves, x_moves) = ttt.find_moves(board)
        ttt.plot_possible_win(self.ax, o_moves, x_moves)

    def test_shared_wins(self):
        board = np.array([[x, n, o], [n, n, n], [o, n, x]], dtype=int)
        (o_moves, x_moves) = ttt.find_moves(board)
        ttt.plot_possible_win(self.ax, o_moves, x_moves)

    def tearDown(self):
        plt.close(self.fig)

#%% plot_powers
class Test_plot_powers(unittest.TestCase):
    r"""
    Tests the plot_powers function with the following cases:
        Nominal
    """
    def setUp(self):
        (self.fig, self.ax) = _make_board()

    def test_nominal(self):
        board = np.array([[x, n, n], [n, o, n], [n, o, n]], dtype=int)
        ttt.plot_board(self.ax, board)
        (o_moves, x_moves) = ttt.find_moves(board)
        ttt.plot_powers(self.ax, board, o_moves, x_moves)

    def tearDown(self):
        plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
