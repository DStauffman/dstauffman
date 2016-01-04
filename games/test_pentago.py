# -*- coding: utf-8 -*-
r"""
Test file for the `games.pentago` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

#%% Imports
import numpy as np
import unittest
import dstauffman.games.pentago as pentago

#%% Setup
pentago.LOGGING = False

#%% Move
pass

#%% GameStats
pass

#%% PentagoGui
pass

#%% _mouse_click_callback
pass

#%% _calc_cur_move
class Test__calc_cur_move(unittest.TestCase):
    r"""
    Tests the _board_to_costs function with the following cases:
        Odd game, odd move
        Odd game, even move
        Even game, odd move
        Even game, even move
    """
    def setUp(self):
        self.odd_num  = 3
        self.even_num = 4
        self.white = 1
        self.black = -1

    def test_odd_odd(self):
        move = pentago._calc_cur_move(self.odd_num, self.odd_num)
        self.assertEqual(move, self.white)

    def test_odd_even(self):
        move = pentago._calc_cur_move(self.odd_num, self.even_num)
        self.assertEqual(move, self.black)

    def test_even_odd(self):
        move = pentago._calc_cur_move(self.even_num, self.odd_num)
        self.assertEqual(move, self.black)

    def test_even_even(self):
        move = pentago._calc_cur_move(self.even_num, self.even_num)
        self.assertEqual(move, self.white)

#%% _execute_move
pass

#%% _rotate_board
class Test__rotate_board(unittest.TestCase):
    r"""
    Tests the _rotate_board function with the following cases:
        TBD
    """
    def setUp(self):
        self.board = np.reshape(np.arange(36), (6, 6))
        self.left  = np.array([[ 2,  8, 14], [ 1,  7, 13], [ 0,  6, 12]])
        self.right = np.array([[12,  6,  0], [13,  7,  1], [14,  8,  2]])
        #[[ 0  1  2  3  4  5]
        # [ 6  7  8  9 10 11]
        # [12 13 14 15 16 17]
        # [18 19 20 21 22 23]
        # [24 25 26 27 28 29]
        # [30 31 32 33 34 35]]

    def test_q1_dl(self):
        expected_board = self.board.copy()
        expected_board[0:3, 0:3] = self.left
        pentago._rotate_board(self.board, quadrant=1, direction=-1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q1_dr(self):
        expected_board = self.board.copy()
        expected_board[0:3, 0:3] = self.right
        pentago._rotate_board(self.board, quadrant=1, direction=1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q2_dl(self):
        expected_board = self.board.copy()
        expected_board[0:3, 3:6] = self.left + 3
        pentago._rotate_board(self.board, quadrant=2, direction=-1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q2_dr(self):
        expected_board = self.board.copy()
        expected_board[0:3, 3:6] = self.right + 3
        pentago._rotate_board(self.board, quadrant=2, direction=1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q3_dl(self):
        expected_board = self.board.copy()
        expected_board[3:6, 0:3] = self.left + 18
        pentago._rotate_board(self.board, quadrant=3, direction=-1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q3_dr(self):
        expected_board = self.board.copy()
        expected_board[3:6, 0:3] = self.right + 18
        pentago._rotate_board(self.board, quadrant=3, direction=1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q4_dl(self):
        expected_board = self.board.copy()
        expected_board[3:6, 3:6] = self.left + 21
        pentago._rotate_board(self.board, quadrant=4, direction=-1)
        np.testing.assert_array_equal(self.board, expected_board)

    def test_q4_dr(self):
        expected_board = self.board.copy()
        expected_board[3:6, 3:6] = self.right + 21
        pentago._rotate_board(self.board, quadrant=4, direction=1)
        np.testing.assert_array_equal(self.board, expected_board)

#%% _check_for_win
class Test__check_for_win(unittest.TestCase):
    r"""
    Tests the _check_for_win function with the following cases:
        TBD
    """
    def setUp(self):
        self.board = pentago.PLAYER['none'] * np.ones((6, 6), dtype=int)
        self.win_mask = np.zeros((6, 6), dtype=bool)

    def test_no_moves(self):
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['none'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_no_winner(self):
        self.board[0, 0] = pentago.PLAYER['white']
        self.board[1, 1] = pentago.PLAYER['black']
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['none'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_white_wins(self):
        self.board[0:5, 0] = pentago.PLAYER['white']
        self.board[1:4, 1] = pentago.PLAYER['black']
        self.board[5, 1] = pentago.PLAYER['black']
        self.win_mask[0:5, 0] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['white'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_black_wins(self):
        self.board[2, 1:6] = pentago.PLAYER['black']
        self.board[3, 1:4] = pentago.PLAYER['white']
        self.board[3, 1] = pentago.PLAYER['white']
        self.win_mask[2, 1:6] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['black'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_white_wins_6(self):
        self.board[0:6, 2] = pentago.PLAYER['white']
        self.board[1:4, 1] = pentago.PLAYER['black']
        self.board[5, 1] = pentago.PLAYER['black']
        self.win_mask[0:6, 2] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['white'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_black_wins_mult(self):
        self.board[1:4, 0] = pentago.PLAYER['white']
        self.board[5, 0]   = pentago.PLAYER['white']
        self.board[1:4, 1] = pentago.PLAYER['white']
        self.board[5, 1]   = pentago.PLAYER['white']
        self.board[1:3, 2] = pentago.PLAYER['white']
        self.board[0:6, 3] = pentago.PLAYER['black']
        self.board[0:6, 5] = pentago.PLAYER['black']
        self.win_mask[0:6, 3] = True
        self.win_mask[0:6, 5] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['black'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_draw_no_moves_left(self):
        self.board[0:3, 0:3] = pentago.PLAYER['white']
        self.board[0:3, 3:6] = pentago.PLAYER['black']
        self.board[3:6, 0:3] = pentago.PLAYER['black']
        self.board[3:6, 3:6] = pentago.PLAYER['white']
        self.board[1, 1] = pentago.PLAYER['black']
        self.board[4, 4] = pentago.PLAYER['black']
        self.board[1, 4] = pentago.PLAYER['white']
        self.board[1, 4] = pentago.PLAYER['white']
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['draw'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_draw_simult_wins(self):
        self.board[1, 1:6] = pentago.PLAYER['black']
        self.board[4, 0:5] = pentago.PLAYER['white']
        self.win_mask[1, 1:6] = True
        self.win_mask[4, 0:5] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['draw'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

#%% _update_game_stats
pass

#%% _display_controls
pass

#%% _plot_cur_move
pass

#%% _plot_piece
pass

#%% _plot_board
pass

#%% _plot_win
pass

#%% wrapper
pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
