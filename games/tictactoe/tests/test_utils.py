# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.utils` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import numpy as np
import unittest
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = ttt.get_root_dir()
        self.assertTrue(folder)

#%% calc_cur_move
class Test_calc_cur_move(unittest.TestCase):
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
        self.o        = o
        self.x        = x

    def test_odd_odd(self):
        move = ttt.calc_cur_move(self.odd_num, self.odd_num)
        self.assertEqual(move, o)

    def test_odd_even(self):
        move = ttt.calc_cur_move(self.odd_num, self.even_num)
        self.assertEqual(move, x)

    def test_even_odd(self):
        move = ttt.calc_cur_move(self.even_num, self.odd_num)
        self.assertEqual(move, x)

    def test_even_even(self):
        move = ttt.calc_cur_move(self.even_num, self.even_num)
        self.assertEqual(move, o)

#%% check_for_win
class Test_check_for_win(unittest.TestCase):
    r"""
    Tests the check_for_win function with the following cases:
        TBD
    """
    def setUp(self):
        self.board = ttt.PLAYER['none'] * np.ones((3, 3), dtype=int)
        self.win_mask = np.zeros((3, 3), dtype=bool)

    def test_no_moves(self):
        (winner, win_mask) = ttt.check_for_win(self.board)
        self.assertEqual(winner, ttt.PLAYER['none'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_no_winner(self):
        self.board[0, 0] = x
        self.board[1, 1] = o
        (winner, win_mask) = ttt.check_for_win(self.board)
        self.assertEqual(winner, ttt.PLAYER['none'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_x_wins(self):
        self.board[0:3, 0] = x
        self.board[1:3, 1] = o
        self.win_mask[0:3, 0] = True
        (winner, win_mask) = ttt.check_for_win(self.board)
        self.assertEqual(winner, x)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_o_wins(self):
        self.board[2, 0:3] = o
        self.board[1, 0:2] = x
        self.win_mask[2, 0:3] = True
        (winner, win_mask) = ttt.check_for_win(self.board)
        self.assertEqual(winner, o)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_black_wins_mult(self):
        pass

    def test_draw_no_moves_left(self):
        pass

    def test_draw_simult_wins(self):
        pass

#%% find_moves
@unittest.skip('Rewriting this function to cover all possible moves.')
class Test_find_moves(unittest.TestCase):
    r"""
    Tests the find_moves function with the following cases:
        TBD
    """
    def setUp(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.white_moves = []
        self.black_moves = []

    def test_no_wins(self):
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_already_won(self):
        self.board[1, 0] = x
        self.board[1, 1] = x
        self.board[1, 2] = x
        with self.assertRaises(ValueError):
            (white_moves, black_moves) = ttt.find_moves(self.board)

    def test_o_place_to_win(self):
        self.board[2, 0] = o
        self.board[2, 1] = o
        self.white_moves.append(ttt.Move(2, 2, 3))
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_wins_blocked(self):
        self.board[2, 0] = o
        self.board[2, 1] = o
        self.board[2, 2] = x
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_no_valid_moves(self):
        self.board[0, 0] = x
        self.board[0, 1] = o
        self.board[0, 2] = x
        self.board[1, 0] = x
        self.board[1, 1] = o
        self.board[1, 2] = o
        self.board[2, 0] = o
        self.board[2, 1] = x
        self.board[2, 2] = o
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_x_place_to_win(self):
        self.board[0, 0] = x
        self.board[2, 2] = x
        self.black_moves.append(ttt.Move(1, 1, 3))
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_same_win_square(self):
        self.board[0, 0] = x
        self.board[2, 2] = x
        self.board[1, 0] = o
        self.board[1, 2] = o
        self.white_moves.append(ttt.Move(1, 1, 3))
        self.black_moves.append(ttt.Move(1, 1, 3))
        (white_moves, black_moves) = ttt.find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

#%% make_move
pass

#%% play_ai_game
pass

#%% create_board_from_moves
pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
