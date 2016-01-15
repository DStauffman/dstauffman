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

#%% Aliases
b = pentago.PLAYER['black']
w = pentago.PLAYER['white']

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

    def test_large_boards(self):
        board1 = self.board.copy()
        board1[0:3, 0:3] = self.left
        board2 = self.board.copy()
        board2[0:3, 0:3] = self.right
        board3 = self.board.copy()
        board3[0:3, 3:6] = self.left + 3
        board4 = self.board.copy()
        board4[0:3, 3:6] = self.right + 3
        board5 = self.board.copy()
        board5[3:6, 0:3] = self.left + 18
        board6 = self.board.copy()
        board6[3:6, 0:3] = self.right + 18
        board7 = self.board.copy()
        board7[3:6, 3:6] = self.left + 21
        board8 = self.board.copy()
        board8[3:6, 3:6] = self.right + 21
        expected_board = np.vstack((board1.ravel(), board2.ravel(), board3.ravel(), board4.ravel(), \
            board5.ravel(), board6.ravel(), board7.ravel(), board8.ravel())).T
        temp_board = np.ones((1, 8), dtype=int) * np.expand_dims(self.board.ravel(), axis=1)
        quad_dirs = [(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, -1), (4, 1)]
        for (ix, (quad, dir_)) in enumerate(quad_dirs):
            temp = temp_board.copy()
            pentago._rotate_board(temp, quadrant=quad, direction=dir_)
            np.testing.assert_array_equal(temp[:, ix], expected_board[:, ix], 'Quad {}, Dir {}, ix {}'.format(quad, dir_, ix))

    def test_not_inplace(self):
        expected_board = self.board.copy()
        expected_board[0:3, 0:3] = self.left
        temp1 = np.expand_dims(self.board.ravel(), axis=1)
        temp2 = self.board.copy().ravel()
        new_board = pentago._rotate_board(temp1, quadrant=1, direction=-1, inplace=False)
        np.testing.assert_array_equal(np.reshape(new_board, (6,6)), expected_board)
        np.testing.assert_array_equal(temp1, temp2[:, np.newaxis])

    def test_bad_not_inplace(self):
        with self.assertRaises(AssertionError):
            pentago._rotate_board(self.board, quadrant=1, direction=-1, inplace=False)

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
        self.board[0, 0] = w
        self.board[1, 1] = b
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['none'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_white_wins(self):
        self.board[0:5, 0] = w
        self.board[1:4, 1] = b
        self.board[5, 1] = b
        self.win_mask[0:5, 0] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, w)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_black_wins(self):
        self.board[2, 1:6] = b
        self.board[3, 1:4] = w
        self.board[3, 1] = w
        self.win_mask[2, 1:6] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, b)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_white_wins_6(self):
        self.board[0:6, 2] = w
        self.board[1:4, 1] = b
        self.board[5, 1] = b
        self.win_mask[0:6, 2] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, w)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_black_wins_mult(self):
        self.board[1:4, 0] = w
        self.board[5, 0]   = w
        self.board[1:4, 1] = w
        self.board[5, 1]   = w
        self.board[1:3, 2] = w
        self.board[0:6, 3] = b
        self.board[0:6, 5] = b
        self.win_mask[0:6, 3] = True
        self.win_mask[0:6, 5] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, b)
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_draw_no_moves_left(self):
        self.board[0:3, 0:3] = w
        self.board[0:3, 3:6] = b
        self.board[3:6, 0:3] = b
        self.board[3:6, 3:6] = w
        self.board[1, 1] = b
        self.board[4, 4] = b
        self.board[1, 4] = w
        self.board[1, 4] = w
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['draw'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

    def test_draw_simult_wins(self):
        self.board[1, 1:6] = b
        self.board[4, 0:5] = w
        self.win_mask[1, 1:6] = True
        self.win_mask[4, 0:5] = True
        (winner, win_mask) = pentago._check_for_win(self.board)
        self.assertEqual(winner, pentago.PLAYER['draw'])
        np.testing.assert_array_equal(win_mask, self.win_mask)

#%% _find_moves
class Test__find_moves(unittest.TestCase):
    r"""
    Tests the _find_moves function with the following cases:
        TBD
    """
    def setUp(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.white_moves = []
        self.black_moves = []

    def test_no_wins(self):
        (white_moves, black_moves) = pentago._find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_already_won(self):
        self.board[2, 0:5] = w
        with self.assertRaises(ValueError):
            (white_moves, black_moves) = pentago._find_moves(self.board)

    def test_place_to_win(self):
        self.board[2, 0:3] = w
        self.board[1, 3] = w
        self.white_moves.append(pentago.Move(0, 3, 2, -1, 5))
        (white_moves, black_moves) = pentago._find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_place_to_win_x6(self):
        self.board[0, 0] = w
        self.board[1, 1] = w
        self.board[3, 3] = w
        self.board[4, 4] = w
        self.board[5, 5] = w
        self.board[0, 2] = b
        self.board[2, 0] = b
        self.white_moves.append(pentago.Move(2, 2, 2, -1, 5))
        self.white_moves.append(pentago.Move(2, 2, 2,  1, 5))
        self.white_moves.append(pentago.Move(2, 2, 3, -1, 5))
        self.white_moves.append(pentago.Move(2, 2, 3,  1, 5))
        (white_moves, black_moves) = pentago._find_moves(self.board)
        np.testing.assert_array_equal(white_moves, self.white_moves)
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_rotate_to_win(self):
        self.board[3, 1]   = w
        self.board[4, 1]   = w
        self.board[4, 3:6] = w
        for i in range(6):
            for j in range(6):
                if self.board[i, j] == pentago.PLAYER['none']:
                    self.white_moves.append(pentago.Move(i, j, 3, 1, 5))
        self.white_moves.append(pentago.Move(4, 2, 1, -1, 5))
        self.white_moves.append(pentago.Move(4, 2, 1, 1, 5))
        self.white_moves.append(pentago.Move(4, 2, 2, -1, 5))
        self.white_moves.append(pentago.Move(4, 2, 2, 1, 5))
        #self.white_moves.append(pentago.Move(3, 1, 3, 1, 5)) # covered in for loop
        self.white_moves.append(pentago.Move(5, 1, 3, -1, 5))
        #self.white_moves.append(pentago.Move(5, 1, 3, 1, 5)) # covered in for loop
        (white_moves, black_moves) = pentago._find_moves(self.board)
        np.testing.assert_array_equal(white_moves, sorted(self.white_moves))
        np.testing.assert_array_equal(black_moves, self.black_moves)

    def test_rotate_to_win_x6(self):
        pass

    def test_wins_blocked(self):
        pass

    def test_no_valid_moves(self):
        pass

    def test_black_place_to_win(self):
        pass

    def test_black_rotate_to_win(self):
        pass

    def test_draw_move(self):
        pass

    def test_same_win_square(self):
        pass

    def test_same_win_rot(self):
        pass

    def test_same_everything(self):
        pass

    def test_win_over_draw(self):
        pass

    def test_win_rot_blocked_draw(self):
        pass

    def test_other_invalids(self):
        pass

    @unittest.skip('Not yet fully implemented.')
    def test_everything(self):
        self.board = np.reshape(np.hstack((np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]), np.zeros(24, dtype=int))), (6, 6))
        self.white_moves = set((pentago.Move(2, 3, 1, 1, 5), pentago.Move(3, 2, 2, 1, 5), pentago.Move(3, 2, 4, 1, 5), \
            pentago.Move(2, 1, 1, -1, 5), pentago.Move(3, 2, 2, -1, 5), pentago.Move(3, 2, 4, -1, 5)))
        (white_moves, black_moves) = pentago._find_moves(self.position)
        white_set = set((this_move for this_move in white_moves))
        white_set.add(pentago.Move(2, 3, 1, 1, 0))
        np.testing.assert_equal(white_set, self.white_moves)
        self.assertTrue(len(black_moves) == 0)

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
