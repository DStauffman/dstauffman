# -*- coding: utf-8 -*-
r"""
Test file for the `games.knight` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in September 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import numpy as np
import unittest
from dstauffman import capture_output
import dstauffman.games.knight as knight

#%% print_board
class Test_print_board(unittest.TestCase):
    r"""
    Tests the print_board function with the following cases:
        Square Board
        Rectangular board
        All piece types on the board
    """
    def setUp(self):
        self.board1 = np.ones((4,4), dtype=int)
        self.board2 = np.zeros((3,5), dtype=int)
        self.board3 = np.arange(12).reshape((4,3))
        self.board3[self.board3 > knight.Piece.max_value] = 0

    def test_square_board(self):
        with capture_output() as (out, _):
            knight.print_board(self.board1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'S S S S\nS S S S\nS S S S\nS S S S')

    def test_rect_board(self):
        with capture_output() as (out, _):
            knight.print_board(self.board2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '. . . . .\n. . . . .\n. . . . .')

    def test_all_board_piece_types(self):
        with capture_output() as (out, _):
            knight.print_board(self.board3)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '. S E\nK W R\nB T L\nx . .')

#%% CHAR_DICT & NUM_DICT & Piece & char_board_to_nums
class Test_char_board_to_nums(unittest.TestCase):
    r"""
    Tests the char_board_to_nums function and CHAR_DICT and NUM_DICT, thus ensuring all the mappings
    are covered.  Uses cases:
        All char dict entries
        All num dict entries
        Bad char key
        Bad num key
        Nominal char board to nums with all values
        Extra empty line char board to nums
    """
    def setUp(self):
        self.nums        = knight.NUM_DICT.keys()
        self.chars       = knight.CHAR_DICT.keys()
        self.enums       = knight.Piece.list_of_values()
        self.char_board  = '. S E K W\nR B T L x'
        self.board       = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.char_board2 = '\n' + self.char_board + '\n'

    def test_char_dict(self):
        for this_char in self.chars:
            this_num = knight.CHAR_DICT[this_char]
            self.assertTrue(this_num in self.nums)

    def test_num_dict(self):
        for this_num in self.nums:
            this_char = knight.NUM_DICT[this_num]
            self.assertTrue(this_char in self.chars)

    def test_bad_char(self):
        with self.assertRaises(KeyError):
            knight.CHAR_DICT['Z']

    def test_bad_num(self):
        with self.assertRaises(KeyError):
            knight.NUM_DICT[10000]

    def test_nominal(self):
        board = knight.char_board_to_nums(self.char_board)
        np.testing.assert_array_equal(board, self.board)

    def test_extra_lines(self):
        board = knight.char_board_to_nums(self.char_board2)
        np.testing.assert_array_equal(board, self.board)

#%% board_to_costs
class Test_board_to_costs(unittest.TestCase):
    r"""
    Tests the board_to_costs function with the following cases:
        All possible costs
        Bad costs (x2)
    """
    def setUp(self):
        char_board = '. S E . W\nR B T L .'
        self.board = knight.char_board_to_nums(char_board)
        self.costs = np.array([[1, 0, 1, 1, 2], [knight.LARGE_INT, knight.LARGE_INT, 1, 5, 1]])

    def test_nominal(self):
        costs = knight.board_to_costs(self.board)
        np.testing.assert_array_equal(costs, self.costs)

    def test_bad_board1(self):
        self.board[0, 0] = knight.Piece.current
        with self.assertRaises(ValueError):
            knight.board_to_costs(self.board)

    def test_bad_board2(self):
        self.board[0, 0] = knight.Piece.visited
        with self.assertRaises(ValueError):
            knight.board_to_costs(self.board)

#%% get_current_position
class Test_get_current_position(unittest.TestCase):
    r"""
    Tests the get_current_position function with the following cases:
        Nominal
        No current piece
        Multiple current pieces
    """
    def setUp(self):
        self.board = knight.Piece.null * np.ones((2, 5), dtype=int)
        self.x = 0
        self.y = 2
        self.board[self.x, self.y] = knight.Piece.current

    def test_nominal(self):
        (x, y) = knight.get_current_position(self.board)
        self.assertEqual(x, self.x)
        self.assertEqual(y, self.y)

    def test_no_current(self):
        self.board[self.x, self.y] = knight.Piece.start
        with self.assertRaises(AssertionError):
            with capture_output() as (out, _):
                knight.get_current_position(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'. . S . .\n. . . . .')

    def test_multiple_currents(self):
        self.board[self.x + 1, self.y + 1] = knight.Piece.current
        with self.assertRaises(AssertionError):
            with capture_output() as (out, _):
                knight.get_current_position(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'. . K . .\n. . . K .')

#%% get_new_position
class Test_get_new_position(unittest.TestCase):
    r"""
    Tests the get_new_position function with the following cases:
        All valid moves
        Not yet done moves
        Invalid moves
    """
    #  Move -1       Move +1        Move -2      Move +2       Move -3       Move +3       Move -4       Move +4
    # . E x . .  |  . . x E .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .
    # . . x . .  |  . . x . .  |  . . . . E  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  E . . . .
    # . . S . .  |  . . S . .  |  . . S x x  |  . . S x x  |  . . S . .  |  . . S . .  |  x x S . .  |  x x S . .
    # . . . . .  |  . . . . .  |  . . . . .  |  . . . . E  |  . . x . .  |  . . x . .  |  E . . . .  |  . . . . .
    # . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . x E .  |  . E x . .  |  . . . . .  |  . . . . .
    def setUp(self):
        self.x = 2
        self.y = 2
        self.board = np.zeros((5,5), dtype=int)
        self.board[self.x, self.y] = knight.Piece.start
        self.valid_moves = [-4, -3, -2, -1, 1, 2, 3, 4]
        self.future_moves = [-8, -7, -6, -5, 5, 6, 7, 8]
        self.bad_moves = [0, 9, -9, 100]
        self.results = [(3, 0), (4, 3), (1, 4), (0, 1), (0, 3), (3, 4), (4, 1), (1, 0)]

    def test_valid_moves(self):
        for (this_move, this_result) in zip(self.valid_moves, self.results):
            (pos1, pos2, pos3) = knight.get_new_position(self.x, self.y, this_move)
            self.assertEqual(pos3, this_result)
            # TODO: assert something about pos1 and pos2?

    def test_future_moves(self):
        for (this_move, this_result) in zip(self.future_moves, self.results):
            with self.assertRaises(ValueError):
                knight.get_new_position(self.x, self.y, this_move)
            #(pos1, pos2, pos3) = knight.get_new_position(self.x, self.y, this_move)
            #self.assertEqual(pos3, this_result)

    def test_bad_moves(self):
        for this_move in self.bad_moves:
            with self.assertRaises(ValueError):
                knight.get_new_position(self.x, self.y, this_move)

#%% check_board_boundaries
class Test_check_board_boundaries(unittest.TestCase):
    r"""
    Tests the check_board_boundaries function with the following cases:
        Good values
        Bad X values
        Bad Y values
        Bad X and Y values
    """
    def setUp(self):
        self.xmax = 4
        self.ymax = 3
        self.x = np.arange(self.xmax + 1)
        self.y = np.arange(self.ymax + 1)
        self.bad_x = [-1, self.xmax + 1]
        self.bad_y = [-1, self.ymax + 1]

    def test_good_values(self):
        for this_x in self.x:
            for this_y in self.y:
                is_valid = knight.check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertTrue(is_valid)

    def test_bad_values1(self):
        for this_x in self.bad_x:
            for this_y in self.y:
                is_valid = knight.check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

    def test_bad_values2(self):
        for this_x in self.x:
            for this_y in self.bad_y:
                is_valid = knight.check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

    def test_bad_values3(self):
        for this_x in self.bad_x:
            for this_y in self.bad_y:
                is_valid = knight.check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

#%% classify_move
pass # TODO: write this

#%% update_board
pass # TODO: write this

#%% undo_move
pass # TODO: write this

#%% get_move_inverse
pass # TODO: write this

#%% check_valid_sequence
pass # TODO: write this

#%% print_sequence
pass # TODO: write this

#%% solve_puzzle
pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
