# -*- coding: utf-8 -*-
r"""
Test file for the `games.knight` module of the dcstools code.  It is intented to contain test
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
from dcstools import capture_output
import dcstools.games.knight as knight

#%% n2c & c2n & CHAR_DICT
class Test_num_to_char_and_back(unittest.TestCase):
    r"""
    Tests the n2c and c2n functions, thus ensuring all the mappings are covered.
    """
    def setUp(self):
        self.nums  = knight.Piece.list_of_values()
        self.chars = knight.CHAR_DICT.keys()

    def test_c2n(self):
        for this_char in self.chars:
            this_num = knight.c2n(this_char)
            self.assertTrue(this_num in self.nums)

    def test_n2c(self):
        for this_num in self.nums:
            this_char = knight.n2c(this_num)
            self.assertTrue(this_char in self.chars)

    def test_bad_char(self):
        with self.assertRaises(KeyError):
            knight.c2n('Z')

    def test_bad_num(self):
        with self.assertRaises(ValueError):
            knight.n2c(10000)

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
        self.assertEqual(output, 'S S S S \nS S S S \nS S S S \nS S S S')

    def test_rect_board(self):
        with capture_output() as (out, _):
            knight.print_board(self.board2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '. . . . . \n. . . . . \n. . . . .')

    def test_all_board_piece_types(self):
        with capture_output() as (out, _):
            knight.print_board(self.board3)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '. S E \nK W R \nB T L \nx . .')

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

#%% char_board_to_nums
pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
