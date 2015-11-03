# -*- coding: utf-8 -*-
r"""
Test file for the `games.fiver` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in November 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import numpy as np
import unittest
from dstauffman import capture_output
import dstauffman.games.fiver as fiver

#%% _pad_piece
class _pad_piece(unittest.TestCase):
    r"""
    Tests the _pad_piece function with the following cases:
        Nominal
    """
    def setUp(self):
        self.piece = np.array([[1, 1, 1, 1], [0, 0, 0, 1]], dtype=int)
        self.max_size = 5
        self.new_piece = np.array([[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int)

    def test_nominal(self):
        new_piece = fiver._pad_piece(self.piece, self.max_size)
        np.testing.assert_array_equal(new_piece, self.new_piece)

#%% _shift_piece
class _shift_piece(unittest.TestCase):
    r"""
    Tests the _shift_piece function with the following cases:
        Nominal
    """
    def setUp(self):
        self.x = np.zeros((5,5), dtype=int)
        self.x[1, :] = 1
        self.y = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int)

    def test_nominal(self):
        y = fiver._shift_piece(self.x)
        np.testing.assert_array_equal(y, self.y)

#%% _rotate_piece
class Test__rotate_piece(unittest.TestCase):
    r"""
    Tests the _rotate_piece function with the following cases:
        Nominal
    """
    def setUp(self):
        self.x = np.arange(25).reshape((5, 5))
        self.y = np.array([[4, 9, 14, 19, 24], [3, 8, 13, 18, 23], [2, 7, 12, 17, 22], \
            [1, 6, 11, 16, 21], [0, 5, 10, 15, 20]], dtype=int)

    def test_nominal(self):
        y = fiver._rotate_piece(self.x)
        np.testing.assert_array_equal(y, self.y)

#%% _flip_piece
class Test__flip_piece(unittest.TestCase):
    r"""
    Tests the _flip_piece function with the following cases:
        Nominal
    """
    def setUp(self):
        self.x = np.arange(25).reshape((5, 5))
        self.y = np.array([[20, 21, 22, 23, 24], [15, 16, 17, 18, 19], [10, 11, 12, 13, 14], \
            [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], dtype=int)

    def test_nominal(self):
        y = fiver._flip_piece(self.x)
        np.testing.assert_array_equal(y, self.y)

#%% _get_unique_pieces
class Test__get_unique_pieces(unittest.TestCase):
    r"""
    Tests the _get_unique_pieces function with the following cases:
        Nominal
    """
    def setUp(self):
        self.pieces = np.zeros((3, 5, 5), dtype=int)
        self.pieces[0, :, 0] = 1
        self.pieces[1, :, 1] = 1
        self.pieces[2, :, 0] = 1
        self.ix_unique = [0, 1]

    def test_nominal(self):
        ix_unique = fiver._get_unique_pieces(self.pieces)
        np.testing.assert_array_equal(ix_unique, self.ix_unique)

#%% _display_progress
class Test__display_progress(unittest.TestCase):
    r"""
    Tests the _display_progress function with the following cases:
        Nominal
    """
    def setUp(self):
        self.ix = np.array([1, 0, 4, 0])
        self.nums = np.array([2, 4, 8, 16])
        self.ratio = (512+64) / 1024

    def test_nominal(self):
        with capture_output() as (out, _):
            ratio = fiver._display_progress(self.ix, self.nums)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Progess: 56.2%')
        self.assertEqual(ratio, self.ratio)

#%% _blobbing
class Test__blobbing(unittest.TestCase):
    r"""
    Tests the _blobbing function with the following cases:
        TBD
    """
    def setUp(self):
        # board 1, whole row
        self.board1       = np.zeros((3, 5), dtype=bool)
        self.board1[1, :] = 1
        # board 2, plus shape, 5 openings
        self.board2         = np.zeros((4, 5), dtype=bool)
        self.board2[1, 2]   = 1
        self.board2[2, 1:4] = 1
        self.board2[3, 2]   = 1
        # board 3, variation of 2 with only 4 openings
        self.board3 = self.board2.copy()
        self.board3[2, 1] = 0
        # board 4, variation of 2 with 7 openings
        self.board4 = self.board2.copy()
        self.board4[3, 1] = 1
        self.board4[3, 3] = 1
        # board 5, multiple blobs, all valid
        self.board5 = np.vstack((self.board1, self.board2))
        # board 6, multiple blobs, some invalid
        self.board6 = np.vstack((self.board1, self.board4))

    def test_1(self):
        out = fiver._blobbing(self.board1)
        self.assertTrue(out)

    def test_2(self):
        out = fiver._blobbing(self.board2)
        self.assertTrue(out)

    def test_3(self):
        out = fiver._blobbing(self.board3)
        self.assertFalse(out)

    def test_4(self):
        out = fiver._blobbing(self.board4)
        self.assertFalse(out)

    def test_5(self):
        out = fiver._blobbing(self.board5)
        self.assertTrue(out)

    def test_6(self):
        out = fiver._blobbing(self.board6)
        self.assertFalse(out)

#%% _save_solution
class Test__save_solution(unittest.TestCase):
    r"""
    Tests the _save_solution function with the following cases:
        TBD
    """
    def setUp(self):
        self.this_board1 = np.arange(25).reshape((5, 5))
        self.this_board2 = np.ones((5,5), dtype=int)
        self.this_board3 = np.rot90(self.this_board1, 2)
        self.this_board4 = np.fliplr(self.this_board1)
        self.solutions = [self.this_board1]

    def test_is_new(self):
        solutions = self.solutions[:]
        self.assertEqual(len(solutions), 1)
        with capture_output() as (out, _):
            fiver._save_solution(solutions, self.this_board2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(len(solutions), 2)
        self.assertEqual(output, 'Solution 2 found!')

    def test_not_new1(self):
        solutions = self.solutions[:]
        self.assertEqual(len(solutions), 1)
        fiver._save_solution(solutions, self.this_board3)
        self.assertEqual(len(solutions), 1)

    def test_not_new2(self):
        solutions = self.solutions[:]
        self.assertEqual(len(solutions), 1)
        fiver._save_solution(solutions, self.this_board4)
        self.assertEqual(len(solutions), 1)

#%% make_all_pieces
class Test_make_all_pieces(unittest.TestCase):
    r"""
    Tests the make_all_pieces function with the following cases:
        Nominal
    """
    def setUp(self):
        self.pieces0 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int)

    def test_nominal(self):
        pieces = fiver.make_all_pieces()
        np.testing.assert_array_equal(pieces[0], self.pieces0)
        # TODO: assert better tests?

#%% make_all_permutations
class Test_make_all_permutations(unittest.TestCase):
    r"""
    Tests the make_all_permutations function with the following cases:
        Nominal
    """
    def setUp(self):
        self.pieces = fiver.make_all_pieces()

    def test_nominal(self):
        all_pieces = fiver.make_all_permutations(self.pieces)
        # TODO: assert something

#%% is_valid
class Test_is_valid(unittest.TestCase):
    r"""
    Tests the is_valid function with the following cases:
        TBD
    """
    def setUp(self):
        # board 1, internal 3x3
        self.board1 = np.ones((5, 5), dtype=int)
        self.board1[1:-1, 1:-1] = 0
        # board 2, missing corner of 3x3
        self.board2 = self.board1.copy()
        self.board2[3,3] = 1
        # piece one, 3x1 in second row
        self.piece1 = np.zeros((5,5), dtype=int)
        self.piece1[1, 1:4] = 2
        # piece two, extra boundary piece that will be invalid
        self.piece2 = self.piece1.copy()
        self.piece2[0, 1] = 2
        # piece three, combinations of 1 & 2
        self.piece3 = np.empty((2, 5, 5), dtype=int)
        self.piece3[0] = self.piece1.copy()
        self.piece3[1] = self.piece2.copy()

    def test_single1(self):
        out = fiver.is_valid(self.board1, self.piece1, use_blobbing=False)
        self.assertTrue(out)

    def test_single2(self):
        out = fiver.is_valid(self.board1, self.piece1, use_blobbing=True)
        self.assertFalse(out)

    def test_single3(self):
        out = fiver.is_valid(self.board2, self.piece1, use_blobbing=True)
        self.assertTrue(out)

    def test_invalid1(self):
        out = fiver.is_valid(self.board1, self.piece2, use_blobbing=False)
        self.assertFalse(out)

    def test_invalid2(self):
        out = fiver.is_valid(self.board1, self.piece2, use_blobbing=True)
        self.assertFalse(out)

    def test_3d1(self):
        out = fiver.is_valid(self.board1, self.piece3, use_blobbing=False)
        np.testing.assert_array_equal(out, np.array([True, False], dtype=bool))

    def test_3d2(self):
        out = fiver.is_valid(self.board1, self.piece3, use_blobbing=True)
        np.testing.assert_array_equal(out, np.array([False, False], dtype=bool))

    def test_3d3(self):
        out = fiver.is_valid(self.board2, self.piece3, use_blobbing=True)
        np.testing.assert_array_equal(out, np.array([True, False], dtype=bool))

#%% find_all_valid_locations
class Test_find_all_valid_locations(unittest.TestCase):
    r"""
    Tests the find_all_valid_locations function with the following cases:
        TBD
    """
    pass #TODO: write this

#%% solve_puzzle
class Test_solve_puzzle(unittest.TestCase):
    r"""
    Tests the solve_puzzle function with the following cases:
        TBD
    """
    pass #TODO: write this

#%% plot_board
class Test_plot_board(unittest.TestCase):
    r"""
    Tests the plot_board function with the following cases:
        TBD
    """
    pass #TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
