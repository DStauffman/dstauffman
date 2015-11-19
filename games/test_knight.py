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

#%% _board_to_costs
class Test__board_to_costs(unittest.TestCase):
    r"""
    Tests the _board_to_costs function with the following cases:
        All possible costs
        Bad costs (x2)
    """
    def setUp(self):
        char_board = '. S E . W\nR B T L .'
        self.board = knight.char_board_to_nums(char_board)
        self.costs = np.array([[1, 0, 1, 1, 2], [knight.LARGE_INT, knight.LARGE_INT, 1, 5, 1]])

    def test_nominal(self):
        costs = knight._board_to_costs(self.board)
        np.testing.assert_array_equal(costs, self.costs)

    def test_bad_board1(self):
        self.board[0, 0] = knight.Piece.current
        with self.assertRaises(ValueError):
            knight._board_to_costs(self.board)

    def test_bad_board2(self):
        self.board[0, 0] = knight.Piece.visited
        with self.assertRaises(ValueError):
            knight._board_to_costs(self.board)

#%% _get_transports
class _get_transports(unittest.TestCase):
    r"""
    Tests the _get_transports function with the following cases:
        Valid transports
        No transports
        Invalid transports
    """
    def setUp(self):
        self.board = knight.Piece.null * np.ones((3, 3), dtype=int)
        self.transports = [(0, 1), (2, 2)]

    def test_valid_transports(self):
        self.board[0, 1] = knight.Piece.transport
        self.board[2, 2] = knight.Piece.transport
        transports = knight._get_transports(self.board)
        for (ix, this_transport) in enumerate(transports):
            self.assertEqual(this_transport, self.transports[ix])

    def test_no_transports(self):
        transports = knight._get_transports(self.board)
        self.assertTrue(transports is None)

    def test_invalid_transports(self):
        self.board[0, 1] = knight.Piece.transport
        with self.assertRaises(AssertionError):
            knight._get_transports(self.board)

#%% _get_current_position
class Test__get_current_position(unittest.TestCase):
    r"""
    Tests the _get_current_position function with the following cases:
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
        (x, y) = knight._get_current_position(self.board)
        self.assertEqual(x, self.x)
        self.assertEqual(y, self.y)

    def test_no_current(self):
        self.board[self.x, self.y] = knight.Piece.start
        with self.assertRaises(AssertionError):
            with capture_output() as (out, _):
                knight._get_current_position(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'. . S . .\n. . . . .')

    def test_multiple_currents(self):
        self.board[self.x + 1, self.y + 1] = knight.Piece.current
        with self.assertRaises(AssertionError):
            with capture_output() as (out, _):
                knight._get_current_position(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output,'. . K . .\n. . . K .')

#%% _get_new_position
class Test__get_new_position(unittest.TestCase):
    r"""
    Tests the _get_new_position function with the following cases:
        All valid moves
        Not yet done moves
        Invalid moves
        Transport move
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
        self.transports = knight._get_transports(self.board)
        self.valid_moves = [-4, -3, -2, -1, 1, 2, 3, 4]
        self.future_moves = [-8, -7, -6, -5, 5, 6, 7, 8]
        self.bad_moves = [0, 9, -9, 100]
        self.results = [(3, 0), (4, 3), (1, 4), (0, 1), (0, 3), (3, 4), (4, 1), (1, 0)]

    def test_valid_moves(self):
        for (this_move, this_result) in zip(self.valid_moves, self.results):
            (pos1, pos2, pos3) = knight._get_new_position(self.x, self.y, this_move, self.transports)
            self.assertEqual(pos3, this_result)
            # TODO: assert something about pos1 and pos2?

    def test_future_moves(self):
        for (this_move, this_result) in zip(self.future_moves, self.results):
            with self.assertRaises(ValueError):
                knight._get_new_position(self.x, self.y, this_move, self.transports)
            #(pos1, pos2, pos3) = knight._get_new_position(self.x, self.y, this_move, self.transports)
            #self.assertEqual(pos3, this_result)

    def test_bad_moves(self):
        for this_move in self.bad_moves:
            with self.assertRaises(ValueError):
                knight._get_new_position(self.x, self.y, this_move, self.transports)

    def test_transport(self):
        self.board[3, 4] = knight.Piece.transport
        self.board[4, 3] = knight.Piece.transport
        self.transports = knight._get_transports(self.board)
        for (this_move, this_result) in zip(self.valid_moves, self.results):
            (pos1, pos2, pos3) = knight._get_new_position(self.x, self.y, this_move, self.transports)
            if pos3 in self.transports:
                self.assertEqual(pos3, (this_result[1], this_result[0]))
            else:
                self.assertEqual(pos3, this_result)

#%% _check_board_boundaries
class Test__check_board_boundaries(unittest.TestCase):
    r"""
    Tests the _check_board_boundaries function with the following cases:
        Good values
        Bad X values
        Bad Y values
        Bad X and Y values
    """
    def setUp(self):
        self.xmax  = 4
        self.ymax  = 3
        self.x     = np.arange(self.xmax + 1)
        self.y     = np.arange(self.ymax + 1)
        self.bad_x = [-1, self.xmax + 1]
        self.bad_y = [-1, self.ymax + 1]

    def test_good_values(self):
        for this_x in self.x:
            for this_y in self.y:
                is_valid = knight._check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertTrue(is_valid)

    def test_bad_values1(self):
        for this_x in self.bad_x:
            for this_y in self.y:
                is_valid = knight._check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

    def test_bad_values2(self):
        for this_x in self.x:
            for this_y in self.bad_y:
                is_valid = knight._check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

    def test_bad_values3(self):
        for this_x in self.bad_x:
            for this_y in self.bad_y:
                is_valid = knight._check_board_boundaries(this_x, this_y, self.xmax, self.ymax)
                self.assertFalse(is_valid)

#%% _classify_move
class Test__classify_move(unittest.TestCase):
    r"""
    Tests the _classify_move function with the following cases:
        Normal
        Off board
        Land on barrier or rock
        Try to jump barrier
        Land on rock
        Visited
        Winning
        Transport
        Water
        Lava
    """
    def setUp(self):
        self.board      = knight.Piece.null * np.ones((2, 5), dtype=int)
        self.move       = 2 # (2 right and 1 down, so end at [1, 4]
        self.move_type  = knight.Move.normal
        self.transports = knight._get_transports(self.board)
        self.start_x    = 0
        self.start_y    = 2
        self.board[self.start_x, self.start_y] = knight.Piece.current

    def test_normal(self):
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, self.move_type)

    def test_off_board(self):
        move_type = knight._classify_move(self.board, -2, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.off_board)

    def test_land_on_barrier_or_rock(self):
        self.board[1, 4] = knight.Piece.rock
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.blocked)
        self.board[1, 4] = knight.Piece.barrier
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.blocked)

    def test_cant_pass_barrier(self):
        self.board[0, 4] = knight.Piece.barrier
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.blocked)

    def test_over_rock(self):
        self.board[0, 4] = knight.Piece.rock
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, self.move_type)

    def test_visited(self):
        self.board[1, 4] = knight.Piece.visited
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.visited)

    def test_winning(self):
        self.board[1, 4] = knight.Piece.final
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.winning)

    def test_transport(self):
        self.board[1, 4] = knight.Piece.transport
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.transport)

    def test_water(self):
        self.board[1, 4] = knight.Piece.water
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.water)

    def test_lava(self):
        self.board[1, 4] = knight.Piece.lava
        move_type = knight._classify_move(self.board, self.move, self.transports, self.start_x, self.start_y)
        self.assertEqual(move_type, knight.Move.lava)

    def test_unexpected_piece(self):
        pass

#%% _update_board
class Test__update_board(unittest.TestCase):
    r"""
    Tests the _update_board function with the following cases:
        Normal
        Invalid move
        Repeated move
    """
    def setUp(self):
        self.old_x       = 0
        self.old_y       = 2
        self.move        = 2 # 2 right and 1 down
        self.new_x       = 1
        self.new_y       = 4
        self.board       = knight.Piece.null * np.ones((2, 5), dtype=int)
        self.board[self.old_x, self.old_y] = knight.Piece.current
        self.cost        = 5
        self.costs       = self.cost * np.ones(self.board.shape, dtype=int)
        self.transports  = knight._get_transports(self.board)

    def test_normal(self):
        (cost, is_repeat, new_x, new_y) = knight._update_board(self.board, self.move, self.costs, \
            self.transports, self.old_x, self.old_y)
        self.assertEqual(cost, self.costs[self.new_x, self.new_y])
        self.assertFalse(is_repeat)
        self.assertEqual(new_x, self.new_x)
        self.assertEqual(new_y, self.new_y)
        self.assertEqual(self.board[self.old_x, self.old_y], knight.Piece.visited)
        self.assertEqual(self.board[self.new_x, self.new_y], knight.Piece.current)

    def test_invalid_move(self):
        (cost, is_repeat, new_x, new_y) = knight._update_board(self.board, -2, self.costs, \
            self.transports, self.old_x, self.old_y)
        self.assertEqual(cost, knight.LARGE_INT)
        self.assertFalse(is_repeat)
        self.assertEqual(new_x, self.old_x)
        self.assertEqual(new_y, self.old_y)
        self.assertEqual(self.board[self.old_x, self.old_y], knight.Piece.current)
        self.assertEqual(self.board[self.new_x, self.new_y], knight.Piece.null)

    def test_repeated_move(self):
        self.board[self.new_x, self.new_y] = knight.Piece.visited
        (cost, is_repeat, new_x, new_y) = knight._update_board(self.board, self.move, self.costs, \
            self.transports, self.old_x, self.old_y)
        self.assertEqual(cost, self.cost)
        self.assertTrue(is_repeat)
        self.assertEqual(new_x, self.new_x)
        self.assertEqual(new_y, self.new_y)
        self.assertEqual(self.board[self.old_x, self.old_y], knight.Piece.visited)
        self.assertEqual(self.board[self.new_x, self.new_y], knight.Piece.current)

#%% _undo_move
class Test__undo_move(unittest.TestCase):
    r"""
    Tests the _undo_move function with the following cases:
        Normal
        Other piece to replace
        Transport
    """
    def setUp(self):
        self.board       = knight.Piece.null * np.ones((2, 5), dtype=int)
        self.board[0, 2] = knight.Piece.visited
        self.last_move   = 2 # 2 right and 1 down
        self.original_board       = knight.Piece.null * np.ones((2, 5), dtype=int)
        self.original_board[0, 2] = knight.Piece.start
        self.transports  = knight._get_transports(self.original_board)
        self.start_x     = 1
        self.start_y     = 4
        self.board[self.start_x, self.start_y] = knight.Piece.current

    def test_normal(self):
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[0, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], self.original_board[1, 4])

    def test_other_piece(self):
        self.original_board[1, 4] = knight.Piece.water
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[0, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], self.original_board[1, 4])

    def test_transport1(self):
        self.board[0, 0] = knight.Piece.transport
        self.board[1, 1] = knight.Piece.transport
        self.transports  = knight._get_transports(self.board)
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[0, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], self.original_board[1, 4])

    def test_transport2(self):
        self.board[0, 0] = knight.Piece.transport
        self.board[0, 2] = knight.Piece.transport
        self.transports  = knight._get_transports(self.board)
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[0, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], self.original_board[1, 4])
        self.assertEqual(self.board[0, 0], knight.Piece.transport)

    def test_transport3(self):
        self.board[0, 2] = knight.Piece.transport
        self.board[0, 4] = knight.Piece.transport
        self.transports  = knight._get_transports(self.board)
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[0, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], self.original_board[1, 4])
        self.assertEqual(self.board[0, 4], knight.Piece.transport)

    def test_transport4(self):
        self.last_move = 4
        self.board[0, 0] = knight.Piece.transport
        self.original_board[0, 0] = knight.Piece.transport
        self.original_board[1, 4] = knight.Piece.transport
        self.transports  = [(0, 0), (1, 4)]
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[1, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], knight.Piece.transport)
        self.assertEqual(self.board[0, 0], knight.Piece.transport)

    def test_transport5(self):
        self.last_move = 4
        self.board[0, 0] = knight.Piece.transport
        self.original_board[0, 0] = knight.Piece.transport
        self.original_board[1, 4] = knight.Piece.transport
        self.transports  = [(1, 4), (0, 0)] # manually reverse their order
        knight._undo_move(self.board, self.last_move, self.original_board, self.transports, self.start_x, self.start_y)
        self.assertEqual(self.board[1, 2], knight.Piece.current)
        self.assertEqual(self.board[1, 4], knight.Piece.transport)
        self.assertEqual(self.board[0, 0], knight.Piece.transport)

#%% _get_move_inverse
class Test__get_move_inverse(unittest.TestCase):
    r"""
    Tests the _get_move_inverse function with the following cases:
        All inverses
        Bad moves
    """
    def setUp(self):
        self.moves     = [-4, -3, -2, -1, 1, 2, 3, 4]
        self.inv_moves = [-2, -1, -4, -3, 3, 4, 1, 2]
        self.bad_moves = [1000, -2000]

    def test_nominal(self):
        for (this_move, this_inv_move) in zip(self.moves, self.inv_moves):
            inv_move = knight._get_move_inverse(this_move)
            self.assertEqual(inv_move, this_inv_move)

    def test_bad_moves(self):
        for this_move in self.bad_moves:
            with self.assertRaises(AssertionError):
                knight._get_move_inverse(this_move)

#%% _predict_cost
class Test__predict_cost(unittest.TestCase):
    r"""
    Tests the _predict_cost function with the following cases:
        Nominal
    """
    def setUp(self):
        self.board = np.zeros((2,5), dtype=int)
        self.board[0, 0] = knight.Piece.start
        self.board[0, 4] = knight.Piece.final
        self.costs = np.array([[2, 1.5, 1, 0.5, 0], [2, 1.5, 1, 1, 0.5]])

    def test_nominal(self):
        costs = knight._predict_cost(self.board)
        np.testing.assert_array_equal(costs, self.costs)

#%% _sort_best_moves
class Test__sort_best_moves(unittest.TestCase):
    r"""
    Tests the _sort_best_moves function with the following cases:
        Nominal
    """
    def setUp(self):
        self.board = np.zeros((2,5), dtype=int)
        self.board[0, 0] = knight.Piece.current
        self.board[0, 4] = knight.Piece.final
        self.moves = knight.MOVES
        self.costs = knight._predict_cost(self.board)
        self.transports = None
        self.start_x = 0
        self.start_y = 0
        self.sorted_moves = np.array([-1, -4, -2, 2, 4, 1], dtype=int)

    def test_nominal(self):
        sorted_moves = knight._sort_best_moves(self.board, self.moves, self.costs, self.transports, \
            self.start_x, self.start_y)
        np.testing.assert_array_equal(sorted_moves, self.sorted_moves)

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
        self.board3[self.board3 > max(knight.Piece).value] = 0

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
        self.enums       = knight.Piece.__members__.values()
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

#%% check_valid_sequence
class Test_check_valid_sequence(unittest.TestCase):
    r"""
    Tests the check_valid_sequence function with the following cases:
        Normal, no printing
        Normal, with printing
        No final position on board
        Invalid sequence
        Repeated square sequence (x3)
        Good, but incomplete sequence
        Good, but extra moves in sequence
    """
    def setUp(self):
        self.board       = knight.Piece.null * np.ones((3, 5), dtype=int)
        self.board[0, 0] = knight.Piece.start
        self.board[2, 4] = knight.Piece.final
        self.moves       = [2, 2]

    def test_normal(self):
        is_valid = knight.check_valid_sequence(self.board, self.moves, print_status=False)
        self.assertTrue(is_valid)

    def test_printing(self):
        with capture_output() as (out, _):
            is_valid = knight.check_valid_sequence(self.board, self.moves, print_status=True)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output, 'Sequence is valid and finished the puzzle.')

    def test_no_final(self):
        self.board[2, 4] = knight.Piece.water
        with self.assertRaises(ValueError):
            knight.check_valid_sequence(self.board, self.moves, print_status=False)

    def test_bad_sequence(self):
        with capture_output() as (out, _):
            is_valid = knight.check_valid_sequence(self.board, [-2, -2], print_status=True)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_valid)
        self.assertEqual(output, 'Sequence is not valid.')

    def test_repeated_sequence1(self):
        is_valid = knight.check_valid_sequence(self.board, [2, 4, 2, 2])
        self.assertFalse(is_valid)

    def test_repeated_sequence2(self):
        with capture_output() as (out, _):
            is_valid = knight.check_valid_sequence(self.board, [2, 4, 2, 2], print_status=True)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_valid)
        self.assertEqual(output, 'No repeats allowed.\nSequence is not valid.')

    def test_repeated_sequence3(self):
        is_valid = knight.check_valid_sequence(self.board, [2, 4, 2, 2], allow_repeats=True)
        self.assertTrue(is_valid)

    def test_good_but_incomplete_sequence(self):
        with capture_output() as (out, _):
            is_valid = knight.check_valid_sequence(self.board, self.moves[:-1], print_status=True)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(is_valid)
        self.assertEqual(output, 'Sequence is valid, but did not finish the puzzle.')

    def test_good_but_extra_sequence(self):
        self.moves.append(2)
        with self.assertRaises(ValueError):
            knight.check_valid_sequence(self.board, self.moves, print_status=False)

#%% print_sequence
class Test_print_sequence(unittest.TestCase):
    r"""
    Tests the print_sequence function with the following cases:
        Normal
        Non standard costs
        Invalid sequence
    """
    def setUp(self):
        self.board       = knight.Piece.null * np.ones((3, 5), dtype=int)
        self.board[0, 0] = knight.Piece.start
        self.board[2, 4] = knight.Piece.final
        self.moves       = [2, 2]
        self.output      = 'Starting position:\nS . . . .\n. . . . .\n. . . . E\n\n' + \
            'After move 1, cost: 1\nx . . . .\n. . K . .\n. . . . E\n\nAfter move 2, cost: 2\n' + \
            'x . . . .\n. . x . .\n. . . . K'
        self.output2     = 'Starting position:\nS W W W W\nW W W W W\nW W W W E\n\n' + \
            'After move 1, cost: 2\nx W W W W\nW W K W W\nW W W W E\n\nAfter move 2, cost: 3\n' + \
            'x W W W W\nW W x W W\nW W W W K'

    def test_normal(self):
        with capture_output() as (out, _):
            knight.print_sequence(self.board, self.moves)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.output)

    def test_other_costs(self):
        self.board = np.where(self.board == knight.Piece.null, knight.Piece.water, self.board)
        with capture_output() as (out, _):
            knight.print_sequence(self.board, self.moves)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.output2)

    def test_invalid_sequence(self):
        self.moves = [-2, -2]
        with capture_output() as (out, _):
            with self.assertRaises(ValueError):
                knight.print_sequence(self.board, self.moves)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.output[:len(output)])

#%% solve_min_puzzle
class Test_solve_min_puzzle(unittest.TestCase):
    r"""
    Tests the solve_min_puzzle function with the following cases:
        min solver
        Unsolvable
        No final position
    """
    def setUp(self):
        self.board       = knight.Piece.null * np.ones((3, 5), dtype=int)
        self.board[0, 0] = knight.Piece.start
        self.board[0, 4] = knight.Piece.final
        self.moves       = [2, -2]

    def test_min(self):
        with capture_output() as (out, _):
            moves = knight.solve_min_puzzle(self.board)
        output = out.getvalue().strip()
        out.close()
        np.testing.assert_array_equal(moves, self.moves)
        expected_output_start = 'Initializing solver.\nSolution found for cost of: 2.'
        self.assertEqual(output[:len(expected_output_start)], expected_output_start)

    def test_no_solution(self):
        board = knight.Piece.null * np.ones((2, 5), dtype=int)
        board[0, 0] = knight.Piece.start
        board[1, 4] = knight.Piece.final
        with capture_output() as (out, _):
            moves = knight.solve_min_puzzle(board)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(len(moves) == 0)
        expected_output_start = 'Initializing solver.\nNo solution found.'
        self.assertEqual(output[:len(expected_output_start)], expected_output_start)

    def test_no_final_position(self):
        self.board[0, 4] = knight.Piece.null
        with capture_output() as (out, _):
            with self.assertRaises(ValueError):
                knight.solve_min_puzzle(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Initializing solver.')

#%% solve_max_puzzle
class Test_solve_max_puzzle(unittest.TestCase):
    r"""
    Tests the solve_max_puzzle function with the following cases:
        max solver
        Unsolvable
    """
    def setUp(self):
        self.board       = knight.Piece.null * np.ones((3, 5), dtype=int)
        self.board[0, 0] = knight.Piece.start
        self.board[0, 4] = knight.Piece.final
        self.moves       = [2, -2] # TODO: should be 8 moves long?

    @unittest.skip('Not yet implemented.')
    def test_max(self):
        with capture_output() as (out, _):
            moves = knight.solve_max_puzzle(self.board)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(moves, self.moves)
        expected_output_start = 'Initializing solver.\nSolution found for cost of: 8.'
        self.assertEqual(output[:len(expected_output_start)], expected_output_start)

    def test_no_solution(self):
        board = knight.Piece.null * np.ones((2, 5), dtype=int)
        board[0, 0] = knight.Piece.start
        board[1, 4] = knight.Piece.final
        with capture_output() as (out, _):
            moves = knight.solve_max_puzzle(board)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(len(moves) == 0)
        expected_output_start = 'Initializing solver.\nNo solution found.'
        self.assertEqual(output[:len(expected_output_start)], expected_output_start)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
