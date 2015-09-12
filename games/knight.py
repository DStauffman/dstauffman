# -*- coding: utf-8 -*-
"""
The "knight" file solves the Knight Board puzzle given to Matt Beck during a job interview.

Notes
-----
#.  Written by David C. Stauffer and Matt Beck in September 2015.

[Knight Board]
The knight board can be represented in x,y coordinates.  The upper left position
is (0,0) and the bottom right is (7,7).  Assume there is a single knight chess
piece on the board that can move according to chess rules.  Sample S[tart] and
E[nd] points are shown below:
    . . . . . . . .
    . . . . . . . .
    . S . . . . . .
    . . . . . . . .
    . . . . . E . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
Level 1: Write a function that accepts a sequence of moves and reports
    whether the sequence contains only valid knight moves.  It should also
    optionally print the state of the knight board to the terminal as shown
    above after each move.  The current position should be marked with a 'K'.
Level 2: Compute a valid sequence of moves from a given start point to a given
    end point.
Level 3: Compute a valid sequence of moves from a given start point to a
    given end point in the fewest number of moves.
Level 4: Now repeat the Level 3 task for this 32x32 board.  Also, modify
    your validator from Level 1 to check your solutions.  This board has the
    following additional rules:
        1) W[ater] squares count as two moves when a piece lands there
        2) R[ock] squares cannot be used
        3) B[arrier] squares cannot be used AND cannot lie in the path
        4) T[eleport] squares instantly move you from one T to the other in
            the same move
        5) L[ava] squares count as five moves when a piece lands there
    . . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . . L L L . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . L L L . . . R R . . . . . . .
    . . . . . . . . B . . . L L L L L L L L . . . R R . . . . . . .
    . . . . . . . . B . . . L L L L L L . . . . . . . . . . . . . .
    . . . . . . . . B . . . . . . . . . . . . R R . . . . . . . . .
    . . . . . . . . B B . . . . . . . . . . . R R . . . . . . . . .
    . . . . . . . . W B B . . . . . . . . . . . . . . . . . . . . .
    . . . R R . . . W W B B B B B B B B B B . . . . . . . . . . . .
    . . . R R . . . W W . . . . . . . . . B . . . . . . . . . . . .
    . . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .
    . . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .
    . . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .
    . . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W
    . . . W W . . . . . . . . . . B . . . . . . . . . W . . . . . .
    W W W W . . . . . . . . . . . B . . . W W W W W W W . . . . . .
    . . . W W W W W W W . . . . . B . . . . . . . . . . . . B B B B
    . . . W W W W W W W . . . . . B B B . . . . . . . . . . B . . .
    . . . W W W W W W W . . . . . . . B W W W W W W B B B B B . . .
    . . . W W W W W W W . . . . . . . B W W W W W W B . . . . . . .
    . . . . . . . . . . . B B B . . . . . . . . . . B B . . . . . .
    . . . . . R R . . . . B . . . . . . . . . . . . . B . . . . . .
    . . . . . R R . . . . B . . . . . . . . . . . . . B . T . . . .
    . . . . . . . . . . . B . . . . . R R . . . . . . B . . . . . .
    . . . . . . . . . . . B . . . . . R R . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .  # The last four rows originally missing
    . . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
Level 5 [HARD]: Compute the longest sequence of moves to complete Level 3 without
    visiting the same square twice.  Use the 32x32 board.
"""

#%% Imports
# backwards compatibility with v2.7
from __future__ import print_function
from __future__ import division
# regular imports
import doctest
from enum import unique
import numpy as np
import unittest
# personal dstauffman libary imports
from dstauffman import IntEnumPlus

#%% Constants
CHAR_DICT = {'.':0, 'S':1, 'E':2, 'K':3, 'W':4, 'R':5, 'B':6, 'T':7, 'L':8, 'x': 9}
NUM_DICT = {value:key for (key, value) in CHAR_DICT.items()}
BOARD1 = r"""
. . . . . . . .
. . . . . . . .
. S . . . . . .
. . . . . . . .
. . . . . E . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
"""
BOARD2 = r"""
. . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
. . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
. . . . . . . . B . . . L L L . . . L L L . . . . . . . . . . .
. . . . . . . . B . . . L L L . . L L L . . . R R . . . . . . .
. . . . . . . . B . . . L L L L L L L L . . . R R . . . . . . .
. . . . . . . . B . . . L L L L L L . . . . . . . . . . . . . .
. . . . . . . . B . . . . . . . . . . . . R R . . . . . . . . .
. . . . . . . . B B . . . . . . . . . . . R R . . . . . . . . .
. . . . . . . . W B B . . . . . . . . . . . . . . . . . . . . .
. . . R R . . . W W B B B B B B B B B B . . . . . . . . . . . .
. . . R R . . . W W . . . . . . . . . B . . . . . . . . . . . .
. . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .
. . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .
. . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .
. . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W
. . . W W . . . . . . . . . . B . . . . . . . . . W . . . . . .
W W W W . . . . . . . . . . . B . . . W W W W W W W . . . . . .
. . . W W W W W W W . . . . . B . . . . . . . . . . . . B B B B
. . . W W W W W W W . . . . . B B B . . . . . . . . . . B . . .
. . . W W W W W W W . . . . . . . B W W W W W W B B B B B . . .
. . . W W W W W W W . . . . . . . B W W W W W W B . . . . . . .
. . . . . . . . . . . B B B . . . . . . . . . . B B . . . . . .
. . . . . R R . . . . B . . . . . . . . . . . . . B . . . . . .
. . . . . R R . . . . B . . . . . . . . . . . . . B . T . . . .
. . . . . . . . . . . B . . . . . R R . . . . . . B . . . . . .
. . . . . . . . . . . B . . . . . R R . . . . . . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
"""
MOVES = frozenset({-4, -3, -2, -1, 1, 2, 3, 4})
#MOVES = frozenset({-4, -3, -2, -1, 1, 2, 3, 4, -5, 5, -6, 6, -7, 7, -8, 8})
# Move design is based on a first direction step of two moves, followed by a single step to the
# left or right of the first one.
# Note for chess nerds: The move is two steps and then one.  With no obstructions, it can be thought
# of as one and then two, but with obstructions, this counts as a different move and is not allowed.
# The number represents the cardinal direction of the move: 1=North, 2=East, 3=South, 4=West
# The sign of the number represents whether the second step is left or right (positive is clockwise)
# Therefore:
"""
#  Move -1       Move +1        Move -2      Move +2       Move -3       Move +3       Move -4       Move +4
# . E x . .  |  . . x E .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .
# . . x . .  |  . . x . .  |  . . . . E  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  E . . . .
# . . S . .  |  . . S . .  |  . . S x x  |  . . S x x  |  . . S . .  |  . . S . .  |  x x S . .  |  x x S . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . E  |  . . x . .  |  . . x . .  |  E . . . .  |  . . . . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . x E .  |  . E x . .  |  . . . . .  |  . . . . .
"""

# Optional extended set?
"""
#  Move -5       Move +5        Move -6      Move +6       Move -7       Move +7       Move -8       Move +8
# . E . . .  |  . . . E .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .
# . x . . .  |  . . . x .  |  . . x x E  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  E x x . .
# . x S . .  |  . . S x .  |  . . S . .  |  . . S . .  |  . . S x .  |  . x S . .  |  . . S . .  |  . . S . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . x x E  |  . . . x .  |  . x . . .  |  E x x . .  |  . . . . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . E .  |  . E . . .  |  . . . . .  |  . . . . .
"""
NORMAL_COST    = 1
TRANSPORT_COST = 1
WATER_COST     = 2
LAVA_COST      = 5
INVALID_COST   = 0

#%% Classes
@unique
class Piece(IntEnumPlus):
    r"""
    Enumerator for all the possible types of squares within the board, including start and end positions
    """
    null      = 0 # Empty square that has never been used
    start     = 1 # Original starting position
    final     = 2 # Final ending position
    current   = 3 # Current knight position
    water     = 4 # water, costs two moves to land on
    rock      = 5 # rock, can be jumped across, but not landed on
    barrier   = 6 # barrier, cannot be landed on or moved across
    transport = 7 # transports to the other transport, can only be exactly 0 or 2 on the board
    lava      = 8 # lava, costs 5 moves to land on
    visited   = 9 # previously visited square that cannot be used again

    @staticmethod
    def is_valid_place(enumerator):
        r"""Determines if the piece is a valid place to put a knight."""
        if enumerator in {Piece.null, Piece.final, Piece.water, Piece.transport, Piece.lava}:
            return True
        else:
            return False

#%% print_baord
def print_board(board):
    r"""
    Prints the current board position to the console window.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((5,5), dtype=int)
    >>> board[2,2] = Piece.current
    >>> print_board(board)
    . . . . .
    . . . . .
    . . K . .
    . . . . .
    . . . . .

    """
    # get the board shape
    (rows, cols) = board.shape
    # loop through pieces
    for i in range(rows):
        for j in range(cols):
            # print each piece in the row without a line continuation
            pad = ' ' if j < cols-1 else ''
            print(NUM_DICT[board[i,j]] + pad, end='')
        # add the line continuation at the end of each row
        print('')

#%% char_board_to_nums
def char_board_to_nums(char_board):
    r"""
    Converts the original board from a character array into a numpy ndarray of int.
    """
    # convert to rows of lines
    lines = char_board.split('\n')
    # remove any empty rows
    lines = [this_line.split(' ') for this_line in lines if this_line]
    # get the size of the board
    rows = len(lines)
    cols = len(lines[0])
    # preallocate null board
    board = np.zeros((rows, cols), dtype=int)
    # loop through and store all pieces and integer equivalents
    for i in range(rows):
        for j in range(cols):
            board[i,j] = CHAR_DICT[lines[i][j]]
    return board

#%% get_current_position
def get_current_position(board):
    r"""
    Gets the current position of the knight.
    """
    ix = np.nonzero(board == Piece.current)
    x = ix[0]
    y = ix[1]
    is_valid = len(x) == 1 and len(y) == 1
    if not is_valid:
        print_board(board)
    assert is_valid, 'Only exactly one current position may be found.'
    return (x, y)

#%% get_new_position
def get_new_position(x, y, move):
    r"""
    Gets the new position of the knight after making the desired move.
    """
    # move the piece
    if move == -1:
        pos1 = (x-1, y)
        pos2 = (x-2, y)
        pos3 = (x-2, y-1)
    elif move == 1:
        pos1 = (x-1, y)
        pos2 = (x-2, y)
        pos3 = (x-2, y+1)
    elif move == -2:
        pos1 = (x,   y+1)
        pos2 = (x,   y+2)
        pos3 = (x-1, y+2)
    elif move == 2:
        pos1 = (x,   y+1)
        pos2 = (x,   y+2)
        pos3 = (x+1, y+2)
    elif move == -3:
        pos1 = (x+1, y)
        pos2 = (x+2, y)
        pos3 = (x+2, y+1)
    elif move == 3:
        pos1 = (x+1, y)
        pos2 = (x+2, y)
        pos3 = (x+2, y-1)
    elif move == -4:
        pos1 = (x,   y-1)
        pos2 = (x,   y-2)
        pos3 = (x+1, y-2)
    elif move == 4:
        pos1 = (x,   y-1)
        pos2 = (x,   y-2)
        pos3 = (x-1, y-2)
    elif move in {-5, 5, -6, 6, -7, 7, -8, 8}:
        raise ValueError('Extended moves are not yet implemented.')
    else:
        raise ValueError('Invalid move of "{}"'.format(move))
    # return the whole set of stuff
    return (pos1, pos2, pos3)

#%% move_cost
def move_cost(board, move):
    r"""
    Determines if the desired move is valid, and if so returns a non zero cost.

    A negative cost means the move finishes the puzzle, and the absolute value should be used.
    """
    # find the size of the board
    (xmax, ymax) = board.shape
    # find current position
    (x, y) = get_current_position(board)
    # find the traversal for the desired move
    (pos1, pos2, pos3) = get_new_position(x, y, move)
    # first check that pos3 is on the board and if not return
    if pos3[0] < 0 or pos3[0] > xmax or pos3[1] < 0 or pos3[1] > ymax:
        return INVALID_COST
    # check that pos1 and pos2 are on the board
    pass # TODO: this only matters for non-rectangular boards
    # get the values for each position
    p1 = board[pos1[0], pos1[1]][0]
    p2 = board[pos2[0], pos2[1]][0]
    p3 = board[pos3[0], pos3[1]][0]
    # check for valid landing squares
    if not Piece.is_valid_place(p3):
        return INVALID_COST
    # check for barriers in path
    if p1 == Piece.barrier or p2 == Piece.barrier:
        return INVALID_COST
    # move is valid, so now determine the cost
    if p3 == Piece.null:
        cost = NORMAL_COST
    elif p3 == Piece.final:
        cost = -NORMAL_COST # TODO: can the final piece be something besides a normal empty square?
    elif p3 == Piece.transport:
        cost = TRANSPORT_COST
    elif p3 == Piece.water:
        cost = WATER_COST
    elif p3 == Piece.lava:
        cost = LAVA_COST
    return cost

#%% update_board
def update_board(board, move):
    r"""
    Updates the new board based on the desired move.

    Notes
    -----
    #.  Modifies `board` in-place.
    """
    # find current position
    (x, y) = get_current_position(board)
    # determine the move cost
    cost = move_cost(board, move)
    # if valid, apply the move
    if cost:
        # set the current position to visited
        board[x, y] = Piece.visited
        # get the new position
        (_, _, (new_x, new_y)) = get_new_position(x, y, move)
        # set the new position to current
        board[new_x, new_y] = Piece.current
    # return the cost of the update, zero means no update occured
    return cost

#%% check_valid_sequence
def check_valid_sequence(board, moves, print_status=False):
    r"""
    Checks that the list of moves is a valid sequence to go from start to final position.
    """
    # initialize output
    is_valid = True
    is_done  = False
    # create internal board for calculations
    temp_board = board.copy()
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    for (i, this_move) in enumerate(moves):
        cost = update_board(temp_board, this_move)
        if cost == 0:
            is_valid = False
            break
        if cost < 0:
            is_done = True
            if i < len(moves)-1:
                raise ValueError('Sequence finished, but then kept going.')
    if print_status:
        if is_valid:
            if is_done:
                print('Sequence is valid and finished the puzzle.')
            else:
                print('Sequence is valid, but did not finish the puzzle.')
        else:
            print('Sequence is not valid.')
    return is_valid

#%% print_sequence
def print_sequence(board, moves):
    r"""
    Prints the every board position for the given move sequence.
    """
    # create internal board for calculations
    temp_board = board.copy()
    print('Starting position: ')
    print_board(temp_board)
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    # loop through move sequence
    for (i,this_move) in enumerate(moves):
        # print header
        print('\nAfter move {}'.format(i+1))
        # update board
        cost = update_board(temp_board, this_move)
        # print new board
        if cost:
            print_board(temp_board)
        else:
            raise ValueError('Bad sequence.')

#%% solve_puzzle
def solve_puzzle(board, solve_type='min'):
    r"""
    Solves the puzzle with the desired solution type, from 'min', 'max', 'first'.
    """
    moves = []
    return moves

#%% Unit test
def main():
    unittest.main(module='dstauffman.games.test_knight', exit=False)
    doctest.testmod(verbose=False)

#%% Script
if __name__ == '__main__':
    # run unit tests
    main()

    #%% Solve puzzle
    # convert board to numeric representation for efficiency
    board1 = char_board_to_nums(BOARD1)
    board2 = char_board_to_nums(BOARD2)
    board3 = np.zeros((5,5), dtype=int)
    board3[2,2] = Piece.start

    print_board(board1)
    print_board(board2)

    # Step 1
    moves1 = [2, 2]
    is_valid = check_valid_sequence(board1, moves1, print_status=True)
    if is_valid:
        print_sequence(board1, moves1)

    # Step 2
    print('')
    moves2 = solve_puzzle(board1, solve_type='first')
    is_valid = check_valid_sequence(board1, moves2, print_status=True)
    if is_valid:
        print_sequence(board1, moves2)
