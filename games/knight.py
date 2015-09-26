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
# hard-coded values
LARGE_INT = 1000000
# dictionaries
CHAR_DICT = {'.':0, 'S':1, 'E':2, 'K':3, 'W':4, 'R':5, 'B':6, 'T':7, 'L':8, 'x': 9}
NUM_DICT  = {value:key for (key, value) in CHAR_DICT.items()}
COST_DICT = {'normal': 1, 'transport': 1, 'water': 2, 'lava': 5, 'invalid': LARGE_INT, 'start': 0}
# boards
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
# moves
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

#%% Classes - Piece
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

#%% Classes - Move
@unique
class Move(IntEnumPlus):
    r"""
    Enumerator for all the cost outcomes for moving a piece.
    """
    off_board = -2
    blocked   = -1
    visited   = 0
    normal    = 1
    transport = 2
    water     = 3
    lava      = 4
    winning   = 5

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

    Parameters
    ----------
    char_board : str
        Board as an original character array, where each line is a row of the board

    Returns
    -------
    board : 2D ndarray of int
        Board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import char_board_to_nums
    >>> char_board = '. . S . .\n. . . . E'
    >>> board = char_board_to_nums(char_board)
    >>> print(board)
    [[0 0 1 0 0]
     [0 0 0 0 2]]

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

#%% board_to_costs
def board_to_costs(board):
    r"""
    Translates a board to the associated costs for landing on any square within the board.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    costs : 2D ndarray of int
        Costs for each square on the board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import board_to_costs, Piece
    >>> import numpy as np
    >>> board = Piece.null * np.ones((3, 3), dtype=int)
    >>> board[0, 0] = Piece.water
    >>> board[1, 1] = Piece.start
    >>> board[2, 2] = Piece.barrier
    >>> costs = board_to_costs(board)
    >>> print(costs) # doctest: +NORMALIZE_WHITESPACE
    [[      2       1       1]
     [      1       0       1]
     [      1       1 1000000]]

    """
    costs = np.empty(board.shape, dtype=int)
    costs.fill(COST_DICT['invalid'])
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            this_piece = board[i, j]
            if this_piece in {Piece.rock, Piece.barrier}:
                continue
            elif this_piece in {Piece.null, Piece.final}:
                costs[i, j] = COST_DICT['normal']
            elif this_piece == Piece.start:
                costs[i, j] = COST_DICT['start']
            elif this_piece == Piece.transport:
                costs[i, j] = COST_DICT['transport']
            elif this_piece == Piece.water:
                costs[i, j] = COST_DICT['water']
            elif this_piece == Piece.lava:
                costs[i, j] = COST_DICT['lava']
            else:
                raise ValueError('Cannot convert piece "{}" to a cost.'.format(this_piece))
    return costs

#%% get_current_position
def get_current_position(board):
    r"""
    Gets the current position of the knight.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    x : int
        Current X position
    y : int
        Current Y position

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import get_current_position, Piece
    >>> import numpy as np
    >>> board = np.zeros((5,5), dtype=int)
    >>> board[1,4] = Piece.current
    >>> (x, y) = get_current_position(board)
    >>> print(x, y)
    1 4

    """
    # find position
    ix = np.nonzero(board == Piece.current)
    # alias X and Y locations
    x = ix[0]
    y = ix[1]
    # check that only exactly one current position was found, and if not, print the messed up board
    is_valid = len(x) == 1 and len(y) == 1
    if not is_valid:
        print_board(board)
    assert is_valid, 'Only exactly one current position may be found.'
    # otherwise if all is good, return the location
    return (x[0], y[0])

#%% get_new_position
def get_new_position(x, y, move):
    r"""
    Gets the new position of the knight after making the desired move.

    Parameters
    ----------
    x : int
        Current X position
    y : int
        Current Y position
    move : int
        Move to be performed

    Returns
    -------
    pos1 : (x,y) tuple
        X and Y positions for step 1 in the move
    pos2 : (x,y) tuple
        X and Y positions for step 2 in the move
    pos3 : (x,y) tuple
        X and Y positions for step 3 in the move

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import get_new_position
    >>> x = 2
    >>> y = 3
    >>> move = 2 # (2 right and 1 down)
    >>> (pos1, pos2, pos3) = get_new_position(x, y, move)
    >>> print(pos1, pos2, pos3)
    (2, 4) (2, 5) (3, 5)

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

#%% check_board_boundaries
def check_board_boundaries(x, y, xmax, ymax):
    r"""
    Checks that a given position is on the board.

    Parameters
    ----------
    x : int
        Current X position
    y : int
        Current Y position
    xmax : int
        Current board X maximum
    ymax : int
        Current board Y maximum

    Returns
    -------
    is_valid : bool
        Whether position is valid (defined as on the board)

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  This current assumes a rectangular board, but is a separate function in case that ever gets
        more complicated.

    Examples
    --------

    >>> from dstauffman.games.knight import check_board_boundaries
    >>> x = 2
    >>> y = 5
    >>> xmax = 7
    >>> ymax = 7
    >>> is_valid = check_board_boundaries(x, y, xmax, ymax)
    >>> print(is_valid)
    True

    """
    # determine if X position is valid
    x_valid = 0 <= x <= xmax
    # determine if Y position is valid
    y_valid = 0 <= y <= ymax
    # combine results and return
    is_valid = x_valid and y_valid
    return is_valid

#%% classify_move
def classify_move(board, move):
    r"""
    Determines if the desired move is valid or not, and what type of move/cost it would have.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed

    Returns
    -------
    move_type : class Move
        Type of move to be performed

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import classify_move, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 2] = Piece.current
    >>> move = 2 # (2 right and 1 down)
    >>> move_type = classify_move(board, move)
    >>> print(move_type)
    Move.normal: 1

    """
    # find the size of the board
    xmax = board.shape[0] - 1
    ymax = board.shape[1] - 1
    # find current position
    (x, y) = get_current_position(board)
    # find the traversal for the desired move
    (pos1, pos2, pos3) = get_new_position(x, y, move)
    # check that the final and intermediate positions were all on the board
    valid_moves = np.array([check_board_boundaries(pos[0], pos[1], xmax, ymax) for pos in (pos1, pos2, pos3)])
    if np.any(~valid_moves):
        return Move.off_board
    # get the values for each position
    p1 = board[pos1[0], pos1[1]]
    p2 = board[pos2[0], pos2[1]]
    p3 = board[pos3[0], pos3[1]]
    # check for error conditions
    if p3 in {Piece.start, Piece.current}:
        raise ValueError("The piece should never be able to move to it's current or starting position.")
    # check for blocked conditions
    if p3 in {Piece.rock, Piece.barrier} or p1 == Piece.barrier or p2 == Piece.barrier:
        return Move.blocked
    # remaining moves are valid, determine type
    if p3 == Piece.visited:
        move_type = Move.visited
    elif p3 == Piece.null:
        move_type = Move.normal
    elif p3 == Piece.final:
        move_type = Move.winning
    elif p3 == Piece.transport:
        move_type = Move.transport
    elif p3 == Piece.water:
        move_type = Move.water
    elif p3 == Piece.lava:
        move_type = Move.lava
    else:
        raise ValueError('Unexpected piece type "{}"'.format(p3))
    return move_type

#%% update_board
def update_board(board, move, costs=None):
    r"""
    Updates the new board based on the desired move.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout

    Returns
    -------
    cost : int
        Cost of the specified move type

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015
    #.  Modifies `board` in-place.

    Examples
    --------

    >>> from dstauffman.games.knight import update_board, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 2] = Piece.current
    >>> move = 2 # (2 right and 1 down)
    >>> cost = update_board(board, move)
    >>> print(cost)
    1

    >>> print_board(board)
    . . x . .
    . . . . K

    """
    # check for optional inputs
    if costs is None:
        costs = np.ones(board.shape, dtype=int)
    # initialize output
    cost = LARGE_INT
    # find current position
    (x, y) = get_current_position(board)
    # determine the move type
    move_type = classify_move(board, move)
    # if valid, apply the move
    if move_type > 0:
        # set the current position to visited
        board[x, y] = Piece.visited
        # get the new position
        (_, _, (new_x, new_y)) = get_new_position(x, y, move)
        # set the new position to current
        board[new_x, new_y] = Piece.current
        # determine what the cost was
        cost = costs[new_x, new_y]
        if move_type == Move.winning:
            cost = -cost
    return cost

#%% undo_move
def undo_move(board, last_move, original_board):
    r"""
    Undoes the last move on the board.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    last_move : int
        Last move that was previously performed
    original_board : 2D ndarray of int
        Original board layout before starting solver

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Modifies `board` in-place.

    Examples
    --------
    >>> from dstauffman.games.knight import undo_move, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2, 5), dtype=int)
    >>> board[0, 2] = Piece.visited
    >>> board[1, 4] = Piece.current
    >>> last_move = 2 # (2 right and 1 down)
    >>> original_board = np.zeros((2, 5), dtype=int)
    >>> original_board[0, 2] = Piece.start
    >>> print_board(board)
    . . x . .
    . . . . K

    >>> undo_move(board, last_move, original_board)
    >>> print_board(board)
    . . K . .
    . . . . .

    """
    # find the current position
    (x, y) = get_current_position(board)
    # set the current position back to a null piece
    board[x, y] = original_board[x, y]
    # find the inverse move
    new_move = get_move_inverse(last_move)
    # get the new position
    (_, _, (new_x, new_y)) = get_new_position(x, y, new_move)
    # set the new position to current
    board[new_x, new_y] = Piece.current

#%% get_move_inverse
def get_move_inverse(move):
    r"""
    Gets the inverse move to go back where you were:
        -/+1 <-> -/+3
        -/+2 <-> -/+4

    Parameters
    ----------
    move : int
        Move to be performed

    Returns
    -------
    inv_move : int
        Move that will undo the last one

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------
    >>> from dstauffman.games.knight import get_move_inverse
    >>> print(get_move_inverse(-1))
    -3
    >>> print(get_move_inverse(-3))
    -1
    >>> print(get_move_inverse(2))
    4
    >>> print(get_move_inverse(4))
    2

    """
    inv_move = np.sign(move) * (np.mod(np.abs(move) + 1, 4) + 1)
    return inv_move

#%% check_valid_sequence
def check_valid_sequence(board, moves, costs, print_status=False):
    r"""
    Checks that the list of moves is a valid sequence to go from start to final position.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed
    costs : 2D ndarray of int
        Costs for each square on the board layout
    print_status : bool, optional
        Whether to print the board after each move is made, default is False

    Returns
    -------
    is_valid : bool
        Whether the sequence is valid or not.

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import check_valid_sequence, Piece, board_to_costs
    >>> import numpy as np
    >>> board = np.zeros((3, 5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[2, 4] = Piece.final
    >>> moves = [2, 2]
    >>> costs = board_to_costs(board)
    >>> is_valid = check_valid_sequence(board, moves, costs)
    >>> print(is_valid)
    True

    """
    # initialize output
    is_valid = True
    is_done  = False
    # create internal board for calculations
    temp_board = board.copy()
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    # check that the board has a final goal
    if not np.any(temp_board == Piece.final):
        raise ValueError('The board does not have a finishing location.')
    for (i, this_move) in enumerate(moves):
        # update the board and determine the cost
        cost = update_board(temp_board, this_move, costs)
        # if cost was zero, then the move was invalid
        if cost == COST_DICT['invalid']:
            is_valid = False
            break
        # check for winning conditions (based on negative cost value)
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
def print_sequence(board, moves, costs=None):
    r"""
    Prints the every board position for the given move sequence.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import print_sequence, Piece
    >>> import numpy as np
    >>> board = np.zeros((3, 5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[2, 4] = Piece.final
    >>> moves = [2, 2]
    >>> print_sequence(board, moves)
    Starting position:
    S . . . .
    . . . . .
    . . . . E
    <BLANKLINE>
    After move 1
    x . . . .
    . . K . .
    . . . . E
    <BLANKLINE>
    After move 2
    x . . . .
    . . x . .
    . . . . K

    """
    # check for optional inputs
    if costs is None:
        costs = np.ones(board.shape, dtype=int)
    # create internal board for calculations
    temp_board = board.copy()
    print('Starting position:')
    print_board(temp_board)
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    # loop through move sequence
    for (i,this_move) in enumerate(moves):
        # print header
        print('\nAfter move {}'.format(i+1))
        # update board
        cost = update_board(temp_board, this_move, costs)
        # print new board
        if cost != COST_DICT['invalid']:
            print_board(temp_board)
        else:
            raise ValueError('Bad sequence.')

#%% solve_puzzle
def solve_puzzle(board, costs, moves=None, solve_type='min', original_board=None):
    r"""
    Solves the puzzle with the desired solution type, from 'min', 'max', 'first'.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout
    moves : ndarray of int
        Moves to be performed
    solve_type : str
        Solver type, from {'min', 'max', 'first'}
    original_board : 2D ndarray of int
        Original board layout before doing any moves

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Modifies `moves` inplace.

    Examples
    --------

    >>> from dstauffman.games.knight import solve_puzzle, Piece, board_to_costs
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0,0] = Piece.start
    >>> board[0,4] = Piece.final
    >>> costs = board_to_costs(board)
    >>> moves = []
    >>> solve_type = 'first'
    >>> solve_puzzle(board, costs, moves, solve_type)
    Solution found!

    >>> print(moves)
    [2, -2]

    """
    # check for correct inputs
    if moves is None:
        raise ValueError('`moves` needs to be initialized to a mutable list.')
    # save original board for use in undoing moves
    if original_board is None:
        original_board = board.copy()
    # check for initial start conditions
    if len(moves) == 0:
        # set the current position to the start
        board[board == Piece.start] = Piece.current
    # try all the next possible moves
    for this_move in MOVES:
        # make the move
        cost = update_board(board, this_move, costs)
        # if the move was invalid then go to the next one, if legal check if done, otherwise call recursively
        if cost == COST_DICT['invalid']:
            # if the move was invalid, determine if it's because of a previously visited square
            if solve_type == 'first':
                continue
            else: # TODO: code this
                raise ValueError('Not yet implemented.')
        elif cost < 0:
            # solution is found
            if solve_type == 'first':
                moves.append(this_move)
                print('Solution found!')
                return
            else: # TODO: Work on this
                raise ValueError('Not yet implemented.')
                # this move is not better, so don't accept it
                undo_move(board, this_move, original_board)
                continue
        else:
            if solve_type == 'first': # or this move is better
                moves.append(this_move)
                temp = len(moves)
                solve_puzzle(board, costs, moves=moves, solve_type=solve_type, \
                    original_board=original_board)
                if len(moves) < temp:
                    continue
                else:
                    return
            else:
                # this move is not better, so don't accept it
                undo_move(board, this_move)
    # all possible moves didn't work, so this leg is dead, back out last move and exit
    if len(moves) > 0:
        last_move = moves.pop()
        undo_move(board, last_move, original_board)
    else:
        print('No solution found.')
    return

#%% Unit test
def main():
    unittest.main(module='dstauffman.games.test_knight', exit=False)
    doctest.testmod(verbose=False)

#%% Script
if __name__ == '__main__':
    # flags for what steps to do
    do_steps = {-1, 0, 1, 2} #do_steps = {-1, 0, 1, 2, 3, 4, 5}

    # Step -1 (Run unit tests)
    if -1 in do_steps:
        main()

    # convert board to numeric representation for efficiency
    board1 = char_board_to_nums(BOARD1)
    board2 = char_board_to_nums(BOARD2)
    board3 = np.zeros((5,5), dtype=int)
    board3[2,2] = Piece.start
    board3[3,4] = Piece.final
    board3 = np.zeros((2,5), dtype=int)
    board3[0,0] = Piece.start
    board3[0,4] = Piece.final
    costs1 = board_to_costs(board1)
    costs2 = board_to_costs(board2)
    costs3 = board_to_costs(board3)

    # Step 0
    if 0 in do_steps:
        print('Step 0: print the boards.')
        print_board(board1)
        print_board(board2)
        #print_board(board3)

    # Step 1
    if 1 in do_steps:
        print('\nStep 1: see if the sequence is valid.')
        moves1 = [2, 2]
        is_valid = check_valid_sequence(board1, moves1, costs1, print_status=True)
        if is_valid:
            print_sequence(board1, moves1)

    # Step 2
    if 2 in do_steps:
        print('\nStep 2: solve the first board for any length solution.')
        moves2 = []
        soln_board = board1.copy()
        #best_cost = LARGE_INT*np.ones(soln_board.shape, dtype=int)
        solve_puzzle(soln_board, costs1, moves=moves2, solve_type='first')
        print(moves2)
        is_valid = check_valid_sequence(board1, moves2, costs1, print_status=True)
        if is_valid:
            print_sequence(board1, moves2)

    # Step 3
    if 3 in do_steps:
        print('\nStep 3: solve the first board for the minimum length solution.')
        moves3 = []
        soln_board = board1.copy()
        #best_cost = LARGE_INT*np.ones(soln_board.shape, dtype=int)
        solve_puzzle(soln_board, costs1, moves=moves3, solve_type='min')
        print(moves3)
        is_valid = check_valid_sequence(board1, moves3, costs1, print_status=True)
        if is_valid:
            print_sequence(board1, moves3)

    # Step 4
    if 4 in do_steps:
        pass

    # Step 5
    if 5 in do_steps:
        pass
