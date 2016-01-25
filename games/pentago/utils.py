# -*- coding: utf-8 -*-
r"""
Utils module file for the "pentago" game.  It defines the generic utility functions.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import logging
import numpy as np
import os
import unittest
from dstauffman import modd
from dstauffman import get_root_dir as dcs_root_dir
from dstauffman.games.pentago.classes   import Move
from dstauffman.games.pentago.constants import INT_TOKEN, PLAYER, ONE_OFF, SIZES, WIN, _rotate_board

#%% get_root_dir
def get_root_dir():
    r"""
    Gets the full path to the root directory of the pentago game.

    Returns
    -------
    folder : str
        Path to the pentago root folder

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------

    >>> from dstauffman.games.pentago import get_root_dir
    >>> folder = get_root_dir()

    """
    folder = os.path.join(dcs_root_dir(), 'games', 'pentago')
    return folder

#%% rotate_board
rotate_board = _rotate_board

#%% calc_cur_move
def calc_cur_move(cur_move, cur_game):
    r"""
    Calculates whose move it is based on the turn and game number.

    Parameters
    ----------
    cur_move : int
        Current move number
    cur_game : int
        Current game number

    Returns
    -------
    move : int
        Current move, from {1=white, -1=black}

    Examples
    --------
    >>> from dstauffman.games.pentago import calc_cur_move
    >>> move = calc_cur_move(0, 0)
    >>> print(move)
    1

    """
    if np.mod(cur_move + cur_game, 2) == 0:
        move = PLAYER['white']
    else:
        move = PLAYER['black']
    return move

#%% check_for_win
def check_for_win(board):
    r"""
    Checks for a win.
    """
    # find white and black wins
    white = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['white'], axis=1) * WIN, axis=0) == 5)[0]
    black = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['black'], axis=1) * WIN, axis=0) == 5)[0]

    # determine winner
    if len(white) == 0:
        if len(black) == 0:
            winner = PLAYER['none']
        else:
            winner = PLAYER['black']
    else:
        if len(black) == 0:
            winner = PLAYER['white']
        else:
            winner = PLAYER['draw']

    # check for a full game board after determining no other win was found
    if winner == PLAYER['none'] and not np.any(board == PLAYER['none']):
        winner = PLAYER['draw']

    # find winning pieces on the board
    if winner == PLAYER['none']:
        win_mask = np.zeros((SIZES['board'],SIZES['board']), dtype=bool)
    else:
        logging.debug('Win detected.  Winner is {}.'.format(list(PLAYER.keys())[list(PLAYER.values()).index(winner)]))
        win_mask = np.reshape(np.sum(WIN[:, white], axis=1) + np.sum(WIN[:, black], axis=1), (SIZES['board'], SIZES['board'])) != 0

    return (winner, win_mask)

#%% find_moves
def find_moves(board):
    r"""
    Finds the best current move.

    Notes
    -----
    #.  Currently this function is only trying to find a win in one move situation.

    Examples
    --------

    >>> from dstauffman.games.pentago import find_moves
    >>> import numpy as np
    >>> board = np.reshape(np.hstack((np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]), np.zeros(24, dtype=int))), (6, 6))
    >>> (white_moves, black_moves) = find_moves(board)
    >>> print(white_moves[0])
    row: 0, col: 1, quad: 1, dir: 1

    >>> print(black_moves)
    []

    """
    #%% get_move_from_one_off
    def get_move_from_one_off(big_board, ix, ONE_OFF):
        r"""
        Turns the given index into a Move instance.
        """
        # preallocate x & y to NaNs in case the winning move is just a rotation
        row = INT_TOKEN * np.ones(len(ix), dtype=int)
        column = row.copy()

        # find missing piece
        pos_ix = np.logical_and(np.logical_xor(np.abs(big_board), ONE_OFF[:,ix]), ONE_OFF[:,ix])

        assert np.all(np.sum(pos_ix, axis=0) <= 1), 'Only exactly one or fewer places should be found.'

        # pull out element number from 0 to 35
        (one_off_row, one_off_col) = np.nonzero(pos_ix)
        # convert to row and column
        row[one_off_col]    = one_off_row // SIZES['board']
        column[one_off_col] = np.mod(one_off_row, SIZES['board'])

        # get quadrant and rotation number
        # based on order that ONE_OFF was built, so permutations of first quads 1,2,3,4, second left,right;
        num = np.ceil(ix/WIN.shape[1]).astype(int)

        # pull out quadrant number
        quadrant = modd(num, 4)

        # pull out rotation direction
        direction = -1 * np.ones(len(ix), dtype=int)
        direction[num < 5] = 1

        # convert to a move class
        move = set()
        for i in range(len(ix)):
            move.add(Move(row[i], column[i], quadrant[i], direction[i], power=5))
        return move
    # expand the board to a linear 2D matrix
    big_board = np.expand_dims(board.ravel(), axis=1)

    # check for wins that shouldn't exist
    test = big_board * WIN
    score = np.sum(test, axis=0)
    if np.any(np.abs(score) >= 5):
        raise ValueError('Board should not already be in a winning position.')

    # cross reference two matrices with element-wise multiplication
    test = big_board * ONE_OFF

    # find score
    score = np.sum(test, axis=0)

    # find white and black rotate to win moves
    rot_white = np.nonzero(score >=  5)[0]
    rot_black = np.nonzero(score <= -5)[0]

    # find white and black one off potentials
    white = np.nonzero((score >=  4) & (score <  5))[0]
    black = np.nonzero((score <= -4) & (score > -5))[0]

    # see if the remaining piece is an open square
    if len(white) > 0:
        pos_white = ONE_OFF[:, white]
        needed    = np.logical_xor(pos_white, big_board)
        free      = np.logical_and(needed, np.logical_not(big_board))
        ix_white  = white[np.any(free, axis=0)]
    else:
        ix_white  = np.array([], dtype=int)
    if len(black) > 0:
        pos_black = ONE_OFF[:, black]
        needed    = np.logical_xor(pos_black, big_board)
        free      = np.logical_and(needed, np.logical_not(big_board))
        ix_black  = black[np.any(free, axis=0)]
    else:
        ix_black  = np.array([], dtype=int)

    # find winning moves
    # placement winning moves
    white_set = get_move_from_one_off(big_board, ix_white, ONE_OFF)
    black_set = get_move_from_one_off(big_board, ix_black, ONE_OFF)
    # rotation only winning moves
    white_rotations = get_move_from_one_off(big_board, rot_white, ONE_OFF)
    black_rotations = get_move_from_one_off(big_board, rot_black, ONE_OFF)

    # fill in all available row and columns positions for the rotate to win moves
    empty = np.nonzero(big_board == PLAYER['none'])[0]
    for ix in empty:
        this_row = ix // SIZES['board']
        this_col = np.mod(ix, SIZES['board'])
        for this_rot in white_rotations:
            this_move = Move(this_row, this_col, this_rot.quadrant, this_rot.direction, power=5)
            white_set.add(this_move)
        for this_rot in black_rotations:
            this_move = Move(this_row, this_col, this_rot.quadrant, this_rot.direction, power=5)
            black_set.add(this_move)

    # check for ties and set their power to -1
    ties = white_set & black_set
    for this_move in ties:
        white_set.remove(this_move)
        black_set.remove(this_move)
        this_move.power = -1
        white_set.add(this_move)
        black_set.add(this_move)

    # convert to list, sort by power, such that ties go at the end
    white_moves = sorted(list(white_set))
    black_moves = sorted(list(black_set))

    return (white_moves, black_moves)

#%% create_board_from_moves
def create_board_from_moves(moves, first_player):
    r"""
    Recreates a board from a move history.
    """
    # make sure the first player is valid
    assert first_player == PLAYER['white'] or first_player == PLAYER['black']
    # create the initial board
    board = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
    # alias this player
    this_player = first_player
    # loop through the move history
    for this_move in moves:
        # check that square is empty
        assert board[this_move.row, this_move.column] == PLAYER['none'], 'Invalid move encountered.'
        # place the piece
        board[this_move.row, this_move.column] = this_player
        # rotate the board
        _rotate_board(board, this_move.quadrant, this_move.direction, inplace=True)
        # update the next player to move
        this_player = PLAYER['white'] if this_player == PLAYER['black'] else PLAYER['black']
    return board

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.pentago.tests.test_utils', exit=False)
    doctest.testmod(verbose=False)
