# -*- coding: utf-8 -*-
r"""
Classes module file for the "pentago" game.  It defines the classes used by the rest of the game.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import numpy as np
import pickle
import unittest
from dstauffman import Frozen, Counter
from dstauffman.games.pentago.constants import PLAYER, SIZES

#%% Classes - State
class State(Frozen):
    r"""
    Class that keeps track of the GUI state.
    """
    def __init__(self):
        self.board       = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
        self.cur_move    = Counter(0)
        self.cur_game    = Counter(0)
        self.move_status = {'ok': False, 'pos': None, 'patch_object': None}
        self.game_hist   = [GameStats(number=self.cur_game, first_move=PLAYER['white'])]

#%% Classes - Move
class Move(Frozen):
    r"""
    Class that keeps track of each individual move.
    """
    def __init__(self, row, column, quadrant, direction, power=None):
        self.row       = row
        self.column    = column
        self.quadrant  = quadrant
        self.direction = direction
        self.power     = power

    def __eq__(self, other):
        r"""Equality is based on row, column, quadrant and direction, but not on power."""
        return (self.row == other.row and self.column == other.column and self.quadrant == other.quadrant \
            and self.direction == other.direction)

    def __ne__(self, other):
        r"""Inequality is based on row, column, quadrant and direction, but not on power."""
        return not self.__eq__(other)

    def __lt__(self, other):
        r"""Sorts by power, then row, then column, then quadrant, then direction."""
        if self.power is None:
            if other.power is not None:
                return True
        else:
            if other.power is None:
                return False
            else:
                if self.power < other.power:
                    return True
                elif self.power > other.power:
                    return False
        if self.row < other.row:
            return True
        elif self.row > other.row:
            return False
        if self.column < other.column:
            return True
        elif self.column > other.column:
            return False
        if self.quadrant < other.quadrant:
            return True
        elif self.quadrant > other.quadrant:
            return False
        if self.direction < other.direction:
            return True
        elif self.direction < other.direction:
            return False
        return False # make True if __le__

    def __hash__(self):
        r"""Hash uses str instead of repr, and thus power does not distinguish values."""
        return hash(self.__str__())

    def __str__(self):
        r"""String returns values except for power."""
        return 'row: {}, col: {}, quad: {}, dir: {}'.format(self.row, self.column, self.quadrant, self.direction)

    def __repr__(self):
        r"""Repr returns all values, including power."""
        return '<' + self.__str__() + ', pwr: {}'.format(self.power) + '>'

    @property
    def rot_key(self):
        r"""Gets the key for the rotation that this move represents."""
        return '{}{}'.format(self.quadrant, 'L' if self.direction == -1 else 'R')

#%% Classes - GameStats
class GameStats(Frozen):
    r"""
    Class that keeps track of all the moves in a game.
    """
    def __init__(self, number, first_move, winner=PLAYER['none'], move_list=None):
        self.number     = number
        self.first_move = first_move
        self.winner     = winner
        if move_list is None:
            self.move_list = []
        else:
            self.move_list = move_list

    def add_move(self, move):
        r"""Adds the given move to the game move history."""
        assert isinstance(move, Move), 'The specified move must be an instance of class Move.'
        self.move_list.append(move)

    def remove_moves(self, cur_move=None):
        r"""Removes the moves from the current move number to the end of the list."""
        if cur_move is None:
            self.move_list.pop()
        else:
            del(self.move_list[cur_move:])

    @property
    def num_moves(self):
        r"""Calculates the number of moves in a move list."""
        return len(self.move_list)

    @staticmethod
    def get_results(game_hist):
        r"""Pulls the results out of a list of game histories."""
        return np.array([x.winner for x in game_hist])

    @staticmethod
    def save(filename, game_hist):
        r"""Saves a list of GameStats objects to disk."""
        with open(filename, 'wb') as file:
            pickle.dump(game_hist, file)

    @staticmethod
    def load(filename):
        r"""Loads a list of GameStats objects to disk."""
        with open(filename, 'rb') as file:
            game_hist = pickle.load(file)
        return game_hist

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.pentago.tests.test_classes', exit=False)
    doctest.testmod(verbose=False)
