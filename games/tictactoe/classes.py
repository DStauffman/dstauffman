# -*- coding: utf-8 -*-
r"""
Classes module file for the "tictactoe" game.  It defines the classes used by the rest of the game.

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
from dstauffman.games.tictactoe.constants import PLAYER

#%% Options
class Options(Frozen):
    r"""
    Class that keeps track of the options for the game.
    """
    # Gameplay default options
    load_previous_game = 'No' # from ['Yes','No','Ask']
    plot_best_moves    = False
    plot_move_power    = False
    o_is_computer      = False
    x_is_computer      = False

    def __init__(self, **kwargs):
        r"""Creates options instance with ability to override defaults."""
        # override attributes
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                raise ValueError('Unexpected attribute: {}'.format(key))

#%% State
class State(Frozen):
    r"""
    Class that keeps track of the GUI state.
    """
    def __init__(self):
        self.board     = PLAYER['none'] * np.ones((3, 3), dtype=int)
        self.cur_move  = Counter(0)
        self.cur_game  = Counter(0)
        self.game_hist = [GameStats(number=self.cur_game, first_move=PLAYER['o'])]

#%% Moves
class Move(Frozen):
    r"""
    Class that keeps track of each individual move.
    """
    def __init__(self, row, column, power=None):
        self.row       = row
        self.column    = column
        self.power     = power

    def __eq__(self, other):
        r"""Equality is based on row and column."""
        return (self.row == other.row and self.column == other.column)

    def __ne__(self, other):
        r"""Inequality is based on row and column."""
        return not self.__eq__(other)

    def __lt__(self, other):
        r"""Sorts by power, then row, then column."""
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
        return False

    def __hash__(self):
        r"""Hash uses str instead of repr, and thus power does not distinguish values."""
        return hash(self.__str__())

    def __str__(self):
        r"""String returns values except for power."""
        return 'row: {}, col: {}'.format(self.row, self.column)

    def __repr__(self):
        r"""Repr returns all values, including power."""
        return '<' + self.__str__() + ', pwr: {}'.format(self.power) + '>'

#%% GameStats
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
    unittest.main(module='dstauffman.games.tictactoe.tests.test_classes', exit=False)
    doctest.testmod(verbose=False)
