# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.constants` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import unittest
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']

#%% COLOR
class Test_COLOR(unittest.TestCase):
    r"""
    Tests the COLOR class with the following cases:
        Test all keys
    """
    def setUp(self):
        self.expected_keys = set(['board', 'win', 'o', 'x', 'edge', 'win_o', 'win_x', 'win_ox'])

    def test_expected(self):
        keys = set(ttt.COLOR.keys())
        self.assertEqual(keys, self.expected_keys)

#%% PLAYER
class Test_PLAYER(unittest.TestCase):
    r"""
    Tests the PLAYER class with the following cases:
        Test all keys
    """
    def setUp(self):
        self.expected_keys = set(['o', 'x', 'none', 'draw'])

    def test_expected(self):
        keys = set(ttt.PLAYER.keys())
        self.assertEqual(keys, self.expected_keys)

#%% SCORING
class Test_SCORING(unittest.TestCase):
    r"""
    Tests the SCORING class with the following cases:
        Test all keys
    """
    def setUp(self):
        self.expected_keys = set(['win', 'block_win', 'win_in_two', 'block_in_two', 'normal_line', 'block_line'])

    def test_expected(self):
        keys = set(ttt.SCORING.keys())
        self.assertEqual(keys, self.expected_keys)

#%% SIZES
class Test_SIZES(unittest.TestCase):
    r"""
    Tests the Color class with the following cases:
        Test all keys
    """
    def setUp(self):
        self.expected_keys = set(['piece', 'square', 'board'])

    def test_expected(self):
        keys = set(ttt.SIZES.keys())
        self.assertEqual(keys, self.expected_keys)

#%% WIN
class Test_WIN(unittest.TestCase):
    r"""
    Tests the WIN class with the following cases:
        Size test
    """
    def test_size(self):
        self.assertEqual(ttt.WIN.shape, (9,8))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
