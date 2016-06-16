# -*- coding: utf-8 -*-
r"""
Test file for the `tictactoe.classes` module of the dstauffman code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import numpy as np
import os
import unittest
import dstauffman.games.tictactoe as ttt

#%% Aliases
o = ttt.PLAYER['o']
x = ttt.PLAYER['x']
n = ttt.PLAYER['none']

#%% Options
class Test_Options(unittest.TestCase):
    r"""
    Tests the Options class with the following cases:
        TBD
    """
    def setUp(self):
        self.options_dict = {'load_previous_game': 'No', 'plot_best_moves': True, \
            'plot_move_power': True, 'o_is_computer': True, 'x_is_computer': True}

    def test_nominal(self):
        opts = ttt.Options()
        self.assertTrue(isinstance(opts, ttt.Options))

    def test_inputs(self):
        opts = ttt.Options(**self.options_dict)
        self.assertTrue(isinstance(opts, ttt.Options))
        for this_key in self.options_dict:
            self.assertEqual(getattr(opts, this_key), self.options_dict[this_key])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            ttt.Options(bad_input_value='Whatever')

#%% State
class Test_State(unittest.TestCase):
    r"""
    Tests the State class with the following cases:
        Nominal
    """
    def test_nominal(self):
        self.state = ttt.State()
        self.assertTrue(isinstance(self.state, ttt.State))

#%% Move
class Test_Move(unittest.TestCase):
    r"""
    Tests the Move class with the following cases:
        create without power
        create with power
        equality
        inequality
        less than
        sorting
        hashing
        create set
        string
        repr
    """
    def setUp(self):
        self.move1 = ttt.Move(0, 1)
        self.move2 = ttt.Move(1, 3, 4)
        self.move3 = ttt.Move(1, 3, 2)

    def test_create1(self):
        self.assertEqual(self.move1.row, 0)
        self.assertEqual(self.move1.column, 1)
        self.assertTrue(self.move1.power is None)

    def test_create2(self):
        self.assertEqual(self.move2.row, 1)
        self.assertEqual(self.move2.column, 3)
        self.assertEqual(self.move2.power, 4)

    def test_equality(self):
        self.assertTrue(self.move2 == self.move3)

    def test_inequality(self):
        self.assertTrue(self.move1 != self.move2)

    def test_lt(self):
        self.assertTrue(self.move3 < self.move2)
        self.assertFalse(self.move1 < self.move1)
        self.assertTrue(self.move2 < ttt.Move(1, 3, 5))
        self.assertFalse(self.move2 < ttt.Move(1, 3, 4))
        self.assertTrue(self.move2 < ttt.Move(2, 3, 4))
        self.assertFalse(self.move2 < ttt.Move(0, 3, 4))
        self.assertTrue(self.move2 < ttt.Move(1, 4, 4))
        self.assertFalse(self.move2 < ttt.Move(1, 2, 4))

    def test_sort(self):
        ix = [self.move3, self.move1, self.move2]
        ix.sort()
        self.assertEqual(ix[0], self.move1)
        self.assertEqual(ix[1], self.move2)
        self.assertEqual(ix[2], self.move3)

    def test_hash(self):
        self.assertTrue(hash(self.move1))

    def test_set(self):
        self.assertTrue(len(set([self.move1, self.move2])) == 2)
        self.assertTrue(len(set([self.move2, self.move3])) == 1)

    def test_str(self):
        out = str(self.move1)
        self.assertEqual(out, 'row: 0, col: 1')

    def test_repr(self):
        rep = repr(self.move1)
        self.assertEqual(rep, '<row: 0, col: 1, pwr: None>')

#%% GameStats
class Test_GameStats(unittest.TestCase):
    r"""
    Tests the GameStats class with the following cases:
        TBD
    """
    def setUp(self):
        self.move1      = ttt.Move(0, 0)
        self.gamestats1 = ttt.GameStats(0, o, n)
        self.gamestats2 = ttt.GameStats(0, o, n, [self.move1, self.move1, self.move1])
        self.game_hist  = [self.gamestats1, self.gamestats2]

    def test_add_move(self):
        self.assertTrue(len(self.gamestats1.move_list) == 0)
        self.gamestats1.add_move(self.move1)
        self.assertTrue(len(self.gamestats1.move_list) == 1)

    def test_bad_add_move(self):
        with self.assertRaises(AssertionError):
            self.gamestats1.add_move(1)

    def test_remove_moves1(self):
        self.assertEqual(self.gamestats2.num_moves, 3)
        self.gamestats2.remove_moves()
        self.assertEqual(self.gamestats2.num_moves, 2)

    def test_remove_moves2(self):
        self.assertEqual(self.gamestats2.num_moves, 3)
        self.gamestats2.remove_moves(1)
        self.assertEqual(self.gamestats2.num_moves, 1)

    def test_remove_moves3(self):
        with self.assertRaises(IndexError):
            self.gamestats1.remove_moves()

    def test_num_moves(self):
        self.assertEqual(self.gamestats2.num_moves, 3)

    def test_get_results(self):
        results = ttt.GameStats.get_results(self.game_hist)
        np.testing.assert_array_equal(results, [n, n])

    def test_save_and_load(self):
        filename = os.path.join(ttt.get_root_dir(), 'tests', 'temp_save.pkl')
        ttt.GameStats.save(filename, self.game_hist)
        self.assertTrue(os.path.isfile(filename))
        game_hist = ttt.GameStats.load(filename)
        for i in range(len(self.game_hist)):
            for j in range(self.game_hist[i].num_moves):
                self.assertEqual(self.game_hist[i].move_list[j], game_hist[i].move_list[j])

    def tearDown(self):
        filename = os.path.join(ttt.get_root_dir(), 'tests', 'temp_save.pkl')
        if os.path.isfile(filename):
            os.remove(filename)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
