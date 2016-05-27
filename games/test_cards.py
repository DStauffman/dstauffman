# -*- coding: utf-8 -*-
r"""
Test file for the `cards` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2016.
"""

#%% Imports
import numpy as np
import unittest

import dstauffman.games.cards as cards
R = cards.Rank
S = cards.Suit

hands1 = [\
    [(R.ACE, S.SPADES), (R.KING, S.SPADES), (R.QUEEN, S.SPADES), (R.JACK, S.SPADES), (R.TEN, S.SPADES)],\
    [(R.FIVE, S.SPADES), (R.THREE, S.SPADES), (R.FOUR, S.SPADES), (R.TWO, S.SPADES), (R.ACE, S.SPADES)],\
    #[(R.ACE, S.SPADES), (R.KING, S.SPADES), (R.QUEEN, S.HEARTS), (R.JACK, S.DIAMONDS), (R.TEN, S.CLUBS)],\
    #[(R.ACE, S.SPADES), (R.KING, S.SPADES), (R.QUEEN, S.HEARTS), (R.JACK, S.DIAMONDS), (R.TEN, S.CLUBS)],\
    ]
hands = [np.array([cards.ranksuit2card(x, y) for (x,y) in hand1], dtype=int) for hand1 in hands1]

#%% eval_hand
class Test_eval_hand(unittest.TestCase):
    r"""
    Tests the eval_hand function with the following cases:
        TBD
    """
    def setUp(self):
        self.hands = hands

    def test_scores(self):
        scores = []
        for hand in hands:
            scores.append(cards.eval_hand(hand))
        #print(scores)
        # TODO: finish writing test

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
