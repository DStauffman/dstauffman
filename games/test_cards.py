# -*- coding: utf-8 -*-
r"""
Test file for the `cards` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2016.
"""

#%% Imports
import unittest
import dstauffman.games.cards as cards

#%% Local aliases
R = cards.Rank
S = cards.Suit
H = cards.Hand
C = cards.Card

hands = [\
    H([C(R.ACE, S.SPADES), C(R.KING, S.SPADES), C(R.QUEEN, S.SPADES), C(R.JACK, S.SPADES), C(R.TEN, S.SPADES)]),\
    H([C(R.FIVE, S.SPADES), C(R.THREE, S.SPADES), C(R.FOUR, S.SPADES), C(R.TWO, S.SPADES), C(R.ACE, S.SPADES)]),\
    #H([C(R.ACE, S.SPADES), C(R.KING, S.SPADES), C(R.QUEEN, S.HEARTS), C(R.JACK, S.DIAMONDS), C(R.TEN, S.CLUBS)]),\
    #H([C(R.ACE, S.SPADES), C(R.KING, S.SPADES), C(R.QUEEN, S.HEARTS), C(R.JACK, S.DIAMONDS), C(R.TEN, S.CLUBS)]),\
    ]

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
            scores.append(hand.score_hand())
        #print(scores)
        # TODO: finish writing test

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
