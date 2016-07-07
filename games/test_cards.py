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

#%% Card
class Test_card(unittest.TestCase):
    r"""
    Tests the Card class with the following cases:
        TBD
    """
    def setUp(self):
        self.ranks = [R.TWO, R.ACE, R.TWO, R.ACE, R.FIVE, R.FIVE]
        self.suits = [S.CLUBS, S.CLUBS, S.SPADES, S.SPADES, S.HEARTS, S.DIAMONDS]
        self.num   = len(self.ranks)
        self.cards = [C(self.ranks[i], self.suits[i]) for i in range(self.num)]

    def test_card_number(self):
        self.assertEqual(self.cards[0]._card, 0)
        self.assertEqual(self.cards[1]._card, 12)
        self.assertEqual(self.cards[2]._card, 39)
        self.assertEqual(self.cards[3]._card, 51)

    def test_rank(self):
        for i in range(self.num):
            self.assertEqual(self.cards[i].get_rank(), self.ranks[i])

    def test_suit(self):
        for i in range(self.num):
            self.assertEqual(self.cards[i].get_suit(), self.suits[i])

    def test_equalities(self):
        # equal
        self.assertEqual(self.cards[0], self.cards[0])
        self.assertEqual(self.cards[0], self.cards[2])
        self.assertEqual(self.cards[1], self.cards[3])
        self.assertEqual(self.cards[4], self.cards[5])
        # not equal
        self.assertNotEqual(self.cards[0], self.cards[1])
        # less than
        self.assertLess(self.cards[0], self.cards[1])
        self.assertLess(self.cards[0], self.cards[3])
        self.assertLess(self.cards[4], self.cards[3])
        self.assertFalse(self.cards[1] < self.cards[0])
        # greater than
        self.assertGreater(self.cards[1], self.cards[0])
        self.assertFalse(self.cards[0] > self.cards[1])
        # less than or equal to
        self.assertLessEqual(self.cards[0], self.cards[0])
        self.assertLessEqual(self.cards[4], self.cards[5])
        self.assertFalse(self.cards[1] <= self.cards[0])
        # greater than or equal to
        self.assertGreaterEqual(self.cards[0], self.cards[0])
        self.assertGreaterEqual(self.cards[4], self.cards[5])
        self.assertFalse(self.cards[0] >= self.cards[1])

#%% Deck
class Test_Deck(unittest.TestCase):
    r"""
    Tests the Deck class with the following cases:
        TBD
    """
    def setUp(self):
        self.deck = cards.Deck()
        self.sorted_deck = 'A♠, K♠, Q♠, J♠, 10♠, 9♠, 8♠, 7♠, 6♠, 5♠, 4♠, 3♠, 2♠, ' + \
            'A♥, K♥, Q♥, J♥, 10♥, 9♥, 8♥, 7♥, 6♥, 5♥, 4♥, 3♥, 2♥, A♦, K♦, Q♦, J♦, 10♦, ' + \
            '9♦, 8♦, 7♦, 6♦, 5♦, 4♦, 3♦, 2♦, A♣, K♣, Q♣, J♣, 10♣, 9♣, 8♣, 7♣, 6♣, 5♣, 4♣, 3♣, 2♣'

    def test_creation(self):
        self.assertEqual(str(self.deck), self.sorted_deck)

    def test_shuffle(self):
        self.deck.shuffle()
        self.assertNotEqual(str(self.deck), self.sorted_deck)

    #def test_sort(self):
    #    self.deck.shuffle()
    #    self.assertNotEqual(str(self.deck), self.sorted_deck)
    #    self.deck.sort()
    #    self.assertEqual(str(self.deck), self.sorted_deck)

    def test_reset(self):
        self.assertEqual(str(self.deck), self.sorted_deck)
        self.deck.shuffle()
        self.assertNotEqual(str(self.deck), self.sorted_deck)
        self.deck.reset()
        self.assertEqual(str(self.deck), self.sorted_deck)

#%% Hand
class Test_Hand(unittest.TestCase):
    r"""
    Tests the Hand Class with the following cases:
        TBD
    """
    def setUp(self):
        self.hand = cards.Hand()
        self.card1 = cards.Card(R.ACE, S.SPADES)
        self.card2 = cards.Card(R.FOUR, S.HEARTS)

    def test_creation(self):
        self.assertEqual(len(self.hand._cards), 0)

    def test_num_cards(self):
        self.assertEqual(self.hand.num_cards(), 0)
        self.hand.add_card(self.card1)
        self.assertEqual(len(self.hand._cards), 1)
        self.assertEqual(self.hand.num_cards(), 1)

    def test_add_card(self):
        self.hand.add_card(self.card1)
        self.assertEqual(self.hand.num_cards(), 1)
        self.assertTrue(self.hand._cards[0] is self.card1)

    def test_play_card(self):
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card2)
        self.hand.add_card(self.card2)
        self.assertEqual(self.hand.num_cards(), 3)
        self.assertTrue(self.hand.play_card() is self.card2)
        self.assertEqual(self.hand.num_cards(), 2)
        self.assertTrue(self.hand.play_card() is self.card2)
        self.assertEqual(self.hand.num_cards(), 1)
        self.assertTrue(self.hand.play_card() is self.card1)
        self.assertEqual(self.hand.num_cards(), 0)
        with self.assertRaises(IndexError):
            self.hand.play_card()

    def test_remove_card(self):
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card2)
        self.assertEqual(self.hand.num_cards(), 2)
        self.assertTrue(self.hand._cards[-1] is self.card2)
        self.hand.remove_card(self.card2)
        self.assertEqual(self.hand.num_cards(), 1)
        self.assertTrue(self.hand._cards[-1] is self.card1)
        self.hand.remove_card(self.card1)
        self.assertEqual(self.hand.num_cards(), 0)
        with self.assertRaises(ValueError):
            self.hand.remove_card(self.card1)

    def test_get_ranks(self):
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card2)
        ranks = self.hand.get_ranks()
        self.assertEqual(ranks[0], R.ACE)
        self.assertEqual(ranks[1], R.FOUR)

    def test_get_suits(self):
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card2)
        suits = self.hand.get_suits()
        self.assertEqual(suits[0], S.SPADES)
        self.assertEqual(suits[1], S.HEARTS)

    def test_shuffle(self):
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card1)
        self.hand.add_card(self.card2)
        was_changed = False
        for i in range(5):
            self.hand.shuffle()
            if self.hand._cards[-1] != self.card2:
                was_changed = True
                break
        self.assertTrue(was_changed)

    # Score metrics
    # 9. five of a kind (not possible without wilds)
    # 8. straight flush (includes royal flush)
    # 7. four of a kind
    # 6. full house
    # 5. flush
    # 4. straight
    # 3. three of a kind
    # 2. two pair
    # 1. pair
    # 0. high card
    def test_score_five_of_kind(self):
        hand1 = H([C(R.ACE, S.SPADES), C(R.ACE, S.DIAMONDS), C(R.ACE, S.CLUBS), C(R.ACE, S.HEARTS), C(R.ACE, S.SPADES)])
        hand2 = H([C(R.TEN, S.SPADES), C(R.TEN, S.DIAMONDS), C(R.TEN, S.CLUBS), C(R.TEN, S.HEARTS), C(R.TEN, S.SPADES)])
        score1 = hand1.score_hand()
        score2 = hand2.score_hand()
        #self.assertGreater(score1, 9)
        #self.assertGreater(score2, 9)
        #self.assertLess(score1, 10)
        #self.assertLess(score2, 10)
        #self.assertGreater(score1, score2)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
