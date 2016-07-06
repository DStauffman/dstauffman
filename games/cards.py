# -*- coding: utf-8 -*-
r"""
Cards module file for the "dstauffman" library.  It contains classes and routines specific to card
games such as poker.

Notes
-----
#.  Written by David C. Stauffer in May 2016.
"""

#%% Imports
from collections import Counter
import doctest
from enum import unique, IntEnum
import unittest

#%% Enums  - Suit
@unique
class Suit(IntEnum):
    r"""
    Enumerator definitions for the possible card suits.
    """
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3

#%% Enums - Rank
@unique
class Rank(IntEnum):
    r"""
    Enumerator definitions for the possible hand ranks.
    """
    TWO   = 0
    THREE = 1
    FOUR  = 2
    FIVE  = 3
    SIX   = 4
    SEVEN = 5
    EIGHT = 6
    NINE  = 7
    TEN   = 8
    JACK  = 9
    QUEEN = 10
    KING  = 11
    ACE   = 12

#%% Constants
NUM_SUITS = 4
NUM_RANKS = 13
suit_symbol                = {}
suit_symbol[Suit.CLUBS]    = '\u2663'
suit_symbol[Suit.DIAMONDS] = '\u2666'
suit_symbol[Suit.HEARTS]   = '\u2665'
suit_symbol[Suit.SPADES]   = '\u2660'

rank_symbol             = {}
rank_symbol[Rank.TWO]   = '2'
rank_symbol[Rank.THREE] = '3'
rank_symbol[Rank.FOUR]  = '4'
rank_symbol[Rank.FIVE]  = '5'
rank_symbol[Rank.SIX]   = '6'
rank_symbol[Rank.SEVEN] = '7'
rank_symbol[Rank.EIGHT] = '8'
rank_symbol[Rank.NINE]  = '9'
rank_symbol[Rank.TEN]   = '10'
rank_symbol[Rank.JACK]  = 'J'
rank_symbol[Rank.QUEEN] = 'Q'
rank_symbol[Rank.KING]  = 'K'
rank_symbol[Rank.ACE]   = 'A'

STRAIGHTS = [set(range(i, i+5)) for i in range(9)]

#%% Classes - Card
class Card(object):
    r"""
    A Single card.
    """
    def __init__(self, rank, suit):
        r"""
        Initialize the card.
        """
        self._card = self._ranksuit2card(rank, suit)

    def __str__(self):
        r"""
        Return a text based version of the card.
        """
        suit = self.get_suit()
        rank = self.get_rank()
        text = rank_symbol[rank] + suit_symbol[suit]
        return text

    @staticmethod
    def _card2rank(card):
        r"""
        Converts a given numeric card to it's rank.
        """
        return card % NUM_RANKS

    @staticmethod
    def _card2suit(card):
        r"""
        Converts a given numeric card to it's suit.
        """
        return card // NUM_RANKS

    @staticmethod
    def _ranksuit2card(rank, suit):
        r"""
        Converts a given rank and suit to a numeric card.
        """
        return rank + NUM_RANKS*suit

    def get_rank(self):
        r"""
        Gets the rank of the current card.
        """
        return self._card2rank(self._card)

    def get_suit(self):
        r"""
        Gets the suit of the current card.
        """
        return self._card2suit(self._card)

class Hand(object):
    r"""
    Poker hand
    """
    def __init__(self, cards=None):
        r"""Creates the initial empty hand."""
        self._cards = []
        if cards is not None:
            for this_card in cards:
                self._cards.append(this_card)

    def __str__(self):
        text = ', '.join(str(card) for card in self._cards)
        return text

    def add_card(self, card):
        self._cards.append(card)

    def remove_card(self, card):
        self._cards.remove(card)

    def get_ranks(self):
        ranks = [card.get_rank() for card in self._cards]
        return ranks

    def get_suits(self):
        suits = [card.get_suit() for card in self._cards]
        return suits

    def score_hand(self):
        # hand order:
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

        # place-holder, delete later
        score = 0

        # get the ranks and suits
        ranks = self.get_ranks()
        suits = self.get_suits()

        # count the number of suits and get unique sets for straights
        suits_count = Counter(suits)
        ranks_count = Counter(ranks)
        ranks_set   = set(ranks)

        # determine if there is a flush or straight
        has_flush    = max(suits_count.values()) >= 5
        has_straight = ranks_set in STRAIGHTS

        # check for five of a kind
        if max(ranks_count.values()) >= 5:
            pass
            #rank_ix = np.argmax(ranks_count[::-1])
            #score = 9 + (rank_ix-NUM_RANKS) / NUM_RANKS
        # check for straight flush
        #elif np.any(suits_count) >= 5 and np.any(straight_count) >= 5:
        #    # check that cards from the straight also make the flush
        #    straight_ix = np.nonzero(straight_count == 5)[0]
        #    for this_straight in straight_ix[::-1]:
        #        for this_suit in NUM_SUITS:
        #            has_all_ranks = True
        #            for ranks in STRAIGHTS[this_straight, :]:
        #                pass
        return score

#%% Unit test
if __name__ == '__main__':
    hand = Hand([Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.HEARTS), \
        Card(Rank.JACK, Suit.DIAMONDS), Card(Rank.TEN, Suit.CLUBS)])
    print(hand)
    unittest.main(module='dstauffman.games.test_cards', exit=False)
    doctest.testmod(verbose=False)
