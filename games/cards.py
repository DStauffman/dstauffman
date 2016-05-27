# -*- coding: utf-8 -*-
r"""
Cards module file for the "dstauffman" library.  It contains classes and routines specific to card
games such as poker.

Notes
-----
#.  Written by David C. Stauffer in May 2016.
"""

#%% Imports
import doctest
from enum import unique
import numpy as np
import unittest
from dstauffman import IntEnumPlus

#%% Enums  - Suit
@unique
class Suit(IntEnumPlus):
    r"""
    Enumerator definitions for the possible card suits.
    """
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3

#%% Enums - Rank
@unique
class Rank(IntEnumPlus):
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

#%% Classes
NUM_SUITS = 4
NUM_RANKS = 13

#%% Constants
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

STRAIGHTS = np.array([\
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],\
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],\
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],\
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],\
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],\
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],\
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=bool)

#%% Functions - card2rank
def card2rank(card):
    r"""
    Converts a given numeric card to it's rank.
    """
    return card % NUM_RANKS

#%% Functions - card2suit
def card2suit(card):
    r"""
    Converts a given numeric card to it's suit.
    """
    return card // NUM_RANKS

#%% Functions - ranksuit2card
def ranksuit2card(rank, suit):
    r"""
    Converts a given rank and suit to a numeric card.
    """
    return rank + NUM_RANKS*suit

#%% Functions - print_card
def print_cards(hand):
    r"""
    Prints the given hand to the screen
    """
    suits = card2suit(hand)
    ranks = card2rank(hand)
    text  = []
    for (suit, rank) in zip(suits, ranks):
        text.append(rank_symbol[rank] + suit_symbol[suit])
    print(', '.join(text))

#%% Functions - eval_hand
def eval_hand(hand):
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


    # check if enough cards for flush
    ranks = card2rank(hand)
    suits = card2suit(hand)
    suits_count = np.bincount(suits, minlength=NUM_SUITS)
    ranks_count = np.bincount(ranks, minlength=NUM_RANKS)
    straight_count = np.sum((ranks_count > 0) * STRAIGHTS, axis=1)
    # check for five of a kind
    if np.any(ranks_count) >= 5:
        rank_ix = np.argmax(ranks_count[::-1])
        score = 9 + (rank_ix-NUM_RANKS) / NUM_RANKS
    # check for straight flush
    elif np.any(suits_count) >= 5 and np.any(straight_count) >= 5:
        # check that cards from the straight also make the flush
        straight_ix = np.nonzero(straight_count == 5)[0]
        for this_straight in straight_ix[::-1]:
            for this_suit in NUM_SUITS:
                has_all_ranks = True
                for ranks in STRAIGHTS[this_straight, :]:
                    pass
    return score

#%% Unit test
if __name__ == '__main__':
    hand1 = [(Rank.ACE, Suit.SPADES), (Rank.KING, Suit.SPADES), (Rank.QUEEN, Suit.HEARTS), \
        (Rank.JACK, Suit.DIAMONDS), (Rank.TEN, Suit.CLUBS)]
    hand = np.array([ranksuit2card(x, y) for (x,y) in hand1])
    print_cards(hand)
    unittest.main(module='dstauffman.games.test_cards', exit=False)
    doctest.testmod(verbose=False)
