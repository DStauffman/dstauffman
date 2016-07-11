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
from random import shuffle as shuffle_func
import unittest

#%% Enums - Suit
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
# suit symbols for printing
SUIT_SYMBOL                = {}
SUIT_SYMBOL[Suit.CLUBS]    = '\u2663'
SUIT_SYMBOL[Suit.DIAMONDS] = '\u2666'
SUIT_SYMBOL[Suit.HEARTS]   = '\u2665'
SUIT_SYMBOL[Suit.SPADES]   = '\u2660'
# rank symbols for printing
RANK_SYMBOL             = {}
RANK_SYMBOL[Rank.TWO]   = '2'
RANK_SYMBOL[Rank.THREE] = '3'
RANK_SYMBOL[Rank.FOUR]  = '4'
RANK_SYMBOL[Rank.FIVE]  = '5'
RANK_SYMBOL[Rank.SIX]   = '6'
RANK_SYMBOL[Rank.SEVEN] = '7'
RANK_SYMBOL[Rank.EIGHT] = '8'
RANK_SYMBOL[Rank.NINE]  = '9'
RANK_SYMBOL[Rank.TEN]   = '10'
RANK_SYMBOL[Rank.JACK]  = 'J'
RANK_SYMBOL[Rank.QUEEN] = 'Q'
RANK_SYMBOL[Rank.KING]  = 'K'
RANK_SYMBOL[Rank.ACE]   = 'A'
# all possible straights
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
        text = RANK_SYMBOL[rank] + SUIT_SYMBOL[suit]
        return text

    def __lt__(self, other):
        return self.get_rank() < other.get_rank()

    def __le__(self, other):
        return self.get_rank() <= other.get_rank()

    def __gt__(self, other):
        return self.get_rank() > other.get_rank()

    def __ge__(self, other):
        return self.get_rank() >= other.get_rank()

    def __eq__(self, other):
        return self.get_rank() == other.get_rank()

    def __ne__(self, other):
        return self.get_rank() != other.get_rank()

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

#%% Classes - Deck
class Deck(object):
    r"""
    Standard Poker Deck.  52 Cards, no wilds.
    """
    def __init__(self, shuffle=False):
        r"""
        Initialize the deck.
        """
        self._cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        if shuffle:
            shuffle_func(self._cards)

    def __str__(self):
        text = ', '.join(str(card) for card in reversed(self._cards))
        return text

    def shuffle(self):
        shuffle_func(self._cards)

    def reset(self):
        self.__init__()

    def get_next_card(self):
        return self._cards.pop()

    def count_remaining_cards(self):
        return len(self._cards)

#%% Classes - Card
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

    @property
    def num_cards(self):
        return len(self._cards)

    def add_card(self, card):
        self._cards.append(card)

    def play_card(self):
        return self._cards.pop()

    def remove_card(self, card):
        self._cards.remove(card)

    def get_ranks(self):
        ranks = [card.get_rank() for card in self._cards]
        return ranks

    def get_suits(self):
        suits = [card.get_suit() for card in self._cards]
        return suits

    def shuffle(self):
        shuffle_func(self._cards)

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

#%% Classes - WarGame
class WarGame(object):
    r"""
    The game of War implemented as a class.
    """
    war_downs = 3
    def __init__(self):
        r"""
        Deal the cards to start the game.
        """
        # display a banner
        print('************')
        print('Playing war!')
        # initialize variables
        self._hand1 = Hand()
        self._hand2 = Hand()
        self._hold1 = Hand()
        self._hold2 = Hand()
        self._pot   = Hand()
        self.move   = 0
        # create and distribute cards
        deck = Deck()
        deck.shuffle()
        while deck.count_remaining_cards():
            self._hand1.add_card(deck.get_next_card())
            self._hand2.add_card(deck.get_next_card())

    def play_move(self):
        r"""
        Make a single move.
        """
        # increment the move number
        self.move += 1
        # get the two cards to play
        card1 = self._hand1.play_card()
        card2 = self._hand2.play_card()
        # carry a pot of cards to win, allowing for wars to extend beyond one hand
        self._pot.add_card(card1)
        self._pot.add_card(card2)
        # determine which card won, with ties going to wars
        if card1 > card2:
            print('Move {}: {} beats {}'.format(self.move, card1, card2), end='')
            while self._pot.num_cards > 0:
                self._hold1.add_card(self._pot.play_card())
            self._check_shuffle()
            self.print_card_count()
        elif card1 < card2:
            print('Move {}: {} loses {}'.format(self.move, card1, card2), end='')
            while self._pot.num_cards > 0:
                self._hold2.add_card(self._pot.play_card())
            self._check_shuffle()
            self.print_card_count()
        else:
            print('Move {}: {} wars {}'.format(self.move, card1, card2), end='')
            self._check_shuffle()
            self._play_war()

    def _check_shuffle(self):
        r"""
        Checks to see a player is out of cards, and if so shuffles their hold cards.
        """
        # If you have no cards in your hand, then shuffle your hold cards and then transfer them to
        # the hand
        if self._hand1.num_cards == 0:
            self._hold1.shuffle()
            while self._hold1.num_cards > 0:
                self._hand1.add_card(self._hold1.play_card())
        if self._hand2.num_cards == 0:
            self._hold2.shuffle()
            while self._hold2.num_cards > 0:
                self._hand2.add_card(self._hold2.play_card())

    def _play_war(self):
        r"""
        Plays the down cards when a war happens.
        """
        print(' Down cards (x{})'.format(self.war_downs), end='')
        for i in range(self.war_downs):
            self._pot.add_card(self._hand1.play_card())
            self._pot.add_card(self._hand2.play_card())
            self._check_shuffle()
            if self.is_winner():
                self.print_card_count()
                return
        self.print_card_count()
        self.play_move()

    def is_winner(self):
        r"""
        Determines if their is a winner (because someone is out of cards)
        """
        return self._hand1.num_cards == 0 or self._hand2.num_cards == 0

    def who_won(self):
        r"""
        Determines who won (based on someone being out of cards, and someone else
        still having cards.

        Notes
        -----
        #.  A game can theoretically draw with repeated wars such that both players run out of cards.
        """
        n1 = self._hand1.num_cards
        n2 = self._hand2.num_cards
        if n1 > 0 and n2 == 0:
            return 'Player 1 won!'
        elif n1 == 0 and n2 > 0:
            return 'Player 2 won!'
        elif n1 == 0 and n2 == 0:
            return 'Draw game!'
        else:
            print('Card count: ', end='')
            self.print_card_count()
            raise ValueError('Undetermined winner!')

    def play_game(self):
        r"""
        Wrapper function that plays a game until a winner is determined.
        """
        while not self.is_winner():
            self.play_move()
        print(self.who_won())

    def print_card_count(self):
        r"""
        Prints the current number of cards that each player has.
        """
        print(' ({} to {})'.format(self._hand1.num_cards+self._hold1.num_cards, \
            self._hand2.num_cards+self._hold2.num_cards))

#%% Unit test
if __name__ == '__main__':
    hand = Hand([Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.HEARTS), \
        Card(Rank.JACK, Suit.DIAMONDS), Card(Rank.TEN, Suit.CLUBS)])
    print(hand)
    print(hand.score_hand())

    deck = Deck()
    deck.reset()
    print(deck)
    unittest.main(module='dstauffman.games.test_cards', exit=False)
    doctest.testmod(verbose=False)

    # play War
    #war = WarGame()
    #war.play_game()
