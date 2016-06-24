# -*- coding: utf-8 -*-
"""
The "profile_knight" file solves runs a subset of knight code using 'pprofile' to do a line by
line profiling.

Notes
-----
#.  Written by David C. Stauffer in June 2015.
"""

#%% Imports
import os
import pprofile
import timeit
import dstauffman.games.knight as knight
from dstauffman import get_output_dir, setup_dir

#%% Script
if __name__ == '__main__':

    do_board = 'none' # from {'none', 'small', 'large', 'both'}

    # convert board to numeric representation for efficiency
    board1 = knight.char_board_to_nums(knight.BOARD1)
    board2 = knight.char_board_to_nums(knight.BOARD2)

    # create the output folder
    folder = os.path.join(get_output_dir(), 'knight')
    setup_dir(folder)

    # create profiler
    profile = pprofile.Profile()

    # Small board
    if do_board in {'small', 'both'}:
        print('\nSolve the smaller board for the minimum length solution.')
        # enable/disable profiler while running solver
        profile.enable()
        moves3 = knight.solve_min_puzzle(board1)
        profile.disable()

        # print solution
        print(moves3)
        is_valid3 = knight.check_valid_sequence(board1, moves3, print_status=True)
        if is_valid3:
            knight.print_sequence(board1, moves3)

    # Large board
    if do_board in {'large', 'both'}:
        print('\nSolve the larger board for the minimum length solution.')
        board2[0, 0] = knight.Piece.start
        #board2[-1, -1] = knight.Piece.final # doesn't use transport
        board2[11, -1] = knight.Piece.final # uses transport

        # enable/disable profiler while running solver
        profile.enable()
        moves4 = knight.solve_min_puzzle(board2)
        profile.disable()

        # print solution
        print(moves4)
        is_valid4 = knight.check_valid_sequence(board2, moves4, print_status=True)
        if is_valid4:
            knight.print_sequence(board2, moves4)

    # timing
    t = timeit.timeit('knight.solve_min_puzzle(board2)', setup='import dstauffman.games.knight as knight; ' + \
        'board2=knight.char_board_to_nums(knight.BOARD2); ' +
        'board2[0, 0] = knight.Piece.start; board2[11, -1] = knight.Piece.final;', number=5)
    print(t/5)

    # Save profile results
    profile.dump_stats(os.path.join(folder, 'profile_results.txt'))
    with open(os.path.join(folder, 'profile_results.callgrind'), 'w+') as file:
        profile.callgrind(file)
