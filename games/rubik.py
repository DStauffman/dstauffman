# -*- coding: utf-8 -*-
"""
The "rubik" file contains code related to Rubik's Cubes.

Notes
-----
#.  Written by David C. Stauffer in September 2015 after he found some old school files of Rubik's
    cube permutations.
"""
# pylint: disable=E1101, C0326, C0103

#%% Imports
# regular imports
import doctest
from math import factorial
import os
# model imports
from dstauffman import get_root_dir

#%% Constants
COLORS = {}
COLORS['U'] = '#ffffff' # white
COLORS['L'] = '#ff6600' # orange
COLORS['F'] = '#00cc00' # green
COLORS['R'] = '#cc0000' # red
COLORS['B'] = '#ffff00' # yellow
COLORS['D'] = '#0000cc' # blue

#%% Functions
def rubiks_cube_permutations(size=3):
    r"""
    Calculates the number of valid permutations in the given sized cube.

    Parameters
    ----------
    size : int
        Size of the Rubik's cube, assumed to be cubic.

    Returns
    -------
    perms : int
        Number of permutations, quite possibly a very long int.

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Look at http://www.therubikzone.com/Number-Of-Combinations.html for an example calculator.

    Examples
    --------

    >>> from dstauffman.games.rubik import rubiks_cube_permutations
    >>> perms = rubiks_cube_permutations(3)
    >>> print(perms)
    43252003274489856000

    """
    if size == 1:
        perms = 1
    elif size == 2:
        perms = factorial(8) * 3**8 // (2 * 3)
        perms = 3674160 # TODO: calculate this
    elif size == 3:
        return factorial(8) * 3**8 * factorial(12) * 2**12 // (2 * 3 * 2)
    elif size == 4:
        perms = 7401196841564901869874093974498574336000000000 # TODO: find generic algorithm for all the rest of them!
    elif size == 5:
        perms = 282870942277741856536180333107150328293127731985672134721536000000000000000
    elif size == 6:
        perms = 157152858401024063281013959519483771508510790313968742344694684829502629887168573442107637760000000000000000000000000
    elif size == 7:
        perms = 19500551183731307835329126754019748794904992692043434567152132912323232706135469180065278712755853360682328551719137311299993600000000000000000000000000000000000
    elif size == 8:
        perms = 35173780923109452777509592367006557398539936328978098352427605879843998663990903628634874024098344287402504043608416113016679717941937308041012307368528117622006727311360000000000000000000000000000000000000000000000000
    elif size == 9:
        perms = 14170392390542612915246393916889970752732946384514830589276833655387444667609821068034079045039617216635075219765012566330942990302517903971787699783519265329288048603083134861573075573092224082416866010882486829056000000000000000000000000000000000000000000000000000000000000000
    #elif size == 10:
    #    perms = 82 983 598 512 782 362 708 769 381 780 036 344 745 129 162 094 677 382 883 567 691 311 764 021 348 095 163 778 336 143 207 042 993 152 056 079 271 030 423 741 110 902 768 732 457 008 486 832 096 777 758 106 509 177 169 197 894 747 758 859 723 340 177 608 764 906 985 646 389 382 047 319 811 227 549 112 086 753 524 742 719 830 990 076 805 422 479 380 054 016 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000
    #elif size == 11:
    #    perms = 108 540 871 852 024 137 837 529 457 366 425 345 163 408 989 164 877 909 166 491 842 616 641 991 981 135 011 689 476 695 849 803 941 790 591 401 795 168 969 498 249 897 355 324 768 178 088 518 513 156 026 831 828 793 854 471 326 717 801 604 260 274 446 021 846 541 136 205 357 444 802 749 291 495 386 649 979 610 567 642 710 417 177 711 042 509 688 835 903 368 099 465 519 253 326 878 312 637 499 376 794 203 125 671 816 898 434 564 096 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000
    else:
        raise ValueError('Not Implemented.')
    return perms

#%% Functions - test_docstrings
def test_docstrings():
    r"""
    Tests the docstrings within this file.
    """
    file = os.path.join(get_root_dir(), 'games', 'rubik.py')
    doctest.testfile(file, report=True, verbose=False, module_relative=True)

#%% Main script
if __name__ == '__main__':
    # flags for running code
    run_tests    = True

    if run_tests:
        # Run docstring test
        test_docstrings()
    for i in range(1, 10):
        print('A {0}x{0}x{0} Cube has {1} permutations.'.format(i, rubiks_cube_permutations(i)))