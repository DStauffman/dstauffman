# -*- coding: utf-8 -*-
r"""
Constants module file for the "dstauffman" library.  It defines constants.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import unittest

#%% Constants
MONTHS_PER_YEAR  = 12
INT_TOKEN        = -1
DEFAULT_COLORMAP = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'
QUAT_SIZE        = 4

#%% Spyder
spyder_custom_colors = r"""
custom/background = '#000000'
custom/currentline = '#000000'
custom/currentcell = '#000000'
custom/occurence = '#cc0000'
custom/ctrlclick = '#55ffff'
custom/sideareas = '#181818'
custom/matched_p = '#355835'
custom/unmatched_p = '#9c171e'
custom/normal = ('#ffffff', False, False)
custom/keyword = ('#00bfff', False, False)
custom/builtin = ('#ff55ff', False, False)
custom/definition = ('#ffffff', True, False)
custom/comment = ('#00ff00', False, True)
custom/string = ('#ffff7f', False, False)
custom/number = ('#ff7979', False, False)
custom/instance = ('#ffff00', False, True)
"""

#%% Functions
# None

#%% Unit Test
def _main():
    r"""Unit test function."""
    print('MONTHS_PER_YEAR      = {}'.format(MONTHS_PER_YEAR))
    print('INT_TOKEN            = {}'.format(INT_TOKEN))
    print('DEFAULT_COLORMAP     = {}'.format(DEFAULT_COLORMAP))
    print('QUAT_SIZE            = {}'.format(QUAT_SIZE))
    print('spyder_custom_colors = ')
    print(spyder_custom_colors)

if __name__ == '__main__':
    unittest.main(module='tests.test_constants', exit=False)
    _main()
