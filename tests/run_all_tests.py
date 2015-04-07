# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the unittest library within the GHAP model using nose.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import nose
import dstauffman as dcs

if __name__ == '__main__':
    nose.run(dcs)