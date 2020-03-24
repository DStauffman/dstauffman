# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the dstauffman library using pytest.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import dstauffman as dcs

#%% Tests
if __name__ == '__main__':
    dcs.run_pytests(dcs.get_root_dir())
