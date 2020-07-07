r"""
Test file to execute all the tests from the dstauffman library using pytest.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from dstauffman import get_root_dir, run_pytests

#%% Tests
if __name__ == '__main__':
    run_pytests(get_root_dir())
