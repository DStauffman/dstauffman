r"""
Test file to execute all the docstrings within the dstauffman code.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from dstauffman import get_root_dir, list_python_files, run_docstrings

#%% Execution
if __name__ == '__main__':
    files = list_python_files(get_root_dir(), recursive=True)
    run_docstrings(files, verbose=False)
