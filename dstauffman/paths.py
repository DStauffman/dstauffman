r"""
Generic path functions that can be called independent of the current working directory.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Moved out of utils and into paths.py file in February 2019 by David C. Stauffer.
"""

#%% Imports
import doctest
import os
from typing import List
import unittest

#%% Functions - is_dunder
def is_dunder(name: str) -> bool:
    """
    Returns True if a __dunder__ name, False otherwise.

    Parameters
    ----------
    name : str
        Name of the file or method to determine if __dunder__ (Double underscore)

    Returns
    -------
    bool
        Whether the name is a dunder method or not

    Notes
    -----
    #.  Copied by David C. Stauffer in September 2020 from enum._is_dunder to allow it to be a
        public method.

    Examples
    --------
    >>> from dstauffman import is_dunder
    >>> print(is_dunder('__init__'))
    True

    >>> print(is_dunder('_private'))
    False

    """
    # Note that this is copied from the enum library, as it is not part of their public API.
    return len(name) > 4 and name[:2] == name[-2:] == '__' and name[2] != '_' and name[-3] != '_'

#%% Functions - get_root_dir
def get_root_dir() -> str:
    r"""
    Return the folder that contains this source file and thus the root folder for the whole code.

    Returns
    -------
    folder : str
        Location of the folder that contains all the source files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------
    >>> from dstauffman import get_root_dir
    >>> folder = get_root_dir()

    """
    # this folder is the root directory based on the location of this file (utils.py)
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return folder

#%% Functions - get_tests_dir
def get_tests_dir() -> str:
    r"""
    Return the default test folder location.

    Returns
    -------
    folder : str
        Location of the folder that contains all the test files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------
    >>> from dstauffman import get_tests_dir
    >>> folder = get_tests_dir()

    """
    # this folder is the 'tests' subfolder
    folder = os.path.join(get_root_dir(), 'tests')
    return folder

#%% Functions - get_data_dir
def get_data_dir() -> str:
    r"""
    Return the default data folder location.

    Returns
    -------
    folder : str
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import get_data_dir
    >>> folder = get_data_dir()

    """
    # this folder is the 'data' subfolder
    folder = os.path.abspath(os.path.join(get_root_dir(), '..', 'data'))
    return folder

#%% Functions - get_images_dir
def get_images_dir() -> str:
    r"""
    Return the default data folder location.

    Returns
    -------
    folder : str
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import get_images_dir
    >>> folder = get_images_dir()

    """
    # this folder is the 'images' subfolder
    folder = os.path.abspath(os.path.join(get_root_dir(), '..', 'images'))
    return folder

#%% Functions - get_output_dir
def get_output_dir() -> str:
    r"""
    Return the default output folder location.

    Returns
    -------
    folder : str
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import get_output_dir
    >>> folder = get_output_dir()

    """
    # this folder is the 'images' subfolder
    folder = os.path.abspath(os.path.join(get_root_dir(), '..', 'results'))
    return folder

#%% Functions - list_python_files
def list_python_files(folder: str, recursive: bool = False, include_all: bool = False) -> List[str]:
    r"""
    Returns a list of all non dunder python files in the folder.

    Parameters
    ----------
    folder : str
        Folder location
    recursive : bool, optional
        Whether to search recursively, default is False
    include_all : bool, optional
        Whether to include all files, even the __dunder__ ones

    Returns
    -------
    files : list
        All *.py files that don't start with __

    Notes
    -----
    #.  Written by David C. Stauffer in March 2020.

    Examples
    --------
    >>> from dstauffman import list_python_files, get_root_dir
    >>> folder = get_root_dir()
    >>> files = list_python_files(folder)

    """
    # find all the files that end in .py and are not dunder (__name__) files
    if not os.path.isdir(folder):
        return []
    if include_all:
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.py')]
    else:
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.py')
            and not is_dunder(file[:-3])]
    if recursive:
        for (root, dirs, _) in os.walk(folder, topdown=True):
            for sub_folder in sorted(dirs):
                this_folder = os.path.join(root, sub_folder)
                files.extend(list_python_files(this_folder, recursive=recursive, include_all=include_all))
    return files

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_paths', exit=False)
    doctest.testmod(verbose=False)
