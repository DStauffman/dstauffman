r"""
Generic path functions that can be called independent of the current working directory.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Moved out of utils and into paths.py file in February 2019 by David C. Stauffer.
"""

# %% Imports
import doctest
from functools import lru_cache
from pathlib import Path
import unittest


# %% Functions - get_root_dir
@lru_cache
def get_root_dir() -> Path:
    r"""
    Return the folder that contains this source file and thus the root folder for the whole code.

    Returns
    -------
    class pathlib.Path
        Location of the folder that contains all the source files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------
    >>> from dstauffman import get_root_dir
    >>> print("p = ", repr(get_root_dir()))  # doctest: +ELLIPSIS
    p = .../dstauffman')

    """
    # this folder is the root directory based on the location of this file (paths.py)
    return Path(__file__).resolve().parent


# %% Functions - get_tests_dir
@lru_cache
def get_tests_dir() -> Path:
    r"""
    Return the default test folder location.

    Returns
    -------
    class pathlib.Path
        Location of the folder that contains all the test files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------
    >>> from dstauffman import get_tests_dir
    >>> print("p = ", repr(get_tests_dir()))  # doctest: +ELLIPSIS
    p = .../tests')

    """
    # this folder is the "tests" subfolder
    return get_root_dir() / "tests"


# %% Functions - get_data_dir
@lru_cache
def get_data_dir() -> Path:
    r"""
    Return the default data folder location.

    Returns
    -------
    class pathlib.Path
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import get_data_dir
    >>> print("p = ", repr(get_data_dir()))  # doctest: +ELLIPSIS
    p = .../data')

    """
    # this folder is the "data" subfolder
    return get_root_dir().parent / "data"


# %% Functions - get_images_dir
@lru_cache
def get_images_dir() -> Path:
    r"""
    Return the default data folder location.

    Returns
    -------
    class pathlib.Path
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import get_images_dir
    >>> print("p = ", repr(get_images_dir()))  # doctest: +ELLIPSIS
    p = .../images')

    """
    # this folder is the "images" subfolder
    return get_root_dir().parent / "images"


# %% Functions - get_output_dir
@lru_cache
def get_output_dir() -> Path:
    r"""
    Return the default output folder location.

    Returns
    -------
    class pathlib.Path
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import get_output_dir
    >>> print("p = ", repr(get_output_dir()))  # doctest: +ELLIPSIS
    p = .../results')

    """
    # this folder is the "results" subfolder
    return get_root_dir().parent / "results"


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_paths", exit=False)
    doctest.testmod(verbose=False)
