r"""
Test file for the `paths` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in February 2019.
"""

#%% Imports
import inspect
import os
import pathlib
import unittest

import dstauffman as dcs

#%% is_dunder
class Test_is_dunder(unittest.TestCase):
    r"""
    Tests the is_dunder function with the following cases:
        True
        False
    """

    def setUp(self) -> None:
        self.true = ['__dunder__', '__init__', '__a__']
        self.false = ['init', '__init__.py', '_private', '__private', 'private__', '____']

    def test_trues(self) -> None:
        for key in self.true:
            self.assertTrue(dcs.is_dunder(key), key + ' Should be a __dunder__ method')

    def test_falses(self) -> None:
        for key in self.false:
            self.assertFalse(dcs.is_dunder(key), key + ' Should not be considered dunder.')


#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with the following cases:
        call the function
    """

    def test_function(self) -> None:
        filepath = inspect.getfile(dcs.get_root_dir.__wrapped__)
        expected_root = pathlib.Path(os.path.split(filepath)[0])
        folder = dcs.get_root_dir()
        self.assertEqual(folder, expected_root)
        self.assertTrue(folder.is_dir())


#%% get_tests_dir
class Test_get_tests_dir(unittest.TestCase):
    r"""
    Tests the get_tests_dir function with the following cases:
        call the function
    """

    def test_function(self) -> None:
        folder = dcs.get_tests_dir()
        self.assertEqual(str(folder), os.path.join(str(dcs.get_root_dir()), 'tests'))


#%% get_data_dir
class Test_get_data_dir(unittest.TestCase):
    r"""
    Tests the get_data_dir function with the following cases:
        call the function
    """

    def test_function(self) -> None:
        folder = dcs.get_data_dir()
        self.assertEqual(str(folder), os.path.abspath(os.path.join(str(dcs.get_root_dir()), '..', 'data')))


#%% get_images_dir
class Test_get_images_dir(unittest.TestCase):
    r"""
    Tests the get_images_dir function with the following cases:
        call the function
    """

    def test_function(self) -> None:
        folder = dcs.get_images_dir()
        self.assertEqual(str(folder), os.path.abspath(os.path.join(str(dcs.get_root_dir()), '..', 'images')))


#%% get_output_dir
class Test_get_output_dir(unittest.TestCase):
    r"""
    Tests the get_output_dir function with the following cases:
        call the function
    """

    def test_function(self) -> None:
        folder = dcs.get_output_dir()
        self.assertEqual(str(folder), os.path.abspath(os.path.join(str(dcs.get_root_dir()), '..', 'results')))


#%% list_python_files
class Test_list_python_files(unittest.TestCase):
    r"""
    Tests the list_python_files function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.folder = dcs.get_root_dir() / 'commands'
        self.expected = [self.folder / x for x in ['help.py', 'repos.py', 'runtests.py']]

    def test_nominal(self) -> None:
        files = dcs.list_python_files(self.folder)
        for (file, exp) in zip(files, self.expected):
            self.assertEqual(file, exp)


#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
