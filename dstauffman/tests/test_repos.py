r"""
Test file for the `repos` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2019.
"""

#%% Imports
from __future__ import annotations
import contextlib
import os
import pathlib
from typing import List
import unittest
from unittest.mock import patch

import dstauffman as dcs

#%% run_docstrings
class Test_run_docstrings(unittest.TestCase):
    r"""
    Tests the run_docstrings function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% run_unittests
class Test_run_unittests(unittest.TestCase):
    r"""
    Tests the run_unittests function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% run_pytests
class Test_run_pytests(unittest.TestCase):
    r"""
    Tests the run_pytests function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% run_coverage
class Test_run_coverage(unittest.TestCase):
    r"""
    Tests the run_coverage function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% find_repo_issues
class Test_find_repo_issues(unittest.TestCase):
    r"""
    Tests the find_repo_issues function with the following cases:
        Nominal
        Different Extensions
        List All
        Trailing Spaces
        Exclusions x2
        Bad New Lines
        Ignore tabs
    """
    folder: pathlib.Path
    linesep: str
    files: List[pathlib.Path]
    bad1: str
    bad2: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = dcs.get_tests_dir()
        cls.linesep = os.linesep.replace('\n', '\\n').replace('\r', '\\r')
        file1 = cls.folder / 'temp_code_01.py'
        file2 = cls.folder / 'temp_code_02.py'
        file3 = cls.folder / 'temp_code_03.m'
        cont1 = 'Line 1\n\nAnother line\n    Line with leading spaces\n'
        cont2 = '\n\n    Start line\nNo Bad tab lines\n    Start and end line    \nAnother line\n\n'
        cont3 = '\n\n    Start line\n\tBad tab line\n    Start and end line    \nAnother line\n\n'
        cls.files = [file1, file2, file3]
        dcs.write_text_file(file1, cont1)
        dcs.write_text_file(file2, cont2)
        dcs.write_text_file(file3, cont3)
        cls.bad1 = "    Line 004: '\\tBad tab line" + cls.linesep + "'"
        cls.bad2 = "    Line 005: '    Start and end line    " + cls.linesep + "'"

    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='.m', list_all=False, trailing=False)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines), 3)

    def test_different_extensions(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions=('.txt',))
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], '')
        self.assertEqual(len(lines), 1)

    def test_list_all(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, list_all=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(self.bad1 in lines)
        self.assertFalse(self.bad2 in lines)

    def test_trailing_spaces(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, trailing=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad2)
        self.assertTrue(lines[2].startswith('Evaluating: "'))
        self.assertEqual(lines[3], self.bad1)
        self.assertEqual(lines[4], self.bad2)
        self.assertEqual(lines[5], '')
        self.assertEqual(len(lines), 6)

    def test_trailing_and_list_all(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, list_all=True, trailing=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertTrue(self.bad1 in lines)
        self.assertTrue(self.bad2 in lines)
        self.assertTrue(len(lines) > 7)

    def test_exclusions_skip(self) -> None:
        exclusions = self.folder
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertEqual(lines, [''])

    def test_exclusions_invalid(self) -> None:
        exclusions = (pathlib.Path(r'C:\non_existant_path'), )
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines),  3)

    def test_bad_newlines(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='.m', check_eol='0')
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('File: "'))
        self.assertIn('" has bad line endings of "', lines[0])
        self.assertTrue(lines[3].startswith('Evaluating: "'))
        self.assertEqual(lines[4], self.bad1)
        self.assertEqual(lines[5], '')
        self.assertEqual(len(lines),  6)

    def test_ignore_tabs(self) -> None:
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='.m', check_tabs=False)
        output = out.getvalue()
        out.close()
        self.assertEqual(output, '')

    @classmethod
    def tearDownClass(cls) -> None:
        for this_file in cls.files:
            this_file.unlink(missing_ok=True)

#%% delete_pyc
class Test_delete_pyc(unittest.TestCase):
    r"""
    Tests the delete_pyc function with the following cases:
        Recursive
        Not recursive
    """
    def setUp(self) -> None:
        self.fold1 = dcs.get_tests_dir()
        self.file1 = self.fold1 / 'temp_file.pyc'
        self.fold2 = self.fold1 / 'temp_sub_dir'
        self.file2 = self.fold2 / 'temp_file2.pyc'
        dcs.write_text_file(self.file1, 'Text.')
        self.fold2.mkdir(exist_ok=True)
        dcs.write_text_file(self.file2, 'More text.')

    def test_recursive(self) -> None:
        self.assertTrue(self.file1.is_file())
        self.assertTrue(self.fold2.is_dir())
        self.assertTrue(self.file2.is_file())
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(self.file1.is_file())
        self.assertFalse(self.file2.is_file())
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"') or this_line.endswith('temp_file2.pyc"'))

    def test_not_recursive(self) -> None:
        self.assertTrue(self.file1.is_file())
        self.assertTrue(self.fold2.is_dir())
        self.assertTrue(self.file2.is_file())
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1, recursive=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(self.file1.is_file())
        self.assertTrue(self.file2.is_file())
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"'))

    def test_no_logging(self) -> None:
        self.assertTrue(self.file1.is_file())
        self.assertTrue(self.fold2.is_dir())
        self.assertTrue(self.file2.is_file())
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1, print_progress=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(self.file1.is_file())
        self.assertFalse(self.file2.is_file())
        self.assertEqual(output, '')

    def tearDown(self) -> None:
        self.file1.unlink(missing_ok=True)
        self.file2.unlink(missing_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.removedirs(self.fold2)

#%% get_python_definitions
class Test_get_python_definitions(unittest.TestCase):
    r"""
    Tests the get_python_definitions function with the following cases:
        Functions
        Classes
        No arguments
        Lots of arguments
    """
    def test_functions(self) -> None:
        funcs = dcs.get_python_definitions('def a():\n    pass\ndef _b():\n    pass\n')
        self.assertEqual(funcs, ['a'])

    def test_classes(self) -> None:
        funcs = dcs.get_python_definitions('def a():\n    pass\nclass b():\n    pass\nclass _c():\n    pass\n')
        self.assertEqual(funcs, ['a', 'b'])

    def test_no_inputs(self) -> None:
        funcs = dcs.get_python_definitions('def _a:\n    pass\ndef b:\n    pass\n')
        self.assertEqual(funcs, ['b'])

    def test_with_inputs(self) -> None:
        funcs = dcs.get_python_definitions('def a(a, b=2):\n    pass\nclass bbb(c, d):\n    pass\nclass _c(e):\n    pass\n')
        self.assertEqual(funcs, ['a', 'bbb'])

    def test_nothing(self) -> None:
        funcs = dcs.get_python_definitions('')
        self.assertEqual(len(funcs), 0)

    def test_constant_values(self) -> None:
        funcs = dcs.get_python_definitions('def a():\n    pass\nCONSTANT = 5\n')
        self.assertEqual(funcs, ['a', 'CONSTANT'])

    def test_include_private(self) -> None:
        funcs = dcs.get_python_definitions('def a():\n    pass\ndef _b():\n    pass\nclass _c():\n    pass\n', include_private=True)
        self.assertEqual(funcs, ['a', '_b', '_c'])

    def test_overload(self) -> None:
        funcs = dcs.get_python_definitions('@overload\ndef fun(x: int, x: Literal[False]) -> int: ...\n\n@overload\ndef fun(x: int, x: Literal[True]) -> float: ...\n' +
            '\ndef fun(x: int, x: bool = False) -> Union[int, float]:\n    pass\n\n')
        self.assertEqual(funcs, ['fun'])

    def test_typed_constants(self) -> None:
        funcs = dcs.get_python_definitions('def fun(x: int, y: int):\n    pass\nCONSTANT: int = 5\n')
        self.assertEqual(funcs, ['fun', 'CONSTANT'])

#%% make_python_init
class Test_make_python_init(unittest.TestCase):
    r"""
    Tests the make_python_init function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.folder   = dcs.get_root_dir()
        self.text     = 'from .enums import'
        self.text2    = 'from .enums     import IntEnumPlus'
        self.line_num = 5
        self.folder2  = dcs.get_tests_dir()
        self.filepath = self.folder2 / 'temp_file.py'
        self.filename = self.folder2 / '__init__2.py'

    def test_nominal_use(self) -> None:
        text = dcs.make_python_init(self.folder)
        lines = text.split('\n')
        self.assertEqual(lines[self.line_num][0:len(self.text2)], self.text2)

    def test_duplicated_funcs(self) -> None:
        with open(self.filepath, 'wt') as file:
            file.write('def Test_Frozen():\n    pass\n')
        with dcs.capture_output() as out:
            text = dcs.make_python_init(self.folder2)
        output = out.getvalue().strip()
        out.close()
        self.assertRegex(text[0:100], r'from \.temp\_file(\s{2,})import Test_Frozen')
        self.assertTrue(output.startswith('Uniqueness Problem'))

    def test_no_lineup(self) -> None:
        text = dcs.make_python_init(self.folder, lineup=False)
        lines = text.split('\n')
        self.assertEqual(lines[self.line_num][0:len(self.text)], self.text)

    def test_big_wrap(self) -> None:
        text = dcs.make_python_init(self.folder, wrap=1000)
        lines = text.split('\n')
        self.assertEqual(lines[self.line_num-3][0:len(self.text2)], self.text2)

    def test_small_wrap(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=30)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "23:30" was too small.')

    def test_really_small_wrap(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=10)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "23:10" was too small.')

    def test_saving(self) -> None:
        text = dcs.make_python_init(self.folder, filename=self.filename)
        lines = text.split('\n')
        self.assertEqual(lines[self.line_num][0:len(self.text2)], self.text2)
        self.assertTrue(self.filename.is_file())

    def tearDown(self) -> None:
        self.filepath.unlink(missing_ok=True)
        self.filename.unlink(missing_ok=True)

#%% write_unit_test_templates
class Test_write_unit_test_templates(unittest.TestCase):
    r"""
    Tests the write_unit_test_templates function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.folder = dcs.get_root_dir()
        self.output = pathlib.Path(str(dcs.get_tests_dir()) + '_template')
        self.author = 'David C. Stauffer'
        self.exclude = dcs.get_tests_dir()

    def test_nominal(self) -> None:
        with patch('dstauffman.repos.write_text_file') as mock_writer:
            with patch('dstauffman.repos.setup_dir') as mock_dir:
                with dcs.capture_output() as out:
                    dcs.write_unit_test_templates(self.folder, self.output, author=self.author, exclude=self.exclude)
                lines = out.getvalue().strip().split('\n')
                out.close()
                self.assertEqual(mock_dir.call_count, 1)
        self.assertGreater(len(lines), 5)
        self.assertTrue(lines[0].startswith('Writing: '))
        self.assertGreater(mock_writer.call_count, 5)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
