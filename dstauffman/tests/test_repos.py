# -*- coding: utf-8 -*-
r"""
Test file for the `repos` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in July 2019.
"""

#%% Imports
import os
import unittest

import dstauffman as dcs

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
    @classmethod
    def setUpClass(cls):
        cls.folder = dcs.get_tests_dir()
        cls.linesep = os.linesep.replace('\n', '\\n').replace('\r', '\\r')
        file1 = os.path.join(cls.folder, 'temp_code_01.py')
        file2 = os.path.join(cls.folder, 'temp_code_02.py')
        file3 = os.path.join(cls.folder, 'temp_code_03.m')
        cont1 = 'Line 1\n\nAnother line\n    Line with leading spaces\n'
        cont2 = '\n\n    Start line\nNo Bad tab lines\n    Start and end line    \nAnother line\n\n'
        cont3 = '\n\n    Start line\n\tBad tab line\n    Start and end line    \nAnother line\n\n'
        cls.files = [file1, file2, file3]
        dcs.write_text_file(file1, cont1)
        dcs.write_text_file(file2, cont2)
        dcs.write_text_file(file3, cont3)
        cls.bad1 = "    Line 004: '\\tBad tab line" + cls.linesep + "'"
        cls.bad2 = "    Line 005: '    Start and end line    " + cls.linesep + "'"

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='m', list_all=False, trailing=False)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines), 3)

    def test_different_extensions(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions=('txt',))
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], '')
        self.assertEqual(len(lines), 1)

    def test_list_all(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, list_all=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(self.bad1 in lines)
        self.assertFalse(self.bad2 in lines)

    def test_trailing_spaces(self):
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

    def test_trailing_and_list_all(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, list_all=True, trailing=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertTrue(self.bad1 in lines)
        self.assertTrue(self.bad2 in lines)
        self.assertTrue(len(lines) > 7)

    def test_exclusions_skip(self):
        exclusions = (self.folder)
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertEqual(lines, [''])

    def test_exclusions_invalid(self):
        exclusions = (r'C:\non_existant_path', )
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines),  3)

    def test_bad_newlines(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='m', check_eol='0')
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('File: "'))
        self.assertTrue(lines[0].endswith('" has bad line endings of "{}".'.format(self.linesep)))
        self.assertTrue(lines[1].startswith('Evaluating: "'))
        self.assertEqual(lines[2], self.bad1)
        self.assertEqual(lines[3], '')
        self.assertEqual(len(lines),  4)

    def test_ignore_tabs(self):
        with dcs.capture_output() as out:
            dcs.find_repo_issues(self.folder, extensions='m', check_tabs=False)
        output = out.getvalue()
        out.close()
        self.assertEqual(output, '')

    @classmethod
    def tearDownClass(cls):
        for this_file in cls.files:
            if os.path.isfile(this_file):
                os.remove(this_file)

#%% delete_pyc
class Test_delete_pyc(unittest.TestCase):
    r"""
    Tests the delete_pyc function with the following cases:
        Recursive
        Not recursive
    """
    def setUp(self):
        self.fold1 = dcs.get_tests_dir()
        self.file1 = os.path.join(self.fold1, 'temp_file.pyc')
        self.fold2 = os.path.join(self.fold1, 'temp_sub_dir')
        self.file2 = os.path.join(self.fold2, 'temp_file2.pyc')
        dcs.write_text_file(self.file1, 'Text.')
        os.makedirs(self.fold2)
        dcs.write_text_file(self.file2, 'More text.')

    def test_recursive(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(os.path.isfile(self.file1))
        self.assertFalse(os.path.isfile(self.file2))
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"') or this_line.endswith('temp_file2.pyc"'))

    def test_not_recursive(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1, recursive=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(os.path.isfile(self.file1))
        self.assertTrue(os.path.isfile(self.file2))
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"'))

    def test_no_logging(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as out:
            dcs.delete_pyc(self.fold1, print_progress=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(os.path.isfile(self.file1))
        self.assertFalse(os.path.isfile(self.file2))
        self.assertEqual(output, '')

    def tearDown(self):
        if os.path.isfile(self.file1):
            os.remove(self.file1)
        if os.path.isfile(self.file2):
            os.remove(self.file2)
        if os.path.isdir(self.fold2):
            os.removedirs(self.fold2)

#%% get_python_definitions
class Test_get_python_definitions(unittest.TestCase):
    r"""
    Tests the get_python_definitions function with these cases:
        Functions
        Classes
        No arguments
        Lots of arguments
    """
    def test_functions(self):
        funcs = dcs.get_python_definitions('def a():\n    pass\ndef _b():\n    pass\n')
        self.assertEqual(funcs, ['a'])

    def test_classes(self):
        funcs = dcs.get_python_definitions('def a():\n    pass\nclass b():\n    pass\nclass _c():\n    pass\n')
        self.assertEqual(funcs, ['a', 'b'])

    def test_no_inputs(self):
        funcs = dcs.get_python_definitions('def _a:\n    pass\ndef b:\n    pass\n')
        self.assertEqual(funcs, ['b'])

    def test_with_inputs(self):
        funcs = dcs.get_python_definitions('def a(a, b=2):\n    pass\nclass bbb(c, d):\n    pass\nclass _c(e):\n    pass\n')
        self.assertEqual(funcs, ['a', 'bbb'])

    def test_nothing(self):
        funcs = dcs.get_python_definitions('')
        self.assertEqual(len(funcs), 0)

    def test_constant_values(self):
        funcs = dcs.get_python_definitions('def a():\n    pass\nCONSTANT = 5\n')
        self.assertEqual(funcs, ['a', 'CONSTANT'])

#%% make_python_init
class Test_make_python_init(unittest.TestCase):
    r"""
    Tests the make_python_init function with these cases:
        TBD
    """
    def setUp(self):
        self.folder   = dcs.get_root_dir()
        self.text     = 'from .bpe import'
        self.text2    = 'from .bpe          import'
        self.folder2  = dcs.get_tests_dir()
        self.filepath = os.path.join(self.folder2, 'temp_file.py')
        self.filename = os.path.join(self.folder2, '__init__2.py')

    def test_nominal_use(self):
        text = dcs.make_python_init(self.folder)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)

    def test_duplicated_funcs(self):
        with open(self.filepath, 'wt') as file:
            file.write('def Test_Frozen():\n    pass\n')
        with dcs.capture_output() as out:
            text = dcs.make_python_init(self.folder2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(text[0:42], 'from .temp_file         import Test_Frozen')
        self.assertTrue(output.startswith('Uniqueness Problem'))

    def test_no_lineup(self):
        text = dcs.make_python_init(self.folder, lineup=False)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text)], self.text)

    def test_big_wrap(self):
        text = dcs.make_python_init(self.folder, wrap=1000)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)

    def test_small_wrap(self):
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=30)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "26:30" was too small.')

    def test_really_small_wrap(self):
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=10)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "26:10" was too small.')

    def test_saving(self):
        text = dcs.make_python_init(self.folder, filename=self.filename)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)
        self.assertTrue(os.path.isfile(self.filename))

    def tearDown(self):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)
        if os.path.isfile(self.filename):
            os.remove(self.filename)
#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
