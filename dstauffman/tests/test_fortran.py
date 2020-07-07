r"""
Test file for the `fortran` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2019.
"""

#%% Imports
import os
import unittest

import dstauffman as dcs

#%% _parse_source
class Test__parse_source(unittest.TestCase):
    r"""
    Tests the _parse_source function with these cases:
        call the function
    """
    def setUp(self):
        lines = ['module test_mod']
        lines.append('')
        lines.append('use constants, only: RK')
        lines.append('use utils ! for unit_vec')
        lines.append('implicit none')
        lines.append('function func')
        lines.append('end function func')
        lines.append('subroutine sub_name(x, y, z)')
        lines.append('end subroutine')
        lines.append('')
        lines.append('! comment line')
        lines.append('    function add(x, y) result(z) ! function to add stuff')
        lines.append('    end function add')
        lines.append('end module test_mod')
        text = '\n'.join(lines)
        self.filename = os.path.join(dcs.get_tests_dir(), 'temp_code.f90')
        dcs.write_text_file(self.filename, text)

    def test_function(self):
        this_code = dcs.fortran._parse_source(self.filename)
        self.assertEqual(this_code.mod_name, 'test_mod')
        self.assertEqual(this_code.uses, ['constants', 'utils'])
        self.assertEqual(this_code.types, [])
        self.assertEqual(this_code.functions, ['func', 'add'])
        self.assertEqual(this_code.subroutines, ['sub_name'])

    def test_not_single(self):
        code = dcs.fortran._parse_source(self.filename, assert_single=False)
        self.assertEqual(len(code), 1)
        this_code = code[0]
        self.assertEqual(this_code.mod_name, 'test_mod')
        self.assertEqual(this_code.uses, ['constants', 'utils'])
        self.assertEqual(this_code.types, [])
        self.assertEqual(this_code.functions, ['func', 'add'])
        self.assertEqual(this_code.subroutines, ['sub_name'])

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

#%% _write_unit_test
pass # TODO

#%% _write_makefile
pass # TODO

#%% create_fortran_unit_tests
pass # TODO

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
