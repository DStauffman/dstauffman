r"""
Test file for the `fortran` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2019.
"""

# %% Imports
import unittest

import dstauffman as dcs


# %% fortran._FortranSource
class Test_fortran__FortranSource(unittest.TestCase):
    r"""
    Tests the fortran._FortranSource class with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% fortran._parse_source
class Test_fortran__parse_source(unittest.TestCase):
    r"""
    Tests the fortran._parse_source function with the following cases:
        call the function
    """

    def setUp(self) -> None:
        lines = ["module test_mod"]
        lines.append("")
        lines.append("use constants, only: RK")
        lines.append("use utils ! for unit_vec")
        lines.append("implicit none")
        lines.append("function func")
        lines.append("end function func")
        lines.append("subroutine sub_name(x, y, z)")
        lines.append("end subroutine")
        lines.append("")
        lines.append("! comment line")
        lines.append("    function add(x, y) result(z) ! function to add stuff")
        lines.append("    end function add")
        lines.append("end module test_mod")
        text = "\n".join(lines)
        self.filename = dcs.get_tests_dir() / "temp_code.f90"
        dcs.write_text_file(self.filename, text)

    def test_function(self) -> None:
        this_code = dcs.fortran._parse_source(self.filename)
        self.assertEqual(this_code.mod_name, "test_mod")
        self.assertEqual(this_code.uses, ["constants", "utils"])
        self.assertEqual(this_code.types, [])
        self.assertEqual(this_code.functions, ["func", "add"])
        self.assertEqual(this_code.subroutines, ["sub_name"])

    def test_not_single(self) -> None:
        code = dcs.fortran._parse_source(self.filename, assert_single=False)
        self.assertEqual(len(code), 1)
        this_code = code[0]
        self.assertEqual(this_code.mod_name, "test_mod")
        self.assertEqual(this_code.uses, ["constants", "utils"])
        self.assertEqual(this_code.types, [])
        self.assertEqual(this_code.functions, ["func", "add"])
        self.assertEqual(this_code.subroutines, ["sub_name"])

    def tearDown(self) -> None:
        self.filename.unlink(missing_ok=True)


# %% fortran._write_unit_test
class Test_fortran__write_unit_test(unittest.TestCase):
    r"""
    Tests the fortran._write_unit_test function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% fortran._write_all_unit_test
class Test_fortran__write_all_unit_test(unittest.TestCase):
    r"""
    Tests the fortran._write_all_unit_test function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% fortran._get_template
class Test_fortran__get_template(unittest.TestCase):
    r"""
    Tests the fortran._get_template function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% fortran._write_makefile
class Test_fortran__write_makefile(unittest.TestCase):
    r"""
    Tests the fortran._write_makefile function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% create_fortran_unit_tests
class Test_create_fortran_unit_tests(unittest.TestCase):
    r"""
    Tests the create_fortran_unit_tests function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% create_fortran_makefile
class Test_create_fortran_makefile(unittest.TestCase):
    r"""
    Tests the create_fortran_makefile function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
