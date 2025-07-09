r"""
Test file for the `version` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in February 2019.

"""

# %% Imports
import unittest

import dstauffman as dcs


# %% version_info
class Test_version_info(unittest.TestCase):
    r"""
    Tests the get_root_dir function with the following cases:
        call the function
    """

    def test_version_info(self) -> None:
        version_info = dcs.version_info
        self.assertTrue(version_info >= (3, 0, 0))

    def test_data(self) -> None:
        data = dcs.version.data
        lines = data.split("\n")[:-1]
        found = False
        for line in lines:
            self.assertIn("dstauffman", line)
            if "dstauffman 3.0" in line:
                found = True
        self.assertTrue(found)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
