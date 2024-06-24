r"""
Test file for the `orbit_const` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

# %% Imports
import unittest

import dstauffman.aerospace as space


# %% Classes for testing
class Test_orbits_all_values(unittest.TestCase):
    r"""
    Tests all the constant values in the module.
    """

    def setUp(self) -> None:
        self.float: list[str] = [
            "PI",
            "TAU",
            "G",
            "SIDEREAL_DAY",
            "SIDEREAL_YEAR",
            "AU",
            "MU_SUN",
            "MU_EARTH",
            "SPEED_OF_LIGHT",
            "ECLIPTIC",
        ]
        self.dicts: list[str] = ["SS_MASSES", "JULIAN", "EARTH", "PALO_ALTO"]
        self.extra: list[str] = ["DEG2RAD", "HAVE_NUMPY"]  # imported constants
        self.master = set(self.float) | set(self.dicts) | set(self.extra)

    def test_values(self) -> None:
        # confirm that all the expected values exist and have the correct type
        for key in self.float:
            self.assertTrue(isinstance(getattr(space, key), float))
        for key in self.dicts:
            self.assertTrue(isinstance(getattr(space, key), dict))

    def test_missing(self) -> None:
        for field in vars(space.orbit_const):
            if field.isupper():
                self.assertTrue(field in self.master, "Test is missing: {}".format(field))


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
