r"""
Test file for the `constants` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

# %% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np


# %% Classes for testing
class Test_all_values(unittest.TestCase):
    r"""
    Tests all the constant values in the module.
    """

    def setUp(self) -> None:
        self.ints: list[str] = ["INT_TOKEN", "NP_INT64_PER_SEC"]
        self.strs: list[str] = ["NP_DATETIME_FORM", "NP_DATETIME_UNITS", "NP_TIMEDELTA_FORM"]
        self.bool: list[str] = [
            "HAVE_COVERAGE",
            "HAVE_DS",
            "HAVE_H5PY",
            "HAVE_MPL",
            "HAVE_NUMPY",
            "HAVE_PANDAS",
            "HAVE_PYTEST",
            "HAVE_SCIPY",
            "IS_WINDOWS",
        ]
        self.xtra: list[str] = ["NP_ONE_DAY", "NP_ONE_HOUR", "NP_ONE_MINUTE", "NP_ONE_SECOND"]
        self.nats: list[str] = ["NP_NAT"]
        self.master = set(self.ints) | set(self.strs) | set(self.bool) | set(self.xtra) | set(self.nats)

    def test_values(self) -> None:
        # confirm that all the expected values exist and have the correct type
        for key in self.ints:
            self.assertTrue(isinstance(getattr(dcs, key), int))
        for key in self.strs:
            self.assertTrue(isinstance(getattr(dcs, key), str))
        for key in self.bool:
            self.assertTrue(isinstance(getattr(dcs, key), bool))
        for key in self.xtra:
            if dcs.HAVE_NUMPY:
                self.assertTrue(isinstance(getattr(dcs, key), np.timedelta64))
            else:
                self.assertIsNone(getattr(dcs, key))
        for key in self.nats:
            if dcs.HAVE_NUMPY:
                self.assertTrue(isinstance(getattr(dcs, key), np.datetime64))
            else:
                self.assertIsNone(getattr(dcs, key))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_np_times(self) -> None:
        self.assertEqual(dcs.NP_ONE_SECOND.astype(np.int64), 10**9)
        self.assertEqual(dcs.NP_ONE_MINUTE.astype(np.int64), 60 * 10**9)
        self.assertEqual(dcs.NP_ONE_HOUR.astype(np.int64), 3600 * 10**9)
        self.assertEqual(dcs.NP_ONE_DAY.astype(np.int64), 86400 * 10**9)
        self.assertTrue(np.isnat(dcs.NP_NAT))

    def test_missing(self) -> None:
        for field in vars(dcs.constants):
            if field.isupper():
                self.assertTrue(field in self.master, "Test is missing: {}".format(field))


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
