r"""
Test file for the `weather` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in November 2023.
"""

# %% Imports
import unittest

from dstauffman import get_data_dir, HAVE_NUMPY, HAVE_PANDAS
import dstauffman.aerospace as space  # noqa: F401  # pylint: disable=unused-import

if HAVE_NUMPY:
    import numpy as np


# %% aerospace.read_tci_data
@unittest.skipIf(not HAVE_PANDAS, "Skipping due to missing pandas dependency.")
class Test_aerospace_read_tci_data(unittest.TestCase):
    r"""
    Tests the aerospace.read_tci_data function with the following cases:
        TBD
    """

    def test_nominal(self) -> None:
        filename = get_data_dir() / "tci_info.txt"
        tci_data = space.read_tci_data(filename)
        self.assertEqual(set(tci_data.keys()), {"Date", "TCI"})


# %% aerospace.read_kp_ap_etc_data
@unittest.skipIf(not HAVE_PANDAS, "Skipping due to missing pandas dependency.")
class Test_aerospace_read_kp_ap_etc_data(unittest.TestCase):
    r"""
    Tests the aerospace.read_kp_ap_etc_data function with the following cases:
        TBD
    """

    def test_nominal(self) -> None:
        filename = get_data_dir() / "Kp_ap_Ap_SN_F107_since_1932.txt"
        kp_data = space.read_kp_ap_etc_data(filename)
        self.assertIn("GMT", set(kp_data.keys()))


# %% aerospace.read_solar_cycles
@unittest.skipIf(not HAVE_PANDAS, "Skipping due to missing pandas dependency.")
class Test_aerospace_read_solar_cycles(unittest.TestCase):
    r"""
    Tests the aerospace.read_solar_cycles function with the following cases:
        TBD
    """

    def test_nominal(self) -> None:
        filename = get_data_dir() / "Solar_Cycles.txt"
        solar_data = space.read_solar_cycles(filename)
        cycle_24_max = solar_data.Maximum[solar_data.Solar_Cycle == 24].values[0]
        self.assertEqual(cycle_24_max, np.datetime64("2014-04-01"))


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
