r"""
Test file for the `binary` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2023.
"""

# %% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np


# %% binary._pad_string
class Test_binary__pad_string(unittest.TestCase):
    r"""
    Tests the binary._pad_string function with the following cases:
        Nominal
        Evenly divided
        Optional arguments
        Empty case
        Zero sep
        Negative sep (raises ValueError)
    """

    def test_nominal(self) -> None:
        text = dcs.binary._pad_string("11000011110000")
        self.assertEqual(text, "11 0000 1111 0000")

    def test_even(self) -> None:
        text = dcs.binary._pad_string("0123456789ab")
        self.assertEqual(text, "0123 4567 89ab")

    def test_options(self) -> None:
        text = dcs.binary._pad_string("00123456789abcdef", sep=2, sep_char="-")
        self.assertEqual(text, "0-01-23-45-67-89-ab-cd-ef")

    def test_empty(self) -> None:
        text = dcs.binary._pad_string("")
        self.assertEqual(text, "")

    def test_zero_pad(self) -> None:
        text = dcs.binary._pad_string("text", sep=0)
        self.assertEqual(text, "text")

    def test_negative(self) -> None:
        with self.assertRaises(ValueError):
            dcs.binary._pad_string("text", sep=-2)


# %% int2bin
class Test_int2bin(unittest.TestCase):
    r"""
    Tests the int2bin function with the following cases:
        Default
        Padded
        Separated
        Different pad character
        Different separation character
    """

    def test_default(self) -> None:
        bin_str = dcs.int2bin(0b0001_1010)
        self.assertEqual(bin_str, "0b11010")

    def test_pad(self) -> None:
        bin_str = dcs.int2bin(0b0101_0000, pad=12)
        self.assertEqual(bin_str, "0b000001010000")

    def test_sep(self) -> None:
        bin_str = dcs.int2bin(0b110011, sep=2)
        self.assertEqual(bin_str, "0b11_00_11")

    def test_pad_char(self) -> None:
        bin_str = dcs.int2bin(15, pad=6, pad_char=" ")
        self.assertEqual(bin_str, "0b  1111")

    def test_sep_char(self) -> None:
        bin_str = dcs.int2bin(128, sep=4, sep_char=" ")
        self.assertEqual(bin_str, "0b1000 0000")


# %% int2hex
class Test_int2hex(unittest.TestCase):
    r"""
    Tests the int2hex function with the following cases:
        Default
        Padded
        Separated
        Different pad character
        Different separation character
        Upper case letters
    """

    def test_default(self) -> None:
        hex_str = dcs.int2hex(0x0001_ABCD)
        self.assertEqual(hex_str, "0x1abcd")

    def test_pad(self) -> None:
        hex_str = dcs.int2hex(0x0123_4567, pad=12)
        self.assertEqual(hex_str, "0x000001234567")

    def test_sep(self) -> None:
        hex_str = dcs.int2hex(0x11AC11, sep=2)
        self.assertEqual(hex_str, "0x11_ac_11")

    def test_pad_char(self) -> None:
        hex_str = dcs.int2hex(254, pad=6, pad_char=" ")
        self.assertEqual(hex_str, "0x    fe")

    def test_sep_char(self) -> None:
        hex_str = dcs.int2hex(65535, sep=3, sep_char=" ")
        self.assertEqual(hex_str, "0xf fff")

    def test_upper(self) -> None:
        hex_str = dcs.int2hex(0xFEDCBA0, upper=True, sep=4)
        self.assertEqual(hex_str, "0xFED_CBA0")


# %% split_bits
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_split_bits(unittest.TestCase):
    r"""
    Tests the split_bits function with the following cases:
        TBD
    """

    def test_nominal(self) -> None:
        bits = dcs.split_bits(np.array([0, 1, 2**16, 2**32 - 1], dtype=np.uint32))
        exp = np.zeros((32, 4), dtype=bool)
        exp[0, 1] = True
        exp[16, 2] = True
        exp[:, 3] = True
        np.testing.assert_array_equal(bits, exp)


# %% read_bit_stream
class Test_read_bit_stream(unittest.TestCase):
    r"""
    Tests the read_bit_stream function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
