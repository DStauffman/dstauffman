r"""
Contains binary manipulation routines.

Notes
-----
#.  Written by David C. Stauffer in March 2023.
"""

# %% Imports
import doctest
import unittest


# %% Functions - _pad_string
def _pad_string(text: str, sep: int = 4, sep_char: str = " ") -> str:
    r"""
    Pads a string every sep characters, right justified.

    Examples
    --------
    >>> from dstauffman.binary import _pad_string
    >>> text = "11000011110000"
    >>> padded = _pad_string(text)
    >>> print(padded)
    11 0000 1111 0000

    """
    if sep == 0:
        return text
    if sep < 0:
        raise ValueError("sep must be greater than zero.")
    num = len(text)
    it = reversed(range(1, num + 1))
    mod = num % sep
    temp = ["".join((text[num - next(it)] for _ in range(mod)))] if mod != 0 else []
    for _ in range(num // sep):
        temp.append("".join((text[num - next(it)] for __ in range(sep))))
    return sep_char.join(temp)


# %% Functions - int2bin
def int2bin(num: int, pad: int = 0, sep: int = 0, pad_char: str = "0", sep_char: str = "_") -> str:
    r"""
    Convert a given integer to a binary string representation.

    Parameters
    ----------
    num : int
        number to convert

    Returns
    -------
    bin_str : str
        equivalent binary string

    Notes
    -----
    #.  Written by David C. Stauffer in March 2023.

    Examples
    --------
    >>> from dstauffman import int2bin
    >>> num = 0b0001_1111
    >>> bin_str = int2bin(num)
    >>> print(bin_str)
    0b11111

    """
    bin_start = "0b"
    bin_str = bin(num)
    if pad != 0:
        if (str_len := len(bin_str) - 2) < pad:
            bin_str = bin_start + pad_char * (pad - str_len) + bin_str[2:]
    if sep != 0:
        bin_str = bin_start + _pad_string(bin_str[2:], sep=sep, sep_char=sep_char)
    return bin_str


# %% Functions - int2hex
def int2hex(num: int, pad: int = 0, sep: int = 0, pad_char: str = "0", sep_char: str = "_", upper: bool = False) -> str:
    r"""
    Convert a given integer to a hexadecimal string representation.

    Parameters
    ----------
    num : int
        number to convert

    Returns
    -------
    hex_str : str
        equivalent hexadecimal string

    Notes
    -----
    #.  Written by David C. Stauffer in March 2023.

    Examples
    --------
    >>> from dstauffman import int2hex
    >>> num = 0x0001_abcd
    >>> hex_str = int2hex(num)
    >>> print(hex_str)
    0x1abcd

    """
    hex_start = "0x"
    hex_str = hex(num)
    if upper:
        hex_str = hex_start + hex_str[2:].upper()
    if pad != 0:
        if (str_len := len(hex_str) - 2) < pad:
            hex_str = hex_start + pad_char * (pad - str_len) + hex_str[2:]
    if sep != 0:
        hex_str = hex_start + _pad_string(hex_str[2:], sep=sep, sep_char=sep_char)
    return hex_str


# %% Functions - read_bit_stream
def _read_bit_stream():
    pass  # TODO: write this


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_binary", exit=False)
    doctest.testmod(verbose=False)
