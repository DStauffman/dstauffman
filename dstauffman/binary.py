r"""
Contains binary manipulation routines.

Notes
-----
#.  Written by David C. Stauffer in March 2023.
"""

# %% Imports
from __future__ import annotations

import doctest
import os
import struct
from typing import BinaryIO, Optional, TYPE_CHECKING
import unittest

from dstauffman.constants import HAVE_NUMPY

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _B2 = np.typing.NDArray[np.bool_]  # 2D
    _I = np.typing.NDArray[np.int_]


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


# %% Functions - split_bits
def split_bits(x: _I, num_bits: Optional[int] = None) -> _B2:
    r"""
    Split integers into rows of true/false by bit position.

    Examples
    --------
    >>> from dstauffman import split_bits
    >>> import numpy as np
    >>> x = np.array([0, 1, 2**16, 2**32 -1 ], dtype=np.uint32)
    >>> bits = split_bits(x)
    >>> print(bits[0, :])
    [False  True False  True]

    >>> print(bits[16, :])
    [False False  True  True]

    >>> print(bits[31, :])
    [False False False  True]

    """
    if num_bits is None:
        num_bits = np.ceil(np.log2(x.max())).astype(int)
    bits = np.zeros((num_bits, x.size), dtype=bool)
    for i in range(num_bits):
        bits[i, :] = np.bitwise_and(np.right_shift(x, i), 1)
    return bits


# %% Functions - read_bit_stream
def read_bit_stream(file: BinaryIO, num_lines: int = 10, reset: bool = False, endian: str = "big") -> None:
    r"""
    Reads in a stream of bits and displays potential outputs.

    Designed as a debug tool for use exploring unknown binary data.

    Parameters
    ----------
    fid :
        data stream
    num_lines : int
        Number of lines (64 bits at a time) to read
    reset : bool, default is False
        Whether to reset back to where you started

    Examples
    --------
    >>> from dstauffman import read_bit_stream, get_tests_dir
    >>> import numpy as np
    >>> filename = get_tests_dir() / "test_big_endian.bin"
    >>> with open(filename, "rb") as file:
    ...     read_bit_stream(file, num_lines=10)
    00000000000000000000000000000011 11111111111111111111111111111101
    uint32 = [3, 4294967293]
    int32  = [3, -3]
    float  = [4.203895392974451e-45, nan]
    double = 8.4879831624e-314
    00000000000000000000000000000011 00000000000000010000000000000011
    uint32 = [3, 65539]
    int32  = [3, 65539]
    float  = [4.203895392974451e-45, 9.183970005338419e-41]
    double = 6.3660197535e-314
    00000000000000000000000000000000 00111111110000000000000000000000
    uint32 = [0, 1069547520]
    int32  = [0, 1069547520]
    float  = [0.0, 1.5]
    double = 5.28426686e-315
    11000000000101010101010101010101 01000000010010010000111111011011
    uint32 = [3222623573, 1078530011]
    int32  = [-1072343723, 1078530011]
    float  = [-2.3333332538604736, 3.1415927410125732]
    double = -5.333333019694659
    00000000000000000000000000000000 00000000000000000000000000000000
    uint32 = [0, 0]
    int32  = [0, 0]
    float  = [0.0, 0.0]
    double = 0.0
    10111111111110000000000000000000 00000000000000000000000000000000
    uint32 = [3220701184, 0]
    int32  = [-1074266112, 0]
    float  = [-1.9375, 0.0]
    double = -1.5
    01000000000010010010000111111011 01010100010001000010110100011000
    uint32 = [1074340347, 1413754136]
    int32  = [1074340347, 1413754136]
    float  = [2.1426990032196045, 3370280550400.0]
    double = 3.141592653589793
    01000000000001011011111100001010 10001011000101000101011101101001
    uint32 = [1074118410, 2333366121]
    int32  = [1074118410, -1961601175]
    float  = [2.089785099029541, -2.8569523269651966e-32]
    double = 2.718281828459045
    The end of the file was reached.

    """
    if endian == "native":
        c = "="
    elif endian == "little":
        c = "<"
    elif endian == "big":
        c = ">"
    else:
        raise ValueError(f'Unexpected value for endian of "{endian}".')
    for _ in range(num_lines):
        # read the next chunk of bytes
        byte_str = file.read(8)
        bytes_read = len(byte_str)
        if bytes_read == 0:
            # reached end of file
            print("The end of the file was reached.")
            break
        if bytes_read < 8:
            # partial packet
            print(f"A final partial packet of {bytes_read} bytes was found.")
            if reset:
                file.seek(-bytes_read, os.SEEK_CUR)
            break
        # read as uint32
        uin1, uin2 = struct.unpack(c + "II", byte_str)
        # read as int32
        int1, int2 = struct.unpack(c + "ii", byte_str)
        # read as float32
        flt1, flt2 = struct.unpack(c + "ff", byte_str)
        # read as double
        dble = struct.unpack(c + "d", byte_str)[0]
        # display results
        bin_str = int2bin(int(byte_str.hex(), 16), pad=64, sep=32, sep_char=" ")[2:]
        print(bin_str)
        print("uint32 = ", end="")
        print([uin1, uin2], sep=" ")
        print("int32  = ", end="")
        print([int1, int2], sep=" ")
        print("float  = ", end="")
        print([flt1, flt2], sep=" ")
        print("double = ", end="")
        print(dble)
    if reset:
        file.seek(-8 * num_lines, os.SEEK_CUR)


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_binary", exit=False)
    doctest.testmod(verbose=False)
