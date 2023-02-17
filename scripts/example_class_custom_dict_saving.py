r"""
Example script for creating your own class to save to HDF5 files with datetimes.

Notes
-----
#.  Written by David C. Stauffer in July 2022.
"""

# %% Imports
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from dstauffman import (
    compare_two_classes,
    convert_datetime_to_np,
    Frozen,
    get_tests_dir,
    NP_DATETIME_UNITS,
    NP_ONE_SECOND,
    SaveAndLoad,
)

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg

    _N = np.typing.NDArray[np.float64]
    _D = np.typing.NDArray[np.datetime64]


# %% Classes
class Results(Frozen, metaclass=SaveAndLoad):
    """Custom class using datetime64's that can be saved and loaded from HDF5 files."""

    time: _N
    data: _N
    date: _D
    name: str
    meta: Dict[str, str]

    # fmt: off
    load: ClassVar[Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5"), DefaultNamedArg(bool, "convert_dates")], Results]]  # noqa: F821
    save: Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5")], None]  # noqa: F821
    # fmt: on

    def __init__(self, num: float = 0, date_zero: Optional[datetime.datetime] = None, name: str = ""):
        self.time = np.arange(num)
        self.data = np.random.rand(*self.time.shape)
        self.date = np.full(self.time.shape, np.datetime64("NaT", NP_DATETIME_UNITS))
        self.name = name
        self.meta = {"who": "cares"}
        if num > 0:
            assert date_zero is not None
            self.date[:] = convert_datetime_to_np(date_zero) + (NP_ONE_SECOND * self.time).astype(np.int64)

    @staticmethod
    def _datetime_fields() -> Tuple[str]:
        return ("date",)

    @staticmethod
    def _string_fields() -> Tuple[str]:
        return ("name",)

    @staticmethod
    def _exclude_fields() -> Tuple[str]:
        return ("meta",)


# %% Main function
if __name__ == "__main__":
    # inputs
    num_pts = 100.0
    origin = datetime.datetime(2022, 7, 1)
    use_hdf5 = True
    filename = get_tests_dir() / "temp_results.hdf5"

    # create original structure and save to disk
    results1 = Results(num_pts, origin, name="Text")
    results1.save(filename, use_hdf5=use_hdf5)

    # reload into a new instance from the saved results
    results2 = Results.load(filename, convert_dates=True, use_hdf5=use_hdf5)

    # make sure everything matches
    compare_two_classes(results1, results2, names=["results1", "results2"], ignore_callables=True)

    # delete the temporary file
    if not use_hdf5:
        filename = filename.with_suffix(".pkl")
    filename.unlink()
