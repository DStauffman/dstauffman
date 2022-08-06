r"""
Example script for creating your own class to save to HDF5 files with datetimes.

Notes
-----
#.  Written by David C. Stauffer in July 2022.
"""

#%% Imports
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, TYPE_CHECKING

import numpy as np

from dstauffman import (
    compare_two_classes,
    convert_datetime_to_np,
    Frozen,
    get_tests_dir,
    NP_DATETIME_FORM,
    NP_DATETIME_UNITS,
    NP_ONE_SECOND,
    SaveAndLoad,
)

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg

#%% Classes
class Results(Frozen, metaclass=SaveAndLoad):
    """Custom class using datetime64's that can be saved and loaded from HDF5 files."""

    # fmt: off
    load: ClassVar[Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5")], Results]]  # noqa: F821
    save: Callable[[Results, Optional[Path], DefaultNamedArg(bool, "use_hdf5")], None]  # noqa: F821
    # fmt: on

    def __init__(self, num: float = 0, date_zero: Optional[datetime.datetime] = None):
        self.time = np.arange(num)
        self.data = np.random.rand(*self.time.shape)
        self.date = np.full(self.time.shape, np.datetime64("NaT", NP_DATETIME_UNITS))
        if num > 0:
            assert date_zero is not None
            self.date[:] = convert_datetime_to_np(date_zero) + (NP_ONE_SECOND * self.time).astype(np.int64)

    def _save_convert_hdf5(self) -> Dict[str, Any]:
        self.date = self.date.astype(np.int64)
        return {}

    def _save_restore_hdf5(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        self.date = self.date.astype(NP_DATETIME_FORM)


#%% Main function
if __name__ == "__main__":
    # inputs
    num_pts = 100.0
    origin = datetime.datetime(2022, 7, 1)
    filename = get_tests_dir() / "temp_results.hdf5"

    # create original structure and save to disk
    results1 = Results(num_pts, origin)
    results1.save(filename)

    # reload into a new instance from the saved results
    results2 = Results.load(filename)

    # make sure everything matches
    compare_two_classes(results1, results2, names=["results1", "results2"], ignore_callables=True)

    # delete the temporary file
    filename.unlink()
