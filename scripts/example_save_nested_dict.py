"""Script to demonstrate the different interpolation options."""

#%% Imports
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, ClassVar, Optional, Set, TYPE_CHECKING

import numpy as np

import slog as lg

import dstauffman as dcs

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg

    _N = np.typing.NDArray[np.float64]


#%% Classes
class MyData(dcs.Frozen, metaclass=dcs.SaveAndLoad):
    """Class based version of data."""

    time: _N
    data: _N
    ver: str

    # fmt: off
    load: ClassVar[Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5"), DefaultNamedArg(bool, "convert_dates")], MyData]]  # noqa: F821
    save: Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5"), DefaultNamedArg(dict, "meta"), DefaultNamedArg(Optional[Set[str]], "exclusions")], None]  # noqa: F821
    # fmt: on

    def __init__(self, num: int = 0, ver: str = ""):
        self.time = np.empty(num)
        self.data = np.empty(num)
        self.ver = ver


class MyCollection(dict, metaclass=dcs.SaveAndLoad):  # type: ignore[misc]
    """Class based wrapper that is just a simple dictionary."""

    # fmt: off
    load: ClassVar[Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5"), DefaultNamedArg(bool, "convert_dates")], MyData]]  # noqa: F821
    save: Callable[[Optional[Path], DefaultNamedArg(bool, "use_hdf5"), DefaultNamedArg(dict, "meta"), DefaultNamedArg(Optional[Set[str]], "exclusions")], None]  # noqa: F821
    # fmt: on


#%% Script
if __name__ == "__main__":
    folder = dcs.get_tests_dir()
    num_pts = 5
    first = {"time": [1, 2, 3, 4, 5], "data": [0, 0.5, 1.0, 1.5, 2], "ver": "1.0"}
    second = {"time": [10, 20, 30, 40, 50], "data": [1.0, 0.5, 0.0, 0.5, 1.0]}

    #%% Example 1
    data1 = copy.deepcopy(first)
    filename1 = folder / "test_file1.hdf5"
    meta = {"num_pts": num_pts}
    exclusions = {
        "ver",
    }
    dcs.save_hdf5(data1, filename1, meta=meta, exclusions=exclusions)

    (out1, meta1) = dcs.load_hdf5(None, filename1, return_meta=True)
    out_dict1 = {k: v for k, v in vars(out1).items() if not lg.is_dunder(k)}

    dcs.compare_two_dicts(meta, meta1, names=["meta", "meta1"])
    dcs.compare_two_dicts(data1, out_dict1, exclude={"ver",}, names=["data1", "out1"])  # fmt: skip

    #%% Example 2
    data2 = {}
    data2["first"] = copy.deepcopy(first)
    data2["second"] = copy.deepcopy(second)
    filename2 = folder / "test_file2.hdf5"
    dcs.save_hdf5(data2, filename2, meta=meta, exclusions=exclusions)

    (out2, meta2) = dcs.load_hdf5(None, filename2, return_meta=True)
    out_dict2 = {k: v for k, v in vars(out2).items() if not lg.is_dunder(k)}
    for key, value in out_dict2.items():
        out_dict2[key] = {k: v for k, v in vars(value).items() if not lg.is_dunder(k)}

    dcs.compare_two_dicts(data2, out_dict2, exclude={"ver",}, names=["data2", "out2"])  # fmt: skip

    #%% Example 3
    data3 = MyData(num_pts)
    data3.time[:] = first["time"]
    data3.data[:] = first["data"]
    data3.ver = first["ver"]  # type: ignore[assignment]
    filename3 = folder / "test_file3.hdf5"
    data3.save(filename3, meta=meta, exclusions=exclusions)

    (out3, meta3) = dcs.load_hdf5(None, filename3, return_meta=True)

    dcs.compare_two_classes(
        data3, out3, exclude={"ver", "load", "save", "pprint"}, names=["data3", "out3"], ignore_callables=True
    )

    #%% Example 4
    data4 = MyCollection({"first": MyData(num_pts), "second": MyData(num_pts)})
    data4["first"].time[:] = first["time"]
    data4["first"].data[:] = first["data"]
    data4["first"].ver = first["ver"]
    data4["second"].time[:] = second["time"]
    data4["second"].data[:] = second["data"]
    filename4 = folder / "test_file4.hdf5"
    data4.save(filename4, meta=meta, exclusions=exclusions)

    out4 = dcs.load_hdf5(None, filename4)

    for key, value in data4.items():
        dcs.compare_two_classes(
            value, getattr(out4, key), exclude={"ver", "load", "save", "pprint"}, names=[f"data4.{key}", f"out4.{key}"]
        )

    #%% Delete files
    filename1.unlink(missing_ok=True)
    filename2.unlink(missing_ok=True)
    filename3.unlink(missing_ok=True)
    filename4.unlink(missing_ok=True)
