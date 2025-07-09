r"""
Test file for the `multipass` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.

"""

# %% Imports
from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _N = NDArray[np.floating]


# %% Supporting functions
def _model_wrapper(x: _N | None, y: _N | None) -> _N:
    if x is None:
        raise ValueError("Bad value for x")
    if y is None:
        raise RuntimeError("Bad value for y")  # pragma: no cover
    return x + np.sin(x) + np.cos(y * 2)  # type: ignore[no-any-return]


# %% MultipassExceptionWrapper
class Test_MultipassExceptionWrapper(unittest.TestCase):
    r"""
    Tests the MultipassExceptionWrapper function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% parfor_wrapper
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_parfor_wrapper(unittest.TestCase):
    r"""
    Tests the parfor_wrapper function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.temp = np.linspace(0, 2 * np.pi)
        self.args = (
            (self.temp.copy(), self.temp.copy()),
            (self.temp + 2 * np.pi, self.temp.copy()),
            (self.temp + 4 * np.pi, self.temp.copy()),
        )
        self.max_cores = 6

    def test_not_parallel(self) -> None:
        results = dcs.parfor_wrapper(_model_wrapper, self.args, use_parfor=False, max_cores=self.max_cores)
        np.testing.assert_array_equal(results[0], _model_wrapper(self.temp, self.temp))

    def test_parallel(self) -> None:
        # TODO: check if this works with numpy, but without tblib installed
        results = dcs.parfor_wrapper(_model_wrapper, self.args, use_parfor=True, max_cores=self.max_cores)
        np.testing.assert_array_equal(results[0], _model_wrapper(self.temp, self.temp))

    def test_not_parallel_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.parfor_wrapper(_model_wrapper, ((self.temp, self.temp), (None, None)), use_parfor=False)
        self.assertEqual(str(context.exception), "Bad value for x")

    def test_parallel_error(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            dcs.parfor_wrapper(_model_wrapper, ((self.temp, self.temp), (self.temp, None)), use_parfor=True)
        self.assertEqual(str(context.exception), "Bad value for y")


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
