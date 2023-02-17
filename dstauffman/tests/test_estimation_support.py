r"""
Test file for the `support` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Incorporated into dstauffman by David C. Stauffer in July 2020.
"""

# %% Imports
from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING, Union
import unittest

from dstauffman import FixedDict, Frozen, HAVE_NUMPY
import dstauffman.estimation as estm

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _I = np.typing.NDArray[np.int_]
    _N = np.typing.NDArray[np.float64]


# %% Classes - _Config
class _Config(Frozen):
    r"""
    High level model configuration parameters.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Config
    >>> config = _Config()

    """

    def __init__(self) -> None:
        # log level for how much information to display while running
        self.log_level: int = logging.INFO

        # output folder/files
        self.output_folder: str = ""
        self.output_results: str = "results_model.hdf5"
        self.output_params: str = "results_param.pkl"

        # Whether to save the final state information to disk
        self.save_final: bool = False

        # randomness
        self.repeatable_randomness: bool = False
        self.repeatable_seed: int = 1000
        self.repeatable_offset: int = 0

        # parallelization
        self.use_parfor: bool = False
        self.max_cores: int = 4

        # default colormap
        self.colormap: str = "Paired"  #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'


class _Model(Frozen):
    r"""
    Example model parameters.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Model
    >>> model = _Model()

    """

    def __init__(self) -> None:
        self.field1: int = 1
        self.field2: Union[_I, List[int]] = np.array([1, 2, 3]) if HAVE_NUMPY else [1, 2, 3]
        self.field3: Dict[str, Union[int, float, _N, List[float]]] = (
            {"a": 5, "b": np.array([1.5, 2.5, 10.0])} if HAVE_NUMPY else {"a": 5, "b": [1.5, 2.5, 10.0]}
        )
        self.field4: Dict[str, Any] = FixedDict()
        self.field4["new"] = np.array([3, 4, 5]) if HAVE_NUMPY else [3, 4, 5]
        self.field4["old"] = "4 - 6"
        self.field4.freeze()


# %% Classes - _Parameters
class _Parameters(Frozen):
    r"""
    Example wrapper parameters master class.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()

    """

    def __init__(self) -> None:
        self.config = _Config()
        self.model = _Model()
        self.models: List[_Model] = [_Model(), _Model()]
        self.models[0].field1 = 100
        self.models[1].field1 = 200
        self.models[1].field2[2] = 300
        self.models[1].field3["a"] = 500.0
        self.models[1].field4["new"][1] = 444


# %% estimation._get_sub_level
class Test_estimation__get_sub_level(unittest.TestCase):
    r"""
    Tests the estimation._get_sub_level function with the following cases:
        TBD
    """
    pass  # TODO: write this


# %% estimation._check_valid_param_name
class Test_estimation__check_valid_param_name(unittest.TestCase):
    r"""
    Tests the estimation._check_valid_param_name function with the following cases:
        Nominal
        Dictionary parameter
        Bad param name
        Bad attribute name
        Element of vector
    """
    param: _Parameters

    @classmethod
    def setUpClass(cls) -> None:
        cls.param = _Parameters()

    def test_nominal(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.config.log_level")
        self.assertTrue(is_valid)

    def test_dictionary(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field3['a']")
        self.assertTrue(is_valid)

    def test_bad_name1(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "not_param.config.log_level")
        self.assertFalse(is_valid)

    def test_bad_name2(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.config.logless_level")
        self.assertFalse(is_valid)

    def test_vector_element(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field2[1]")
        self.assertTrue(is_valid)

    def test_array_in_dict_name(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field3['b'][2]")
        self.assertTrue(is_valid)
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field4['new'][0]")
        self.assertTrue(is_valid)

    def test_list_vector_element(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[0].field2[1]")
        self.assertTrue(is_valid)

    def test_list_array_in_dict_name(self) -> None:
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[0].field3['b'][2]")
        self.assertTrue(is_valid)
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[1].field4['new'][0]")
        self.assertTrue(is_valid)


# %% estimation.get_parameter
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_get_parameter(unittest.TestCase):
    r"""
    Tests the estimation.get_parameter function with the following cases:
        Nominal (covers four cases in one)
    """

    def setUp(self) -> None:
        self.values = [-100, -2, -3, 44.0]
        self.names = [
            "param.config.log_level",
            "param.models[0].field1",
            "param.models[1].field2[2]",
            "param.models[1].field3['b'][1]",
        ]
        self.param = _Parameters()
        self.param.config.log_level = self.values[0]  # type: ignore[assignment]
        self.param.models[0].field1 = self.values[1]  # type: ignore[assignment]
        self.param.models[1].field2[2] = self.values[2]  # type: ignore[call-overload]
        self.param.models[1].field3["b"][1] = self.values[3]  # type: ignore[index]

    def test_nominal(self) -> None:
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.values)


# %% estimation.set_parameter
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_set_parameter(unittest.TestCase):
    r"""
    Tests the estimation.set_parameter function with the following cases:
        Nominal (covers four cases in one)
    """

    def setUp(self) -> None:
        self.orig = [20, 100, 300, 2.5]
        self.values = [-100, -2, -3, 44.0]
        self.names = [
            "param.config.log_level",
            "param.models[0].field1",
            "param.models[1].field2[2]",
            "param.models[1].field3['b'][1]",
        ]
        self.param = _Parameters()
        self.param.config.log_level = self.orig[0]  # type: ignore[assignment]
        self.param.models[0].field1 = self.orig[1]  # type: ignore[assignment]
        self.param.models[1].field2[2] = self.orig[2]  # type: ignore[call-overload]
        self.param.models[1].field3["b"][1] = self.orig[3]  # type: ignore[index]

    def test_nominal(self) -> None:
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.orig)
        estm.set_parameter(self.param, self.names, self.values)
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.values)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
