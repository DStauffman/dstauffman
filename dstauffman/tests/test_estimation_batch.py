r"""
Test file for the `batch` module of the "dstauffman.estimation" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Renamed from bpe.py to estimation.py by David C. Stauffer in May 2020.
"""

# %% Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING
import unittest
from unittest.mock import Mock, patch

from slog import capture_output, LogLevel

from dstauffman import compare_two_classes, Frozen, get_tests_dir, HAVE_H5PY, HAVE_NUMPY
import dstauffman.estimation as estm

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _I = NDArray[np.int_]
    _N = NDArray[np.floating]


# %% Setup for testing
# Classes - SimParams
class SimParams(Frozen):
    r"""Simulation model parameters."""

    def __init__(self, time: _N, *, magnitude: float, frequency: float, phase: float) -> None:
        # fmt: off
        self.time      = time
        self.magnitude = magnitude
        self.frequency = frequency
        self.phase     = phase
        # fmt: on

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        if not isinstance(other, type(self)):
            return False
        for key in vars(self):
            if np.any(getattr(self, key) != getattr(other, key)):
                return False
        return True


# Functions - _get_truth_index
def _get_truth_index(results_time: _N, truth_time: _N) -> tuple[_I, _I]:
    r"""Finds the indices to the truth data from the results time."""
    # Hard-coded values
    precision = 1e-7
    # find the indices to truth
    ix_truth = np.flatnonzero((truth_time >= results_time[0] - precision) & (truth_time <= results_time[-1] + precision))
    # find the indices to results (in case truth isn't long enough)
    ix_results = np.flatnonzero(results_time <= truth_time[-1] + precision)
    # return the indices
    return (ix_truth, ix_results)


# Functions - sim_model
def sim_model(sim_params: SimParams) -> _N:
    r"""Simple example simulation model."""
    return sim_params.magnitude * np.sin(  # type: ignore[no-any-return]
        2 * np.pi * sim_params.frequency * sim_params.time / 1000 + sim_params.phase * np.pi / 180
    )


# Functions - truth
def truth(time: _N, magnitude: float = 5.0, frequency: float = 10.0, phase: float = 90.0) -> _N:
    r"""Simple example truth data."""
    return magnitude * np.sin(2 * np.pi * frequency * time / 1000 + phase * np.pi / 180)  # type: ignore[no-any-return]  # pragma: no cover


# Functions - cost_wrapper
def cost_wrapper(results_data: _N, *, results_time: _N, truth_time: _N, truth_data: _N, sim_params: SimParams) -> _N:
    r"""Example Cost wrapper for the model."""
    # Pull out overlapping time points and indices
    (ix_truth, ix_results) = _get_truth_index(results_time, truth_time)
    sub_truth = truth_data[ix_truth]
    sub_result = results_data[ix_results]

    # calculate the innovations
    innovs: _N = sub_result - sub_truth
    return innovs


# Functions - get_parameter
def get_parameter(sim_params: SimParams, *, names: list[str]) -> _N:
    r"""Simple example parameter getter."""
    num = len(names)
    values = np.full(num, np.nan)
    for ix, name in enumerate(names):
        if hasattr(sim_params, name):
            values[ix] = getattr(sim_params, name)
        else:  # pragma: no cover
            raise ValueError('Bad parameter name: "{}".'.format(name))
    return values


# Functions - set_parameter
def set_parameter(sim_params: SimParams, *, names: list[str], values: _N) -> None:
    r"""Simple example parameter setter."""
    num = len(names)
    assert len(values) == num, "Names and Values must have the same length."
    for ix, name in enumerate(names):
        if hasattr(sim_params, name):
            setattr(sim_params, name, values[ix])
        else:  # pragma: no cover
            raise ValueError('Bad parameter name: "{}".'.format(name))


# %% estimation.OptiOpts
class Test_estimation_OptiOpts(unittest.TestCase):
    r"""
    Tests the estimation.OptiOpts class with the following cases:
        Initialization
        Equality
        Inequality
    """

    def test_init(self) -> None:
        opti_opts = estm.OptiOpts()
        self.assertTrue(isinstance(opti_opts, estm.OptiOpts))

    def test_equality(self) -> None:
        opti_opts1 = estm.OptiOpts()
        opti_opts2 = estm.OptiOpts()
        self.assertEqual(opti_opts1, opti_opts2)

    def test_inequality(self) -> None:
        opti_opts1 = estm.OptiOpts()
        opti_opts2 = estm.OptiOpts()
        opti_opts2.grow_radius = 5.5
        self.assertNotEqual(opti_opts1, opti_opts2)

    def test_inequality2(self) -> None:
        opti_opts = estm.OptiOpts()
        self.assertNotEqual(opti_opts, 2)

    def test_pprint(self) -> None:
        opti_opts = estm.OptiOpts()
        with capture_output() as ctx:
            opti_opts.pprint()
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "OptiOpts")
        self.assertEqual(lines[1], " model_func      = None")
        self.assertEqual(lines[-1], " max_cores       = None")


# %% estimation.OptiParam
class Test_estimation_OptiParam(unittest.TestCase):
    r"""
    Tests the estimation.OptiParam class with the following cases:
        Initialization
        Equality
        Inequality
        Get array (x5)
        Get names
    """

    def test_init(self) -> None:
        opti_param = estm.OptiParam("test")
        self.assertTrue(isinstance(opti_param, estm.OptiParam))

    def test_equality(self) -> None:
        opti_param1 = estm.OptiParam("test")
        opti_param2 = estm.OptiParam("test")
        self.assertEqual(opti_param1, opti_param2)

    def test_inequality(self) -> None:
        opti_param1 = estm.OptiParam("test")
        opti_param2 = estm.OptiParam("test")
        opti_param2.min_ = 5.5
        self.assertNotEqual(opti_param1, opti_param2)

    def test_inequality2(self) -> None:
        opti_param = estm.OptiParam("test")
        self.assertNotEqual(opti_param, 2)

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_get_array(self) -> None:
        opti_param = estm.OptiParam("test")
        params = [opti_param, opti_param]
        best = estm.OptiParam.get_array(params)
        np.testing.assert_array_equal(best, np.array([np.nan, np.nan]))

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_get_array2(self) -> None:
        opti_param = estm.OptiParam("test")
        params = [opti_param, opti_param]
        values = estm.OptiParam.get_array(params, type_="min")
        np.testing.assert_array_equal(values, np.array([-np.inf, -np.inf]))

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_get_array3(self) -> None:
        opti_param = estm.OptiParam("test")
        params = [opti_param, opti_param]
        values = estm.OptiParam.get_array(params, type_="max")
        np.testing.assert_array_equal(values, np.array([np.inf, np.inf]))

    @unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_get_array4(self) -> None:
        opti_param = estm.OptiParam("test")
        params = [opti_param, opti_param]
        for this_type in ["best", "min", "min_", "max", "max_", "minstep", "typical"]:
            # just test that these all execute, but don't worry about values
            estm.OptiParam.get_array(params, type_=this_type)

    def test_get_array5(self) -> None:
        opti_param = estm.OptiParam("test")
        params = [opti_param, opti_param]
        with self.assertRaises(ValueError):
            estm.OptiParam.get_array(params, type_="bad_name")

    def test_get_names(self) -> None:
        opti_param1 = estm.OptiParam("test1")
        opti_param2 = estm.OptiParam("test2")
        params = [opti_param1, opti_param2]
        names = estm.OptiParam.get_names(params)
        self.assertEqual(names, ["test1", "test2"])

    def test_pprint(self) -> None:
        opti_param = estm.OptiParam("test")
        with capture_output() as ctx:
            opti_param.pprint()
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "OptiParam")
        self.assertEqual(lines[1], " name    = test")
        self.assertEqual(lines[2], " best    = nan")
        self.assertEqual(lines[3], " min_    = -inf")
        self.assertEqual(lines[4], " max_    = inf")
        self.assertEqual(lines[5], " minstep = 0.0001")
        self.assertEqual(lines[6], " typical = 1.0")


# %% estimation.BpeResults
class Test_estimation_BpeResults(unittest.TestCase):
    r"""
    Tests the estimation.BpeResults class with the following cases:
        Save (HDF5)
        Load (HDF5)
        str method
        pprint method
    """

    def setUp(self) -> None:
        self.bpe_results = estm.BpeResults()
        self.bpe_results.num_evals = 5
        self.filename = get_tests_dir() / "test_estimation_results.hdf5"
        self.filename2 = self.filename.with_suffix(".pkl")
        estm.batch.logger.setLevel(LogLevel.L0)

    @unittest.skipIf(not HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_save(self) -> None:
        self.bpe_results.save(self.filename)
        self.assertTrue(self.filename.is_file())

    @unittest.skipIf(not HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_load(self) -> None:
        self.bpe_results.save(self.filename)
        bpe_results = estm.BpeResults.load(self.filename)
        self.assertTrue(compare_two_classes(bpe_results, self.bpe_results, suppress_output=True))

    def test_str(self) -> None:
        with capture_output() as ctx:
            print(self.bpe_results)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "BpeResults")
        self.assertTrue(lines[1].startswith("  begin_params = None"))

    def test_pprint(self) -> None:
        self.bpe_results.param_names = ["a".encode("utf-8")]
        self.bpe_results.begin_params = [1.0]  # type: ignore[assignment]
        self.bpe_results.final_params = [2.0]  # type: ignore[assignment]
        with capture_output() as ctx:
            self.bpe_results.pprint()
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Initial cost: None")
        self.assertEqual(lines[1], "Initial parameters:")
        self.assertEqual(lines[2].strip(), "a = 1.0")
        self.assertEqual(lines[3], "Final cost: None")
        self.assertEqual(lines[4], "Final parameters:")
        self.assertEqual(lines[5].strip(), "a = 2.0")

    def test_pprint2(self) -> None:
        with capture_output() as ctx:
            self.bpe_results.pprint()
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "")

    def tearDown(self) -> None:
        self.filename.unlink(missing_ok=True)
        self.filename2.unlink(missing_ok=True)


# %% estimation.CurrentResults
class Test_estimation_CurrentResults(unittest.TestCase):
    r"""
    Tests the estimation.CurrentResults class with the following cases:
        Printing
    """

    def setUp(self) -> None:
        self.current_results = estm.CurrentResults()

    def test_printing(self) -> None:
        with capture_output() as ctx:
            print(self.current_results)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Current Results:")
        self.assertEqual(lines[1], "  Trust Radius: None")
        self.assertEqual(lines[2], "  Best Cost: None")
        self.assertEqual(lines[3], "  Best Params: None")


# %% estimation.batch._print_divider
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_batch__print_divider(unittest.TestCase):
    r"""
    Tests the estimation.batch._print_divider function with the following cases:
        With new line
        Without new line
    """

    def setUp(self) -> None:
        self.output = "******************************"

    def test_with_new_line(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        estm.batch._print_divider()
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L5, " ")
        mock_logger.log.assert_any_call(LogLevel.L5, "******************************")

    def test_no_new_line(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(LogLevel.L8)
        estm.batch._print_divider(new_line=False)
        mock_logger.log.assert_called_with(LogLevel.L5, "******************************")

    def test_alternative_level(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(LogLevel.L2)
        estm.batch._print_divider(level=LogLevel.L0)
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L0, " ")
        mock_logger.log.assert_any_call(LogLevel.L0, "******************************")

    def test_not_logging(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(LogLevel.L2)
        estm.batch._print_divider()
        self.assertEqual(mock_logger.log.call_count, 2)
        mock_logger.log.assert_any_call(LogLevel.L5, " ")
        mock_logger.log.assert_any_call(LogLevel.L5, "******************************")
        # TODO: how to test that this wouldn't log anything?


# %% estimation.batch._calc_sum_squares
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_batch__calc_sum_squares(unittest.TestCase):
    r"""
    Tests the estimation.batch._calc_sum_squares function with the following cases:
        sum of squares on just a scalar input
        normal sum of squares on vector input
        sum of squares on vector with axis specified
        sum of squares on vector with bad axis specified
        sum of squares on matrix without keeping dimensions, no axis
        sum of squares on matrix without keeping dimensions, axis 0
        sum of squares on matrix without keeping dimensions, axis 1
        sum of squares on matrix with keeping dimensions
        sum of squares on complex numbers
        sum of squares on complex numbers that would return a real if done incorrectly
    """

    def setUp(self) -> None:
        # fmt: off
        self.inputs1   = np.array([0, 1, 0, -1])
        self.outputs1  = 2
        self.inputs2   = [[0, 1, 0, -1], [1, 1, 1, 1]]
        self.outputs2a = 6
        self.outputs2b = np.array([1, 2, 1, 2])
        self.outputs2c = np.array([2, 4])
        self.outputs2d = np.array([[2], [4]])
        self.inputs3   = np.hstack((self.inputs1, np.nan))
        self.inputs4   = [[0, 0, np.nan], [1, np.nan, 1]]
        self.outputs4a = 2
        self.outputs4b = np.array([1, 0, 1])
        self.outputs4c = np.array([0, 2])
        # fmt: on

    def test_scalar_input(self) -> None:
        out = estm.batch._calc_sum_squares(-1.5)
        self.assertEqual(out, 1.5**2)

    def test_empty(self) -> None:
        out = estm.batch._calc_sum_squares([])
        self.assertTrue(np.isnan(out))

    def test_rss_series(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs1)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1a(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs1, axis=0)
        assert isinstance(out, np.int_)  # type: ignore[unreachable]
        self.assertAlmostEqual(out, self.outputs1)  # type: ignore[unreachable]

    def test_axis_drop1b(self) -> None:
        with self.assertRaises(ValueError):
            estm.batch._calc_sum_squares(self.inputs1, axis=1)

    def test_axis_drop2a(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs2)
        self.assertAlmostEqual(out, self.outputs2a)

    def test_axis_drop2b(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs2, axis=0, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2b)

    def test_axis_drop2c(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs2, axis=1, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2c)

    def test_axis_keep(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs2, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(out, self.outputs2d)

    def test_complex_sum_squares(self) -> None:
        out = estm.batch._calc_sum_squares(1.5j)
        self.assertEqual(out, 1.5**2)

    def test_complex_conj(self) -> None:
        out = estm.batch._calc_sum_squares(np.array([1 + 1j, 1 - 1j]))
        self.assertAlmostEqual(out, 4)

    def test_with_nans(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs3, ignore_nans=False)
        self.assertTrue(np.isnan(out))

    def test_ignore_nans1(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs3, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs1)

    def test_ignore_nans2(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs4, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs4a)

    def test_ignore_nans3(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs4, ignore_nans=True, axis=0)
        np.testing.assert_array_almost_equal(out, self.outputs4b)

    def test_ignore_nans4(self) -> None:
        out = estm.batch._calc_sum_squares(self.inputs4, ignore_nans=True, axis=1)
        np.testing.assert_array_almost_equal(out, self.outputs4c)

    def test_all_nans(self) -> None:
        x = np.full((4, 3), np.nan)
        out: int | float | _I | _N = estm.batch._calc_sum_squares(x, ignore_nans=True)
        self.assertTrue(np.isnan(out))
        out = estm.batch._calc_sum_squares(x, axis=0, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (3,))
        out = estm.batch._calc_sum_squares(x, axis=1, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4,))
        out = estm.batch._calc_sum_squares(x, axis=0, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (1, 3))
        out = estm.batch._calc_sum_squares(x, axis=1, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4, 1))


# %% estimation.batch._function_wrapper
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_batch__function_wrapper(unittest.TestCase):
    r"""
    Tests the estimation.batch._function_wrapper function with the following cases:
        Nominal
        Model args
        Cost args
    """

    def setUp(self) -> None:
        self.results = np.array([1, 2, np.nan])
        self.innovs = np.array([1, 2, 0])
        self.model_args: dict[str, Any] = {}
        self.cost_args: dict[str, Any] = {}
        self.model_func = lambda *args, **kwargs: np.array([1, 2, np.nan])
        self.cost_func = lambda *args, **kwargs: np.array([1, 2, np.nan])

    def test_nominal(self) -> None:
        (innovs, results) = estm.batch._function_wrapper(
            model_func=self.model_func,
            model_args=self.model_args,
            cost_func=self.cost_func,
            cost_args=self.cost_args,
            return_results=True,
        )
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_model_args(self) -> None:
        (innovs, results) = estm.batch._function_wrapper(
            model_func=self.model_func,
            model_args={"a": 5},
            cost_func=self.cost_func,
            cost_args=self.cost_args,
            return_results=True,
        )
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_cost_args(self) -> None:
        (innovs, results) = estm.batch._function_wrapper(
            model_func=self.model_func,
            model_args=self.model_args,
            cost_func=self.cost_func,
            cost_args={"a": 5},
            return_results=True,
        )
        np.testing.assert_array_equal(results, self.results)
        np.testing.assert_array_equal(innovs, self.innovs)

    def test_innov_only(self) -> None:
        innovs = estm.batch._function_wrapper(
            model_func=self.model_func, model_args=self.model_args, cost_func=self.cost_func, cost_args={"a": 5}
        )
        np.testing.assert_array_equal(innovs, self.innovs)


# %% estimation.batch._parfor_function_wrapper
class Test_estimation_batch__parfor_function_wrapper(unittest.TestCase):
    r"""
    Tests the estimation.batch._parfor_function_wrapper function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% estimation.batch._finite_differences
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_batch__finite_differences(unittest.TestCase):
    r"""
    Tests the estimation.batch._finite_differences function with the following cases:
        Nominal
        Normalized
    """

    def setUp(self) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        time       = np.arange(251.0)  # fmt: skip
        sim_params = SimParams(time, magnitude=3.5, frequency=12.0, phase=180.0)
        truth_time = np.arange(-10.0, 201.0)
        truth_data = 5 * np.sin(2 * np.pi * 10 * time / 1000 + 90 * np.pi / 180)

        # fmt: off
        self.opti_opts                = estm.OptiOpts()
        self.opti_opts.model_func     = sim_model
        self.opti_opts.model_args     = {"sim_params": sim_params}
        self.opti_opts.cost_func      = cost_wrapper
        self.opti_opts.cost_args      = {"results_time": time, "truth_time": truth_time, "truth_data": truth_data}
        self.opti_opts.get_param_func = get_parameter
        self.opti_opts.set_param_func = set_parameter
        self.opti_opts.output_folder  = None
        self.opti_opts.output_results = None
        self.opti_opts.params         = []
        # fmt: on

        # Parameters to estimate
        self.opti_opts.params.append(estm.OptiParam("magnitude", best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("frequency", best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("phase", best=180, min_=0, max_=360, typical=100, minstep=0.1))

        self.model_args = self.opti_opts.model_args

        self.bpe_results = estm.BpeResults()
        self.cur_results = estm.CurrentResults()

        # initialize current results
        self.cur_results.innovs = estm.batch._function_wrapper(
            model_func=self.opti_opts.model_func,
            cost_func=self.opti_opts.cost_func,
            model_args=self.model_args,
            cost_args=self.opti_opts.cost_args,
        )
        self.bpe_results.num_evals += 1
        self.cur_results.trust_rad = self.opti_opts.trust_radius
        self.cur_results.cost = 0.5 * estm.batch._calc_sum_squares(self.cur_results.innovs, ignore_nans=True)
        names = estm.OptiParam.get_names(self.opti_opts.params)
        self.cur_results.params = self.opti_opts.get_param_func(names=names, **self.model_args)

        # set relevant results variables
        self.bpe_results.param_names = [name.encode("utf-8") for name in names]
        self.bpe_results.begin_params = self.cur_results.params.copy()
        self.bpe_results.begin_innovs = self.cur_results.innovs.copy()
        self.bpe_results.begin_cost   = self.cur_results.cost  # fmt: skip
        self.bpe_results.costs.append(self.cur_results.cost)

        self.two_sided = False
        self.normalized = False

    def test_nominal(self, mock_logger: Mock) -> None:
        (jacobian, gradient, hessian) = estm.batch._finite_differences(
            self.opti_opts,
            self.model_args,
            self.bpe_results,
            self.cur_results,
            two_sided=self.two_sided,
            normalized=self.normalized,
        )
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3,))
        self.assertEqual(hessian.shape, (3, 3))

    def test_normalized(self, mock_logger: Mock) -> None:
        self.normalized = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(
            self.opti_opts,
            self.model_args,
            self.bpe_results,
            self.cur_results,
            two_sided=self.two_sided,
            normalized=self.normalized,
        )
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3,))
        self.assertEqual(hessian.shape, (3, 3))

    def test_two_sided(self, mock_logger: Mock) -> None:
        self.two_sided = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(
            self.opti_opts,
            self.model_args,
            self.bpe_results,
            self.cur_results,
            two_sided=self.two_sided,
            normalized=self.normalized,
        )
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3,))
        self.assertEqual(hessian.shape, (3, 3))

    def test_norm_and_two_sided(self, mock_logger: Mock) -> None:
        self.normalized = True
        self.two_sided = True
        (jacobian, gradient, hessian) = estm.batch._finite_differences(
            self.opti_opts,
            self.model_args,
            self.bpe_results,
            self.cur_results,
            two_sided=self.two_sided,
            normalized=self.normalized,
        )
        self.assertEqual(jacobian.shape, (201, 3))
        self.assertEqual(gradient.shape, (3,))
        self.assertEqual(hessian.shape, (3, 3))


# %% estimation.batch._levenberg_marquardt
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_batch__levenberg_marquardt(unittest.TestCase):
    r"""
    Tests the estimation.batch._levenberg_marquardt function with the following cases:
        with lambda_
        without lambda_
    """

    def setUp(self) -> None:
        self.jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.innovs = np.array([7.0, 8.0, 9.0])
        self.lambda_ = 5.0
        self.delta_param = np.array([-0.46825397, -1.3015873])

    def test_nominal(self) -> None:
        delta_param = estm.batch._levenberg_marquardt(self.jacobian, self.innovs, self.lambda_)
        np.testing.assert_array_almost_equal(delta_param, self.delta_param)

    def test_lambda_zero(self) -> None:
        b = -np.linalg.pinv(self.jacobian).dot(self.innovs)
        delta_param = estm.batch._levenberg_marquardt(self.jacobian, self.innovs, 0)
        np.testing.assert_array_almost_equal(delta_param, b)


# %% estimation.batch._predict_func_change
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_batch__predict_func_change(unittest.TestCase):
    r"""
    Tests the estimation.batch._predict_func_change function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.delta_param = np.array([1.0, 2.0])
        self.gradient = np.array([3.0, 4.0])
        self.hessian = np.array([[5.0, 2.0], [2.0, 5.0]])
        self.pred_change = 27.5

    def test_nominal(self) -> None:
        delta_func = estm.batch._predict_func_change(self.delta_param, self.gradient, self.hessian)
        self.assertEqual(delta_func, self.pred_change)


# %% estimation.batch._check_for_convergence
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_batch__check_for_convergence(unittest.TestCase):
    r"""
    Tests the estimation.batch._check_for_convergence function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        self.opti_opts: estm.OptiOpts = type("Class1", (object,), {"tol_cosmax_grad": 1, "tol_delta_step": 2, "tol_delta_cost": 3})  # type: ignore[assignment]
        self.cosmax = 10
        self.delta_step_len = 10
        self.pred_func_change = 10

    def test_not_converged(self, mock_logger: Mock) -> None:
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, self.pred_func_change)
        self.assertFalse(convergence)

    def test_convergence1(self, mock_logger: Mock) -> None:
        convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, self.delta_step_len, self.pred_func_change)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(
            LogLevel.L3, "Declare convergence because cosmax of %s <= options.tol_cosmax_grad of %s", 0.5, 1
        )

    def test_convergence2(self, mock_logger: Mock) -> None:
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, 1.5, self.pred_func_change)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(
            LogLevel.L3, "Declare convergence because delta_step_len of %s <= options.tol_delta_step of %s", 1.5, 2
        )

    def test_convergence3(self, mock_logger: Mock) -> None:
        convergence = estm.batch._check_for_convergence(self.opti_opts, self.cosmax, self.delta_step_len, -2.5)
        self.assertTrue(convergence)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(
            LogLevel.L3, "Declare convergence because abs(pred_func_change) of %s <= options.tol_delta_cost of %s", 2.5, 3
        )

    def test_convergence4(self, mock_logger: Mock) -> None:
        convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        self.assertTrue(convergence)
        self.assertEqual(mock_logger.log.call_count, 3)
        mock_logger.log.assert_any_call(
            LogLevel.L3, "Declare convergence because cosmax of %s <= options.tol_cosmax_grad of %s", 0.5, 1
        )
        mock_logger.log.assert_any_call(
            LogLevel.L3, "Declare convergence because delta_step_len of %s <= options.tol_delta_step of %s", 1.5, 2
        )
        mock_logger.log.assert_any_call(
            LogLevel.L3, "Declare convergence because abs(pred_func_change) of %s <= options.tol_delta_cost of %s", 2.5, 3
        )

    def test_no_logging(self, mock_logger: Mock) -> None:
        mock_logger.setLevel(logging.NOTSET)  # CRITICAL
        with capture_output("err") as ctx:
            convergence = estm.batch._check_for_convergence(self.opti_opts, 0.5, 1.5, 2.5)
        error = ctx.get_error()
        ctx.close()
        self.assertTrue(convergence)
        self.assertEqual(error, "")


# %% estimation.batch._double_dogleg
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_estimation_batch__double_dogleg(unittest.TestCase):
    r"""
    Tests the estimation.batch._double_dogleg function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.delta_param = np.array([1.0, 2.0])
        self.gradient = np.array([3.0, 4.0])
        self.grad_hessian_grad = 5.0
        self.x_bias = 0.1
        self.trust_radius = 2.0

    def test_large_trust_radius(self) -> None:
        # Newton step in trust radius
        self.trust_radius = 10000.0
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(
            self.delta_param, self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius
        )

    def test_small_bias(self) -> None:
        # Newton step outside trust_radius
        self.x_bias = 0.01
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(
            self.delta_param, self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius
        )

    def test_gradient_step(self) -> None:
        # Newton step outside trust_radius
        self.x_bias = 0.001
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(
            self.delta_param, self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius
        )

    def test_dogleg1(self) -> None:
        # Dogleg step 1
        self.x_bias = 0.001
        self.grad_hessian_grad = 75.0
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(
            self.delta_param, self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius
        )

    def test_dogleg2(self) -> None:
        # Dogleg step 2
        self.x_bias = 0.001
        self.grad_hessian_grad = 75.0
        self.delta_param = 0.001 * np.array([1, 2])
        (new_delta_param, step_len, step_scale, step_type) = estm.batch._double_dogleg(
            self.delta_param, self.gradient, self.grad_hessian_grad, self.x_bias, self.trust_radius
        )


# %% estimation.batch._dogleg_search
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_batch__dogleg_search(unittest.TestCase):
    r"""
    Tests the estimation.batch._dogleg_search function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        time       = np.arange(251.0)  # fmt: skip
        sim_params = SimParams(time, magnitude=3.5, frequency=12.0, phase=180.0)
        truth_time = np.arange(-10.0, 201.0)
        truth_data = 5 * np.sin(2 * np.pi * 10 * time / 1000 + 90 * np.pi / 180)

        # fmt: off
        self.opti_opts                = estm.OptiOpts()
        self.opti_opts.model_func     = sim_model
        self.opti_opts.model_args     = {"sim_params": sim_params}
        self.opti_opts.cost_func      = cost_wrapper
        self.opti_opts.cost_args      = {"results_time": time, "truth_time": truth_time, "truth_data": truth_data}
        self.opti_opts.get_param_func = get_parameter
        self.opti_opts.set_param_func = set_parameter
        self.opti_opts.output_folder  = None
        self.opti_opts.output_results = None
        self.opti_opts.params         = []
        # fmt: on

        # Parameters to estimate
        self.opti_opts.params.append(estm.OptiParam("magnitude", best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("frequency", best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("phase", best=180, min_=0, max_=360, typical=100, minstep=0.1))

        self.model_args = self.opti_opts.model_args

        self.bpe_results = estm.BpeResults()
        self.cur_results = estm.CurrentResults()

        # initialize current results
        self.cur_results.innovs = estm.batch._function_wrapper(
            model_func=self.opti_opts.model_func,
            cost_func=self.opti_opts.cost_func,
            cost_args=self.opti_opts.cost_args,
            model_args=self.model_args,
        )
        self.bpe_results.num_evals += 1
        self.cur_results.trust_rad = self.opti_opts.trust_radius
        self.cur_results.cost = 0.5 * estm.batch._calc_sum_squares(self.cur_results.innovs, ignore_nans=True)
        names = estm.OptiParam.get_names(self.opti_opts.params)
        self.cur_results.params = self.opti_opts.get_param_func(names=names, **self.model_args)

        # set relevant results variables
        # fmt: off
        self.bpe_results.param_names  = [name.encode("utf-8") for name in names]
        self.bpe_results.begin_params = self.cur_results.params.copy()
        self.bpe_results.begin_innovs = self.cur_results.innovs.copy()
        self.bpe_results.begin_cost   = self.cur_results.cost
        self.bpe_results.costs.append(self.cur_results.cost)

        self.delta_param = np.array([1.0, 2.0, 3.0])
        self.gradient    = np.array([4.0, 5.0, 6.0])
        self.hessian     = np.array([[5.0, 2.0, 1.0], [1.0, 2.0, 5.0], [3.0, 3.0, 3.0]])
        self.jacobian    = np.random.rand(201, 3)
        self.normalized  = False
        # fmt: on

    def test_nominal(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        estm.batch._dogleg_search(
            self.opti_opts,
            self.opti_opts.model_args,
            self.bpe_results,
            self.cur_results,
            self.delta_param,
            self.jacobian,
            self.gradient,
            self.hessian,
            normalized=self.normalized,
        )

    def test_normalized(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        self.normalized = True
        estm.batch._dogleg_search(
            self.opti_opts,
            self.opti_opts.model_args,
            self.bpe_results,
            self.cur_results,
            self.delta_param,
            self.jacobian,
            self.gradient,
            self.hessian,
            normalized=self.normalized,
        )

    def test_levenberg_marquardt(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        self.opti_opts.search_method = "levenberg_marquardt"
        estm.batch._dogleg_search(
            self.opti_opts,
            self.opti_opts.model_args,
            self.bpe_results,
            self.cur_results,
            self.delta_param,
            self.jacobian,
            self.gradient,
            self.hessian,
            normalized=self.normalized,
        )

    def test_bad_method(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        self.opti_opts.search_method = "bad_method"
        with self.assertRaises(ValueError):
            estm.batch._dogleg_search(
                self.opti_opts,
                self.opti_opts.model_args,
                self.bpe_results,
                self.cur_results,
                self.delta_param,
                self.jacobian,
                self.gradient,
                self.hessian,
                normalized=self.normalized,
            )

    def test_minimums(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        self.opti_opts.params[0].min_ = 10  # type: ignore[index]
        estm.batch._dogleg_search(
            self.opti_opts,
            self.opti_opts.model_args,
            self.bpe_results,
            self.cur_results,
            self.delta_param,
            self.jacobian,
            self.gradient,
            self.hessian,
            normalized=self.normalized,
        )

    def test_huge_trust_radius(self, mock_logger: Mock) -> None:
        assert self.opti_opts.model_args is not None
        # TODO: figure out how to get this to shrink a Newton step.
        self.opti_opts.trust_radius = 1000000
        estm.batch._dogleg_search(
            self.opti_opts,
            self.opti_opts.model_args,
            self.bpe_results,
            self.cur_results,
            self.delta_param,
            self.jacobian,
            self.gradient,
            self.hessian,
            normalized=self.normalized,
        )


# %% estimation.batch._analyze_results
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_batch__analyze_results(unittest.TestCase):
    r"""
    Tests the estimation.batch._analyze_results function with the following cases:
        Nominal
        Normalized
    """

    def setUp(self) -> None:
        self.opti_opts = estm.OptiOpts()
        self.opti_opts.params = [estm.OptiParam("a"), estm.OptiParam("b")]
        self.bpe_results = estm.BpeResults()
        self.bpe_results.param_names = [x.encode("utf-8") for x in ["a", "b"]]
        self.jacobian = np.array([[1, 2], [3, 4], [5, 6]])
        self.normalized = False

    def test_nominal(self, mock_logger: Mock) -> None:
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)

    def test_normalized(self, mock_logger: Mock) -> None:
        self.normalized = True
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)

    def test_no_iters(self, mock_logger: Mock) -> None:
        self.opti_opts.max_iters = 0
        estm.batch._analyze_results(self.opti_opts, self.bpe_results, self.jacobian, self.normalized)


# %% estimation.validate_opti_opts
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_validate_opti_opts(unittest.TestCase):
    r"""
    Tests the estimation.validate_opti_opts function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        # fmt: off
        self.opti_opts                = estm.OptiOpts()
        self.opti_opts.model_func     = str
        self.opti_opts.model_args     = {"a": 1}
        self.opti_opts.cost_func      = str
        self.opti_opts.cost_args      = {"b": 2}
        self.opti_opts.get_param_func = str
        self.opti_opts.set_param_func = repr
        self.opti_opts.output_folder  = None
        self.opti_opts.output_results = None
        self.opti_opts.params         = [1, 2]  # type: ignore[list-item]
        # fmt: on

    def support(self) -> None:
        with self.assertRaises(AssertionError):
            estm.validate_opti_opts(self.opti_opts)

    def test_nominal(self, mock_logger: Mock) -> None:
        is_valid = estm.validate_opti_opts(self.opti_opts)
        self.assertTrue(is_valid)
        mock_logger.log.assert_any_call(LogLevel.L5, "******************************")
        mock_logger.log.assert_any_call(LogLevel.L5, "Validating optimization options.")

    def test_no_logging(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(LogLevel.L3)
        is_valid = estm.validate_opti_opts(self.opti_opts)
        self.assertTrue(is_valid)
        mock_logger.log.assert_any_call(LogLevel.L5, "******************************")
        mock_logger.log.assert_any_call(LogLevel.L5, "Validating optimization options.")

    def test_not_valid1(self, mock_logger: Mock) -> None:
        self.opti_opts.model_func = None
        self.support()

    def test_not_valid2(self, mock_logger: Mock) -> None:
        self.opti_opts.model_args = None
        self.support()

    def test_not_valid3(self, mock_logger: Mock) -> None:
        self.opti_opts.cost_func = None
        self.support()

    def test_not_valid4(self, mock_logger: Mock) -> None:
        self.opti_opts.cost_args = None
        self.support()

    def test_not_valid5(self, mock_logger: Mock) -> None:
        self.opti_opts.get_param_func = None
        self.support()

    def test_not_valid6(self, mock_logger: Mock) -> None:
        self.opti_opts.set_param_func = None
        self.support()

    def test_not_valid7(self, mock_logger: Mock) -> None:
        self.opti_opts.params = []
        self.support()

    def test_not_valid8(self, mock_logger: Mock) -> None:
        self.opti_opts.params = None
        self.support()

    def test_not_valid9(self, mock_logger: Mock) -> None:
        self.opti_opts.slope_method = "bad_sided"
        self.support()

    def test_not_valid10(self, mock_logger: Mock) -> None:
        self.opti_opts.search_method = "wild_ass_guess"
        self.support()


# %% estimation.run_bpe
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.estimation.batch.logger")
class Test_estimation_run_bpe(unittest.TestCase):
    r"""
    Tests the estimation.run_bpe function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        estm.batch.logger.setLevel(LogLevel.L5)
        time       = np.arange(251.0)  # fmt: skip
        sim_params = SimParams(time, magnitude=3.5, frequency=12.0, phase=180.0)
        truth_time = np.arange(-10.0, 201.0)
        truth_data = 5 * np.sin(2 * np.pi * 10 * time / 1000 + 90 * np.pi / 180)

        # fmt: off
        self.opti_opts                = estm.OptiOpts()
        self.opti_opts.model_func     = sim_model
        self.opti_opts.model_args     = {"sim_params": sim_params}
        self.opti_opts.cost_func      = cost_wrapper
        self.opti_opts.cost_args      = {"results_time": time, "truth_time": truth_time, "truth_data": truth_data}
        self.opti_opts.get_param_func = get_parameter
        self.opti_opts.set_param_func = set_parameter
        self.opti_opts.output_folder  = None
        self.opti_opts.output_results = None
        self.opti_opts.params         = []
        # fmt: on

        # Parameters to estimate
        self.opti_opts.params.append(estm.OptiParam("magnitude", best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("frequency", best=20, min_=1, max_=1000, typical=60, minstep=0.01))
        self.opti_opts.params.append(estm.OptiParam("phase", best=180, min_=0, max_=360, typical=100, minstep=0.1))

    def test_nominal(self, mock_logger: Mock) -> None:
        mock_logger.level = LogLevel.L5
        (bpe_results, results) = estm.run_bpe(self.opti_opts)
        # TODO: check logging results?
        self.assertTrue(isinstance(bpe_results, estm.BpeResults))
        self.assertTrue(isinstance(results, np.ndarray))

    def test_no_logging(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)
        # TODO: check logging results

    def test_max_likelihood(self, mock_logger: Mock) -> None:
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        self.opti_opts.is_max_like = True
        estm.run_bpe(self.opti_opts)

    def test_normalized(self, mock_logger: Mock) -> None:
        pass  # TODO: method not yet coded all the way

    def test_two_sided(self, mock_logger: Mock) -> None:
        mock_logger.level = LogLevel.L5
        self.opti_opts.slope_method = "two_sided"
        estm.run_bpe(self.opti_opts)
        # for (ix, line) in enumerate(lines):
        #     if line == 'Running iteration 1.':
        #         self.assertTrue(lines[ix+1].startswith('  Running model with magnitude'))
        #         self.assertTrue(lines[ix+2].startswith('  Running model with magnitude'))
        #         break
        # else:
        #     self.assertTrue(False, 'two sided had issues')
        # rerun with no logging
        estm.batch.logger.setLevel(logging.CRITICAL)
        estm.run_bpe(self.opti_opts)

    def test_to_convergence(self, mock_logger: Mock) -> None:
        self.opti_opts.max_iters = 100
        mock_logger.level = LogLevel.L5
        estm.run_bpe(self.opti_opts)
        # for line in lines:
        #     if line.startswith('Declare convergence'):
        #         break
        # else:
        #     self.assertTrue(False, "Didn't converge")

    @unittest.skipIf(not HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_saving(self, mock_logger: Mock) -> None:
        mock_logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        self.opti_opts.max_iters = 1
        self.opti_opts.output_folder = get_tests_dir()
        self.opti_opts.output_results = Path("temp_results.hdf5")
        estm.run_bpe(self.opti_opts)
        # TODO: test with more iterations and files?

    def test_startup_finish_funcs(self, mock_logger: Mock) -> None:
        self.opti_opts.start_func = lambda sim_params: {"settings": None, "additional": None}
        self.opti_opts.final_func = lambda sim_params, settings, additional: None
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)

    def test_failed(self, mock_logger: Mock) -> None:
        # TODO: this case doesn't fail yet.  Make it do so.
        self.opti_opts.max_iters = 1
        self.opti_opts.tol_delta_step = 100
        self.opti_opts.step_limit = 1
        estm.batch.logger.setLevel(logging.CRITICAL)
        mock_logger.level = logging.CRITICAL
        estm.run_bpe(self.opti_opts)

    def tearDown(self) -> None:
        if self.opti_opts.output_results is not None and self.opti_opts.output_folder is not None:
            files = (self.opti_opts.output_results, Path("bpe_results_iter_1.hdf5"), Path("cur_results_iter_1.hdf5"))
            for this_file in files:
                self.opti_opts.output_folder.joinpath(this_file).unlink(missing_ok=True)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
