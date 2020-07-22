r"""
Test file for the `support` module of the dstauffman.estimation code.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
#.  Incorporated into dstauffman by David C. Stauffer in July 2020.
"""

#%% Imports
import logging
import unittest

import numpy as np

from dstauffman import Frozen, FixedDict
import dstauffman.estimation as estm

#%% Classes - _Config
class _Config(Frozen):
    r"""
    High level model configuration parameters.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Config
    >>> config = _Config()

    """
    def __init__(self):
        # log level for how much information to display while running
        self.log_level = logging.INFO

        # output folder/files
        self.output_folder  = ''
        self.output_results = 'results_model.hdf5'
        self.output_params  = 'results_param.pkl'

        # Whether to save the final state information to disk
        self.save_final = False

        # randomness
        self.repeatable_randomness = False
        self.repeatable_seed       = 1000
        self.repeatable_offset     = 0

        # parallelization
        self.use_parfor = False
        self.max_cores  = 4

        # default colormap
        self.colormap = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'

class _Model(Frozen):
    r"""
    Example model parameters.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Model
    >>> model = _Model()

    """
    def __init__(self):
        self.field1 = 1
        self.field2 = np.array([1, 2, 3])
        self.field3 = {'a': 5, 'b': np.array([1.5, 2.5, 10.])}
        self.field4 = FixedDict()
        self.field4['new'] = np.array([3, 4, 5])
        self.field4['old'] = '4 - 6'
        self.field4.freeze()

#%% Classes - _Parameters
class _Parameters(Frozen):
    r"""
    Example wrapper parameters master class.

    Examples
    --------
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()

    """
    def __init__(self):
        self.config = _Config()
        self.model = _Model()
        self.models = [_Model(), _Model()]
        self.models[0].field1 = 100
        self.models[1].field1 = 200
        self.models[1].field2[2] = 300
        self.models[1].field3['a'] = 500.
        self.models[1].field4['new'][1] = 444

#%% _check_valid_param_name
class Test__check_valid_param_name(unittest.TestCase):
    r"""
    Tests the _check_valid_param_name function with the following cases:
        Nominal
        Dictionary parameter
        Bad param name
        Bad attribute name
        Element of vector
    """
    @classmethod
    def setUpClass(cls):
        cls.param = _Parameters()

    def test_nominal(self):
        is_valid = estm.support._check_valid_param_name(self.param, 'param.config.log_level')
        self.assertTrue(is_valid)

    def test_dictionary(self):
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field3['a']")
        self.assertTrue(is_valid)

    def test_bad_name1(self):
        is_valid = estm.support._check_valid_param_name(self.param, 'not_param.config.log_level')
        self.assertFalse(is_valid)

    def test_bad_name2(self):
        is_valid = estm.support._check_valid_param_name(self.param, 'param.config.logless_level')
        self.assertFalse(is_valid)

    def test_vector_element(self):
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field2[1]")
        self.assertTrue(is_valid)

    def test_array_in_dict_name(self):
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field3['b'][2]")
        self.assertTrue(is_valid)
        is_valid = estm.support._check_valid_param_name(self.param, "param.model.field4['new'][0]")
        self.assertTrue(is_valid)

    def test_list_vector_element(self):
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[0].field2[1]")
        self.assertTrue(is_valid)

    def test_list_array_in_dict_name(self):
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[0].field3['b'][2]")
        self.assertTrue(is_valid)
        is_valid = estm.support._check_valid_param_name(self.param, "param.models[1].field4['new'][0]")
        self.assertTrue(is_valid)

#%% get_parameter
class Test_get_parameter(unittest.TestCase):
    r"""
    Tests the get_parameter function with the following cases:
        Nominal (covers four cases in one)
    """
    def setUp(self):
        self.values = [-100, -2, -3, 44.]
        self.names  = ['param.config.log_level', 'param.models[0].field1', "param.models[1].field2[2]", "param.models[1].field3['b'][1]"]
        self.param  = _Parameters()
        self.param.config.log_level = self.values[0]
        self.param.models[0].field1 = self.values[1]
        self.param.models[1].field2[2] = self.values[2]
        self.param.models[1].field3['b'][1] = self.values[3]

    def test_nominal(self):
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.values)

#%% set_parameter
class Test_set_parameters(unittest.TestCase):
    r"""
    Tests the set_parameter function with the following cases:
        Nominal (covers four cases in one)
    """
    def setUp(self):
        self.orig   = [20, 100, 300, 2.5]
        self.values = [-100, -2, -3, 44.]
        self.names  = ['param.config.log_level', 'param.models[0].field1', "param.models[1].field2[2]", "param.models[1].field3['b'][1]"]
        self.param  = _Parameters()
        self.param.config.log_level = self.orig[0]
        self.param.models[0].field1 = self.orig[1]
        self.param.models[1].field2[2] = self.orig[2]
        self.param.models[1].field3['b'][1] = self.orig[3]

    def test_nominal(self):
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.orig)
        estm.set_parameter(self.param, self.names, self.values)
        values = estm.get_parameter(self.param, self.names)
        np.testing.assert_array_almost_equal(values, self.values)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
