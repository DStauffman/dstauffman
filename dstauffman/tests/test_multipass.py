r"""
Test file for the `matlab` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

#%% Supporting functions
def _model_wrapper(x, y):
    if x is None:
        raise ValueError('Bad value for x')
    if y is None:
        raise RuntimeError('Bad value for y')
    return x + np.sin(x) + np.cos(y*2)

#%% parfor_wrapper
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_load_matlab(unittest.TestCase):
    r"""
    Tests the load_matlab function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.temp = np.linspace(0, 2*np.pi)
        self.args = ((self.temp.copy(), self.temp.copy()), (self.temp + 2*np.pi, self.temp.copy()), \
            (self.temp + 4*np.pi, self.temp.copy()))
        self.max_cores = 6

    def test_not_parallel(self):
        results = dcs.parfor_wrapper(_model_wrapper, self.args, use_parfor=False, max_cores=self.max_cores)
        np.testing.assert_array_equal(results[0], _model_wrapper(self.temp, self.temp))

    def test_parallel(self):
        # TODO: check if this works with numpy, but without tblib installed
        results = dcs.parfor_wrapper(_model_wrapper, self.args, use_parfor=True, max_cores=self.max_cores)
        np.testing.assert_array_equal(results[0], _model_wrapper(self.temp, self.temp))

    def test_not_parallel_error(self):
        with self.assertRaises(ValueError) as context:
            dcs.parfor_wrapper(_model_wrapper, ((self.temp, self.temp), (None, None)), use_parfor=False)
        self.assertEqual(str(context.exception), 'Bad value for x')

    def test_parallel_error(self):
        with self.assertRaises(RuntimeError) as context:
            dcs.parfor_wrapper(_model_wrapper, ((self.temp, self.temp), (self.temp, None)), use_parfor=True)
        self.assertEqual(str(context.exception), 'Bad value for y')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
