r"""
Test file for the `classes` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import contextlib
import os
import unittest

from dstauffman import get_tests_dir, HAVE_H5PY, HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.KfInnov
class Test_aerospace_KfInnov(unittest.TestCase):
    r"""
    Tests the aerospace.KfInnov class with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        innov = space.KfInnov()
        self.assertTrue(isinstance(innov, space.KfInnov)) # TODO: test better

#%% aerospace.Kf
class Test_aerospace_Kf(unittest.TestCase):
    r"""
    Tests the aerospace.Kf class with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.filename = os.path.join(get_tests_dir(), 'test_kf.hdf5')

    def test_nominal(self) -> None:
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf)) # TODO: test better

    @unittest.skipIf(not HAVE_H5PY or not HAVE_NUMPY, 'Skipping due to missing h5py/numpy dependency.')
    def test_save_and_load(self) -> None:
        kf = space.Kf(num_points=2, num_states=6, num_axes=3, num_innovs=4)
        kf.chan = ['a', 'b', 'c']
        kf.save(self.filename)
        kf2 = space.Kf.load(self.filename)
        self.assertEqual(kf.chan, kf2.chan)
        np.testing.assert_array_equal(kf.time, kf2.time)

    def tearDown(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.filename)

#%% aerospace.KfRecord
class Test_aerospace_KfRecord(unittest.TestCase):
    r"""
    Tests the aerospace.KfRecord class with the following cases:
        TBD
    """
    pass # TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
