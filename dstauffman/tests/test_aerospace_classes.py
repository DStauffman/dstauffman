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

import numpy as np

from dstauffman import get_tests_dir
import dstauffman.aerospace as space

#%% aerospace.KfInnov
class Test_aerospace_KfInnov(unittest.TestCase):
    r"""
    Tests the aerospace.KfInnov class with the following cases:
        Nominal
    """
    def test_nominal(self):
        innov = space.KfInnov()
        self.assertTrue(isinstance(innov, space.KfInnov)) # TODO: test better

#%% aerospace.Kf
class Test_aerospace_Kf(unittest.TestCase):
    r"""
    Tests the aerospace.Kf class with the following cases:
        TBD
    """
    def setUp(self):
        self.filename = os.path.join(get_tests_dir(), 'test_kf.hdf5')

    def test_nominal(self):
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf)) # TODO: test better

    def test_save_and_load(self):
        kf = space.Kf(num_points=2, num_states=6, num_axes=3, num_innovs=4)
        kf.chan = ['a', 'b', 'c']
        kf.save(self.filename)
        kf2 = space.Kf.load(self.filename)
        self.assertEqual(kf.chan, kf2.chan)
        np.testing.assert_array_equal(kf.time, kf2.time)

    def tearDown(self):
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
