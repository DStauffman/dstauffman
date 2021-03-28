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

from dstauffman import capture_output, get_tests_dir, HAVE_H5PY, HAVE_NUMPY, NP_DATETIME_FORM
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
        Nominal
        With sizes
        Different time type
    """
    def setUp(self) -> None:
        self.fields = ('time', 'P', 'stm', 'H', 'Pz', 'K', 'z')

    def test_nominal(self) -> None:
        kf_record = space.KfRecord()
        self.assertIsInstance(kf_record, space.KfRecord)
        for key in self.fields:
            self.assertIsNone(getattr(kf_record, key), f'Expected None for field {key}')

    def test_arguments(self) -> None:
        kf_record = space.KfRecord(num_points=30, num_states=6, num_active=3, num_axes=2)
        assert kf_record.time is not None
        assert kf_record.P is not None
        assert kf_record.stm is not None
        assert kf_record.H is not None
        assert kf_record.Pz is not None
        assert kf_record.K is not None
        assert kf_record.z is not None
        self.assertEqual(kf_record.time.shape, (30, ), 'Time shape mismatch.')
        self.assertEqual(kf_record.P.shape, (3, 3, 30), 'P shape mismatch.')
        self.assertEqual(kf_record.stm.shape, (3, 3, 30), 'stm shape mismatch.')
        self.assertEqual(kf_record.H.shape, (2, 6, 30), 'H shape mismatch.')
        self.assertEqual(kf_record.Pz.shape, (2, 2, 30), 'Pz shape mismatch.')
        self.assertEqual(kf_record.K.shape, (3, 2, 30), 'K shape mismatch.')
        self.assertEqual(kf_record.z.shape, (2, 30), 'z shape mismatch.')

    def test_alternative_time(self) -> None:
        kf_record = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=NP_DATETIME_FORM)
        assert kf_record.time is not None
        assert kf_record.P is not None
        assert kf_record.stm is not None
        assert kf_record.H is not None
        assert kf_record.Pz is not None
        assert kf_record.K is not None
        assert kf_record.z is not None
        self.assertEqual(kf_record.time.dtype, '<M8[ns]')
        self.assertEqual(kf_record.time.shape, (60, ), 'Time shape mismatch.')
        self.assertEqual(kf_record.P.shape, (6, 6, 60), 'P shape mismatch.')
        self.assertEqual(kf_record.stm.shape, (6, 6, 60), 'stm shape mismatch.')
        self.assertEqual(kf_record.H.shape, (3, 9, 60), 'H shape mismatch.')
        self.assertEqual(kf_record.Pz.shape, (3, 3, 60), 'Pz shape mismatch.')
        self.assertEqual(kf_record.K.shape, (6, 3, 60), 'K shape mismatch.')
        self.assertEqual(kf_record.z.shape, (3, 60), 'z shape mismatch.')

    def test_pprint1(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as out:
            kf_record.pprint()
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'KfRecord')
        self.assertEqual(lines[1], ' time = [0. 1. 2. 3. 4.]')

    def test_pprint2(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as out:
            kf_record.pprint(max_elements=0)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'KfRecord')
        self.assertEqual(lines[1], ' time = <ndarray float64 (5,)>')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
