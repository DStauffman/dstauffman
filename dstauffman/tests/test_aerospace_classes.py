r"""
Test file for the `classes` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import copy
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
        self.assertTrue(isinstance(innov, space.KfInnov))
        self.assertEqual(innov.name, '')
        self.assertIsNone(innov.chan)
        self.assertEqual(innov.units, '')
        self.assertIsNone(innov.time)
        self.assertIsNone(innov.innov)
        self.assertIsNone(innov.norm)
        self.assertIsNone(innov.status)
        self.assertIsNone(innov.fploc)
        self.assertIsNone(innov.snr)

    def test_copy(self) -> None:
        innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        innov.chan = ['X', 'Y']
        innov2 = copy.copy(innov)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        self.assertEqual(innov.name, 'Gnd')
        self.assertEqual(innov.units, 'm')
        self.assertEqual(innov.time.shape, (60, ))
        self.assertTrue(np.issubdtype(innov.time.dtype, np.floating))
        self.assertEqual(innov.innov.shape, (2, 60))
        self.assertEqual(innov.norm.shape, (2, 60))
        self.assertEqual(innov.status.shape, (60, ))
        self.assertIsNone(innov.fploc)
        self.assertIsNone(innov.snr)
        # copy artifacts
        self.assertEqual(innov.name, innov2.name)
        self.assertIs(innov.chan, innov2.chan)
        self.assertEqual(innov.units, innov2.units)
        self.assertIs(innov.time, innov2.time)
        self.assertIs(innov.innov, innov2.innov)
        self.assertIs(innov.norm, innov2.norm)
        self.assertIs(innov.status, innov2.status)

    def test_deepcopy(self) -> None:
        innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=NP_DATETIME_FORM)
        innov.chan = ['X', 'Y']
        innov2 = copy.deepcopy(innov)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        self.assertEqual(innov.name, 'Gnd')
        self.assertEqual(innov.units, 'm')
        self.assertEqual(innov.time.shape, (60, ))
        self.assertTrue(np.issubdtype(innov.time.dtype, np.datetime64))
        self.assertEqual(innov.innov.shape, (2, 60))
        self.assertEqual(innov.norm.shape, (2, 60))
        self.assertEqual(innov.status.shape, (60, ))
        self.assertIsNone(innov.fploc)
        self.assertIsNone(innov.snr)
        # copy artifacts
        self.assertEqual(innov.name, innov2.name)
        self.assertIsNot(innov.chan, innov2.chan)
        self.assertEqual(innov.units, innov2.units)
        self.assertIsNot(innov.time, innov2.time)
        self.assertIsNot(innov.innov, innov2.innov)
        self.assertIsNot(innov.norm, innov2.norm)
        self.assertIsNot(innov.status, innov2.status)
        np.testing.assert_array_equal(innov.time, innov2.time)
        np.testing.assert_array_equal(innov.innov, innov2.innov)
        np.testing.assert_array_equal(innov.norm, innov2.norm)
        np.testing.assert_array_equal(innov.status, innov2.status)

    def test_combine_nominal(self) -> None:
        innov1 = space.KfInnov(name='Name 1', units='m', num_innovs=30, num_axes=2, time_dtype=float)
        innov1.chan = ['X', 'Y']
        innov2 = space.KfInnov(name='Name 2', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        innov2.chan = ['X', 'Y']
        innov3 = innov1.combine(innov2)
        assert innov3.time is not None
        assert innov3.innov is not None
        assert innov3.norm is not None
        assert innov3.status is not None
        self.assertEqual(innov3.name, 'Name 1')
        self.assertEqual(innov3.units, 'm')
        self.assertEqual(innov3.chan, ['X', 'Y'])
        self.assertEqual(innov3.time.shape, (90, ))
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.floating))
        self.assertEqual(innov3.innov.shape, (2, 90))
        self.assertEqual(innov3.norm.shape, (2, 90))
        self.assertEqual(innov3.status.shape, (90, ))
        self.assertIsNone(innov3.fploc)
        self.assertIsNone(innov3.snr)

    def test_combine_to_empty(self) -> None:
        innov1 = space.KfInnov()
        innov1.name = 'Name 1'
        innov1.units = 'm'
        innov1.chan = ['X', 'Y']
        innov2 = space.KfInnov(name='Name 2', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        innov2.chan = ['X', 'Y']
        innov3 = innov1.combine(innov2)
        self.assertEqual(innov3.name, 'Name 2')
        self.assertEqual(innov3.units, 'm')
        self.assertEqual(innov3.time.shape, (60, ))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.floating))  # type: ignore[union-attr]
        self.assertEqual(innov3.innov.shape, (2, 60))  # type: ignore[union-attr]
        self.assertEqual(innov3.norm.shape, (2, 60))  # type: ignore[union-attr]
        self.assertEqual(innov3.status.shape, (60, ))  # type: ignore[union-attr]
        self.assertIsNone(innov3.fploc)
        self.assertIsNone(innov3.snr)

    def test_combine_inplace(self) -> None:
        innov1 = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=3, time_dtype=NP_DATETIME_FORM)
        innov1.chan = ['X', 'Y', 'Z']
        innov1.fploc = np.random.rand(2, 60)
        innov1.snr = np.random.rand(60)
        innov2 = space.KfInnov(name='Gnd', units='m', num_innovs=50, num_axes=3, time_dtype=NP_DATETIME_FORM)
        innov2.chan = ['X', 'Y', 'Z']
        innov2.fploc = np.random.rand(2, 50)
        innov2.snr = np.random.rand(50)
        innov3 = innov1.combine(innov2, inplace=True)
        self.assertEqual(innov3.name, 'Gnd')
        self.assertEqual(innov3.units, 'm')
        self.assertEqual(innov3.time.shape, (110, ))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.datetime64))  # type: ignore[union-attr]
        self.assertEqual(innov3.innov.shape, (3, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.norm.shape, (3, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.status.shape, (110, ))  # type: ignore[union-attr]
        self.assertEqual(innov3.fploc.shape, (2, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.snr.shape, (110, ))  # type: ignore[union-attr]

    def test_chop(self) -> None:
        innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.chan = ['X', 'Y']
        innov.time[:] = np.arange(60.)
        innov.innov[:] = np.random.rand(2, 60)
        innov.norm[:] = 10 * innov.innov
        innov.status[:] = np.ones(60, dtype=int)
        innov2 = innov.chop(ti=10, tf=20, include_last=False)
        assert innov2.time is not None
        assert innov2.innov is not None
        assert innov2.norm is not None
        assert innov2.status is not None
        innov3 = innov.chop(ti=20, tf=40, include_last=True)
        assert innov3.time is not None
        assert innov3.innov is not None
        assert innov3.norm is not None
        assert innov3.status is not None
        self.assertIsNot(innov, innov2)
        self.assertEqual(innov.name, 'Gnd')
        self.assertEqual(innov.units, 'm')
        self.assertEqual(innov2.name, 'Gnd')
        self.assertEqual(innov2.units, 'm')
        self.assertEqual(innov2.chan, ['X', 'Y'])
        self.assertEqual(innov3.name, 'Gnd')
        self.assertEqual(innov3.units, 'm')
        self.assertEqual(innov3.chan, ['X', 'Y'])
        np.testing.assert_array_equal(innov.time, np.arange(60.))
        np.testing.assert_array_equal(innov2.time, np.arange(10., 20.))
        np.testing.assert_array_equal(innov3.time, np.arange(20., 41.))
        self.assertEqual(innov2.innov.shape, (2, 10))
        self.assertEqual(innov2.norm.shape, (2, 10))
        self.assertEqual(innov2.status.shape, (10, ))
        self.assertEqual(innov3.innov.shape, (2, 21))
        self.assertEqual(innov3.norm.shape, (2, 21))
        self.assertEqual(innov3.status.shape, (21, ))

    def test_chop_inplace(self) -> None:
        innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.chan = ['X', 'Y']
        innov.time[:] = np.arange(60.)
        innov.innov[:] = np.random.rand(2, 60)
        innov.norm[:] = 10 * innov.innov
        innov.status[:] = np.ones(60, dtype=int)
        innov2 = innov.chop(ti=10, tf=20, inplace=True)
        assert innov2.time is not None
        assert innov2.innov is not None
        assert innov2.norm is not None
        assert innov2.status is not None
        self.assertIs(innov, innov2)
        self.assertEqual(innov.name, 'Gnd')
        self.assertEqual(innov.units, 'm')
        self.assertEqual(innov.chan, ['X', 'Y'])
        np.testing.assert_array_equal(innov2.time, np.arange(10., 21.))
        self.assertEqual(innov2.innov.shape, (2, 11))
        self.assertEqual(innov2.norm.shape, (2, 11))
        self.assertEqual(innov2.status.shape, (11, ))

    def test_chop_return_ends(self) -> None:
        innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.chan = ['X', 'Y']
        innov.time[:] = np.arange(60.)
        innov.innov[:] = np.random.rand(2, 60)
        innov.norm[:] = 10 * innov.innov
        innov.status[:] = np.ones(60, dtype=int)
        (innov3, innov2, innov4) = innov.chop(ti=10, tf=20, return_ends=True)
        assert innov2.time is not None
        assert innov2.innov is not None
        assert innov2.norm is not None
        assert innov2.status is not None
        assert innov3.time is not None
        assert innov3.innov is not None
        assert innov3.norm is not None
        assert innov3.status is not None
        assert innov4.time is not None
        assert innov4.innov is not None
        assert innov4.norm is not None
        assert innov4.status is not None
        self.assertIsNot(innov, innov2)
        for this_innov in [innov, innov2, innov3, innov4]:
            self.assertEqual(this_innov.name, 'Gnd')
            self.assertEqual(this_innov.units, 'm')
            self.assertEqual(this_innov.chan, ['X', 'Y'])
        np.testing.assert_array_equal(innov2.time, np.arange(10., 21.))
        self.assertEqual(innov2.innov.shape, (2, 11))
        self.assertEqual(innov2.norm.shape, (2, 11))
        self.assertEqual(innov2.status.shape, (11, ))
        np.testing.assert_array_equal(innov3.time, np.arange(0., 10.))
        self.assertEqual(innov3.innov.shape, (2, 10))
        self.assertEqual(innov3.norm.shape, (2, 10))
        self.assertEqual(innov3.status.shape, (10, ))
        np.testing.assert_array_equal(innov4.time, np.arange(21., 60.))
        self.assertEqual(innov4.innov.shape, (2, 39))
        self.assertEqual(innov4.norm.shape, (2, 39))
        self.assertEqual(innov4.status.shape, (39, ))

#%% aerospace.Kf
class Test_aerospace_Kf(unittest.TestCase):
    r"""
    Tests the aerospace.Kf class with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.filename = get_tests_dir() / 'test_kf.hdf5'

    def test_nominal(self) -> None:
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf))
        self.assertEqual(kf.name, '')
        self.assertIsNone(kf.chan)
        self.assertIsNone(kf.time)
        self.assertIsNone(kf.att)
        self.assertIsNone(kf.pos)
        self.assertIsNone(kf.vel)
        self.assertIsNone(kf.active)
        self.assertIsNone(kf.state)
        self.assertIsNone(kf.istate)
        self.assertIsNone(kf.covar)
        self.assertTrue(isinstance(kf.innov, space.KfInnov))

    @unittest.skipIf(not HAVE_H5PY or not HAVE_NUMPY, 'Skipping due to missing h5py/numpy dependency.')
    def test_save_and_load(self) -> None:
        kf = space.Kf(num_points=2, num_states=6, num_axes=3, num_innovs=4)
        kf.chan = ['a', 'b', 'c']
        kf.save(self.filename)
        kf2 = space.Kf.load(self.filename)
        self.assertEqual(kf.chan, kf2.chan)
        np.testing.assert_array_equal(kf.time, kf2.time)

    def test_combine_nominal(self) -> None:
        kf1 = space.Kf(name='Name 1', num_points=30, num_states=6, time_dtype=float)
        kf1.chan = ['X', 'Y']
        kf2 = space.Kf(name='Name 2', num_points=60, num_states=6, time_dtype=float)
        kf2.chan = ['X', 'Y']
        kf3 = kf1.combine(kf2)
        assert kf3.time is not None
        assert kf3.att is not None
        assert kf3.state is not None
        assert kf3.istate is not None
        self.assertEqual(kf3.name, 'Name 1')
        self.assertEqual(kf3.chan, ['X', 'Y'])
        self.assertEqual(kf3.time.shape, (90, ))
        self.assertTrue(np.issubdtype(kf3.time.dtype, np.floating))
        self.assertEqual(kf3.att.shape, (4, 90))
        self.assertEqual(kf3.pos.shape, (3, 90))
        self.assertEqual(kf3.vel.shape, (3, 90))
        self.assertEqual(kf3.state.shape, (6, 90))
        self.assertEqual(kf3.istate.shape, (6, ))

    # def test_combine_to_empty(self) -> None:
    #     innov1 = space.KfInnov()
    #     innov1.name = 'Name 1'
    #     innov1.units = 'm'
    #     innov1.chan = ['X', 'Y']
    #     innov2 = space.KfInnov(name='Name 2', units='m', num_innovs=60, num_axes=2, time_dtype=float)
    #     innov2.chan = ['X', 'Y']
    #     innov3 = innov1.combine(innov2)
    #     self.assertEqual(innov3.name, 'Name 2')
    #     self.assertEqual(innov3.units, 'm')
    #     self.assertEqual(innov3.time.shape, (60, ))  # type: ignore[union-attr]
    #     self.assertTrue(np.issubdtype(innov3.time.dtype, np.floating))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.innov.shape, (2, 60))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.norm.shape, (2, 60))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.status.shape, (60, ))  # type: ignore[union-attr]
    #     self.assertIsNone(innov3.fploc)
    #     self.assertIsNone(innov3.snr)

    # def test_combine_inplace(self) -> None:
    #     innov1 = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=3, time_dtype=NP_DATETIME_FORM)
    #     innov1.chan = ['X', 'Y', 'Z']
    #     innov1.fploc = np.random.rand(2, 60)
    #     innov1.snr = np.random.rand(60)
    #     innov2 = space.KfInnov(name='Gnd', units='m', num_innovs=50, num_axes=3, time_dtype=NP_DATETIME_FORM)
    #     innov2.chan = ['X', 'Y', 'Z']
    #     innov2.fploc = np.random.rand(2, 50)
    #     innov2.snr = np.random.rand(50)
    #     innov3 = innov1.combine(innov2, inplace=True)
    #     self.assertEqual(innov3.name, 'Gnd')
    #     self.assertEqual(innov3.units, 'm')
    #     self.assertEqual(innov3.time.shape, (110, ))  # type: ignore[union-attr]
    #     self.assertTrue(np.issubdtype(innov3.time.dtype, np.datetime64))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.innov.shape, (3, 110))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.norm.shape, (3, 110))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.status.shape, (110, ))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.fploc.shape, (2, 110))  # type: ignore[union-attr]
    #     self.assertEqual(innov3.snr.shape, (110, ))  # type: ignore[union-attr]

    def tearDown(self) -> None:
        self.filename.unlink(missing_ok=True)

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

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
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

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
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

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_pprint1(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as out:
            kf_record.pprint()
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'KfRecord')
        self.assertEqual(lines[1], ' time = [0. 1. 2. 3. 4.]')

    @unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_pprint2(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as out:
            kf_record.pprint(max_elements=0)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'KfRecord')
        self.assertEqual(lines[1], ' time = <ndarray float64 (5,)>')

    def test_chop(self) -> None:
        kf_record = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=float)
        assert kf_record.P.shape is not None
        self.assertEqual(kf_record.stm.shape, (6, 6, 60), 'stm shape mismatch.')
        self.assertEqual(kf_record.H.shape, (3, 9, 60), 'H shape mismatch.')
        self.assertEqual(kf_record.Pz.shape, (3, 3, 60), 'Pz shape mismatch.')
        self.assertEqual(kf_record.K.shape, (6, 3, 60), 'K shape mismatch.')
        self.assertEqual(kf_record.z.shape, (3, 60), 'z shape mismatch.')
        kf_record.time[:] = np.arange(60.)
        kf_record2 = kf_record.chop(ti=10, tf=20, include_last=False)
        assert kf_record2.time is not None
        assert kf_record2.P is not None
        assert kf_record2.stm is not None
        assert kf_record2.H is not None
        assert kf_record2.Pz is not None
        assert kf_record2.K is not None
        assert kf_record2.z is not None
        kf_record3 = kf_record.chop(ti=20, tf=40, include_last=True)
        assert kf_record3.time is not None
        assert kf_record3.P is not None
        assert kf_record3.stm is not None
        assert kf_record3.H is not None
        assert kf_record3.Pz is not None
        assert kf_record3.K is not None
        assert kf_record3.z is not None
        self.assertIsNot(kf_record, kf_record2)
        np.testing.assert_array_equal(kf_record.time, np.arange(60.))
        np.testing.assert_array_equal(kf_record2.time, np.arange(10., 20.))
        np.testing.assert_array_equal(kf_record3.time, np.arange(20., 41.))
        self.assertEqual(kf_record2.stm.shape, (6, 6, 10))
        self.assertEqual(kf_record2.H.shape, (3, 9, 10))
        self.assertEqual(kf_record2.Pz.shape, (3, 3, 10))
        self.assertEqual(kf_record2.K.shape, (6, 3, 10))
        self.assertEqual(kf_record2.z.shape, (3, 10))
        self.assertEqual(kf_record3.stm.shape, (6, 6, 21))
        self.assertEqual(kf_record3.H.shape, (3, 9, 21))
        self.assertEqual(kf_record3.Pz.shape, (3,3, 21))
        self.assertEqual(kf_record3.K.shape, (6, 3, 21))
        self.assertEqual(kf_record3.z.shape, (3, 21))

    # def test_chop_inplace(self) -> None:
    #     kf_record = space.KfRecord(num_points=30, num_states=6, num_active=3, num_axes=2, time_dtype=float)
    #     assert kf_record.time is not None
    #     assert kf_record.kf_record is not None
    #     assert kf_record.norm is not None
    #     assert kf_record.status is not None
    #     kf_record.chan = ['X', 'Y']
    #     kf_record.time[:] = np.arange(60.)
    #     kf_record.kf_record[:] = np.random.rand(2, 60)
    #     kf_record.norm[:] = 10 * kf_record.kf_record
    #     kf_record.status[:] = np.ones(60, dtype=int)
    #     kf_record2 = kf_record.chop(ti=10, tf=20, inplace=True)
    #     assert kf_record2.time is not None
    #     assert kf_record2.kf_record is not None
    #     assert kf_record2.norm is not None
    #     assert kf_record2.status is not None
    #     self.assertIs(kf_record, kf_record2)
    #     self.assertEqual(kf_record.name, 'Gnd')
    #     self.assertEqual(kf_record.units, 'm')
    #     self.assertEqual(kf_record.chan, ['X', 'Y'])
    #     np.testing.assert_array_equal(kf_record2.time, np.arange(10., 21.))
    #     self.assertEqual(kf_record2.kf_record.shape, (2, 11))
    #     self.assertEqual(kf_record2.norm.shape, (2, 11))
    #     self.assertEqual(kf_record2.status.shape, (11, ))

    # def test_chop_return_ends(self) -> None:
    #     innov = space.KfInnov(name='Gnd', units='m', num_innovs=60, num_axes=2, time_dtype=float)
    #     assert kf_record.time is not None
    #     assert kf_record.kf_record is not None
    #     assert kf_record.norm is not None
    #     assert kf_record.status is not None
    #     kf_record.chan = ['X', 'Y']
    #     kf_record.time[:] = np.arange(60.)
    #     kf_record.kf_record[:] = np.random.rand(2, 60)
    #     kf_record.norm[:] = 10 * kf_record.kf_record
    #     kf_record.status[:] = np.ones(60, dtype=int)
    #     (kf_record3, kf_record2, kf_record4) = kf_record.chop(ti=10, tf=20, return_ends=True)
    #     assert kf_record2.time is not None
    #     assert kf_record2.kf_record is not None
    #     assert kf_record2.norm is not None
    #     assert kf_record2.status is not None
    #     assert kf_record3.time is not None
    #     assert kf_record3.kf_record is not None
    #     assert kf_record3.norm is not None
    #     assert kf_record3.status is not None
    #     assert kf_record4.time is not None
    #     assert kf_record4.kf_record is not None
    #     assert kf_record4.norm is not None
    #     assert kf_record4.status is not None
    #     self.assertIsNot(kf_record, kf_record2)
    #     for this_kf_record in [kf_record, kf_record2, kf_record3, kf_record4]:
    #         self.assertEqual(this_kf_record.name, 'Gnd')
    #         self.assertEqual(this_kf_record.units, 'm')
    #         self.assertEqual(this_kf_record.chan, ['X', 'Y'])
    #     np.testing.assert_array_equal(kf_record2.time, np.arange(10., 21.))
    #     self.assertEqual(kf_record2.kf_record.shape, (2, 11))
    #     self.assertEqual(kf_record2.norm.shape, (2, 11))
    #     self.assertEqual(kf_record2.status.shape, (11, ))
    #     np.testing.assert_array_equal(kf_record3.time, np.arange(0., 10.))
    #     self.assertEqual(kf_record3.kf_record.shape, (2, 10))
    #     self.assertEqual(kf_record3.norm.shape, (2, 10))
    #     self.assertEqual(kf_record3.status.shape, (10, ))
    #     np.testing.assert_array_equal(kf_record4.time, np.arange(21., 60.))
    #     self.assertEqual(kf_record4.kf_record.shape, (2, 39))
    #     self.assertEqual(kf_record4.norm.shape, (2, 39))
    #     self.assertEqual(kf_record4.status.shape, (39, ))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
