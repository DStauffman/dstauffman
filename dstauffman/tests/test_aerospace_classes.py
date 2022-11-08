r"""
Test file for the `classes` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import copy
import unittest

from slog import capture_output

from dstauffman import get_tests_dir, HAVE_H5PY, HAVE_NUMPY, NP_DATETIME_FORM, NP_DATETIME_UNITS, NP_TIMEDELTA_FORM
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.KfInnov
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_KfInnov(unittest.TestCase):
    r"""
    Tests the aerospace.KfInnov class with the following cases:
        Nominal
        Copy
        Deep copy
        Combine (x3 - nominal, empty, inplace)
        Chopping (x3 - nominal, inplace, return ends)
    """

    def test_nominal(self) -> None:
        innov = space.KfInnov()
        self.assertTrue(isinstance(innov, space.KfInnov))
        self.assertEqual(innov.name, "")
        self.assertIsNone(innov.chan)
        self.assertEqual(innov.units, "")
        self.assertIsNone(innov.time)
        self.assertIsNone(innov.innov)
        self.assertIsNone(innov.norm)
        self.assertIsNone(innov.status)
        self.assertIsNone(innov.fploc)
        self.assertIsNone(innov.snr)

    def test_copy(self) -> None:
        innov = space.KfInnov(name="Gnd", units="m", chan=("X", "Y"), num_innovs=60, num_axes=2, time_dtype=float)
        innov2 = copy.copy(innov)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        self.assertEqual(innov.name, "Gnd")
        self.assertEqual(innov.units, "m")
        self.assertEqual(innov.chan, ("X", "Y"))
        self.assertEqual(innov.time.shape, (60,))
        self.assertTrue(np.issubdtype(innov.time.dtype, np.floating))
        self.assertEqual(innov.innov.shape, (2, 60))
        self.assertEqual(innov.norm.shape, (2, 60))
        self.assertEqual(innov.status.shape, (60,))
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
        innov = space.KfInnov(name="Gnd", units="m", chan=["X", "Y"], num_innovs=60, num_axes=2, time_dtype=NP_DATETIME_FORM)
        innov2 = copy.deepcopy(innov)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        self.assertEqual(innov.name, "Gnd")
        self.assertEqual(innov.units, "m")
        self.assertEqual(innov.chan, ["X", "Y"])
        self.assertEqual(innov.time.shape, (60,))
        self.assertTrue(np.issubdtype(innov.time.dtype, np.datetime64))
        self.assertEqual(innov.innov.shape, (2, 60))
        self.assertEqual(innov.norm.shape, (2, 60))
        self.assertEqual(innov.status.shape, (60,))
        self.assertIsNone(innov.fploc)
        self.assertIsNone(innov.snr)
        # copy artifacts
        self.assertEqual(innov.name, innov2.name)
        self.assertIsNot(innov.chan, innov2.chan)
        self.assertEqual(innov.chan, innov2.chan)
        self.assertEqual(innov.units, innov2.units)
        self.assertIsNot(innov.time, innov2.time)
        self.assertIsNot(innov.innov, innov2.innov)
        self.assertIsNot(innov.norm, innov2.norm)
        self.assertIsNot(innov.status, innov2.status)
        np.testing.assert_array_equal(innov.time, innov2.time)  # type: ignore[arg-type]
        np.testing.assert_array_equal(innov.innov, innov2.innov)  # type: ignore[arg-type]
        np.testing.assert_array_equal(innov.norm, innov2.norm)  # type: ignore[arg-type]
        np.testing.assert_array_equal(innov.status, innov2.status)  # type: ignore[arg-type]

    def test_combine_nominal(self) -> None:
        innov1 = space.KfInnov(name="Name 1", units="m", chan=("X", "Y"), num_innovs=30, num_axes=2, time_dtype=float)
        innov2 = space.KfInnov(name="Name 2", units="m", chan=("X", "Y"), num_innovs=60, num_axes=2, time_dtype=float)
        innov3 = innov1.combine(innov2)
        assert innov3.time is not None
        assert innov3.innov is not None
        assert innov3.norm is not None
        assert innov3.status is not None
        self.assertEqual(innov3.name, "Name 1")
        self.assertEqual(innov3.units, "m")
        self.assertEqual(innov3.chan, ("X", "Y"))
        self.assertEqual(innov3.time.shape, (90,))
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.floating))
        self.assertEqual(innov3.innov.shape, (2, 90))
        self.assertEqual(innov3.norm.shape, (2, 90))
        self.assertEqual(innov3.status.shape, (90,))
        self.assertIsNone(innov3.fploc)
        self.assertIsNone(innov3.snr)

    def test_combine_to_empty(self) -> None:
        innov1 = space.KfInnov()
        innov1.name = "Name 1"
        innov1.units = "m"
        innov1.chan = ["X", "Y"]
        innov2 = space.KfInnov(name="Name 2", units="m", chan=["X", "Y"], num_innovs=60, num_axes=2, time_dtype=float)
        innov3 = innov1.combine(innov2)
        self.assertEqual(innov3.name, "Name 2")
        self.assertEqual(innov3.units, "m")
        self.assertEqual(innov3.chan, ["X", "Y"])
        self.assertEqual(innov3.time.shape, (60,))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.floating))  # type: ignore[union-attr]
        self.assertEqual(innov3.innov.shape, (2, 60))  # type: ignore[union-attr]
        self.assertEqual(innov3.norm.shape, (2, 60))  # type: ignore[union-attr]
        self.assertEqual(innov3.status.shape, (60,))  # type: ignore[union-attr]
        self.assertIsNone(innov3.fploc)
        self.assertIsNone(innov3.snr)

    def test_combine_inplace(self) -> None:
        innov1 = space.KfInnov(
            name="Gnd", units="m", chan=["X", "Y", "Z"], num_innovs=60, num_axes=3, time_dtype=NP_DATETIME_FORM
        )
        innov1.fploc = np.random.rand(2, 60)
        innov1.snr = np.random.rand(60)
        innov2 = space.KfInnov(
            name="Gnd", units="m", chan=["X", "Y", "Z"], num_innovs=50, num_axes=3, time_dtype=NP_DATETIME_FORM
        )
        innov2.fploc = np.random.rand(2, 50)
        innov2.snr = np.random.rand(50)
        innov3 = innov1.combine(innov2, inplace=True)
        self.assertIs(innov1, innov3)
        self.assertEqual(innov3.name, "Gnd")
        self.assertEqual(innov3.units, "m")
        self.assertEqual(innov3.chan, ["X", "Y", "Z"])
        self.assertEqual(innov3.time.shape, (110,))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(innov3.time.dtype, np.datetime64))  # type: ignore[union-attr]
        self.assertEqual(innov3.innov.shape, (3, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.norm.shape, (3, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.status.shape, (110,))  # type: ignore[union-attr]
        self.assertEqual(innov3.fploc.shape, (2, 110))  # type: ignore[union-attr]
        self.assertEqual(innov3.snr.shape, (110,))  # type: ignore[union-attr]

    def test_chop(self) -> None:
        innov = space.KfInnov(name="Gnd", units="m", chan=["X", "Y"], num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.time[:] = np.arange(60.0)
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
        self.assertEqual(innov.name, "Gnd")
        self.assertEqual(innov.units, "m")
        self.assertEqual(innov.chan, ["X", "Y"])
        self.assertEqual(innov2.name, "Gnd")
        self.assertEqual(innov2.units, "m")
        self.assertEqual(innov2.chan, ["X", "Y"])
        self.assertEqual(innov3.name, "Gnd")
        self.assertEqual(innov3.units, "m")
        self.assertEqual(innov3.chan, ["X", "Y"])
        np.testing.assert_array_equal(innov.time, np.arange(60.0))
        np.testing.assert_array_equal(innov2.time, np.arange(10.0, 20.0))
        np.testing.assert_array_equal(innov3.time, np.arange(20.0, 41.0))
        self.assertEqual(innov2.innov.shape, (2, 10))
        self.assertEqual(innov2.norm.shape, (2, 10))
        self.assertEqual(innov2.status.shape, (10,))
        self.assertEqual(innov3.innov.shape, (2, 21))
        self.assertEqual(innov3.norm.shape, (2, 21))
        self.assertEqual(innov3.status.shape, (21,))

    def test_chop_inplace(self) -> None:
        innov = space.KfInnov(name="Gnd", units="m", chan=["X", "Y"], num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.time[:] = np.arange(60.0)
        innov.innov[:] = np.random.rand(2, 60)
        innov.norm[:] = 10 * innov.innov
        innov.status[:] = np.ones(60, dtype=int)
        innov2 = innov.chop(ti=10, tf=20, inplace=True)
        assert innov2.time is not None
        assert innov2.innov is not None
        assert innov2.norm is not None
        assert innov2.status is not None
        self.assertIs(innov, innov2)
        self.assertEqual(innov.name, "Gnd")
        self.assertEqual(innov.units, "m")
        self.assertEqual(innov.chan, ["X", "Y"])
        np.testing.assert_array_equal(innov2.time, np.arange(10.0, 21.0))
        self.assertEqual(innov2.innov.shape, (2, 11))
        self.assertEqual(innov2.norm.shape, (2, 11))
        self.assertEqual(innov2.status.shape, (11,))

    def test_chop_return_ends(self) -> None:
        innov = space.KfInnov(name="Gnd", units="m", chan=["X", "Y"], num_innovs=60, num_axes=2, time_dtype=float)
        assert innov.time is not None
        assert innov.innov is not None
        assert innov.norm is not None
        assert innov.status is not None
        innov.time[:] = np.arange(60.0)
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
            self.assertEqual(this_innov.name, "Gnd")
            self.assertEqual(this_innov.units, "m")
            self.assertEqual(this_innov.chan, ["X", "Y"])
        np.testing.assert_array_equal(innov2.time, np.arange(10.0, 21.0))
        self.assertEqual(innov2.innov.shape, (2, 11))
        self.assertEqual(innov2.norm.shape, (2, 11))
        self.assertEqual(innov2.status.shape, (11,))
        np.testing.assert_array_equal(innov3.time, np.arange(0.0, 10.0))
        self.assertEqual(innov3.innov.shape, (2, 10))
        self.assertEqual(innov3.norm.shape, (2, 10))
        self.assertEqual(innov3.status.shape, (10,))
        np.testing.assert_array_equal(innov4.time, np.arange(21.0, 60.0))
        self.assertEqual(innov4.innov.shape, (2, 39))
        self.assertEqual(innov4.norm.shape, (2, 39))
        self.assertEqual(innov4.status.shape, (39,))


#%% aerospace.Kf
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_Kf(unittest.TestCase):
    r"""
    Tests the aerospace.Kf class with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.filename = get_tests_dir() / "test_kf.hdf5"
        self.date_zero = np.datetime64("2021-07-01T12:00:00", NP_DATETIME_UNITS)

    def test_nominal(self) -> None:
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf))
        self.assertEqual(kf.name, "")
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

    @unittest.skipIf(not HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_save_and_load(self) -> None:
        kf = space.Kf(num_points=2, num_states=3, num_axes=2, num_innovs=4)
        kf.chan = ["a", "b", "c"]
        kf.save(self.filename)
        kf2 = space.Kf.load(self.filename)
        self.assertEqual(kf.chan, kf2.chan)
        np.testing.assert_array_equal(kf.time, kf2.time)  # type: ignore[arg-type]

    def test_combine_nominal(self) -> None:
        kf1 = space.Kf(
            name="Name 1",
            units="m",
            chan=("a", "b", "c", "d", "e", "f"),
            innov_chan=("X", "Y"),
            num_points=30,
            num_states=6,
            time_dtype=float,
            num_innovs=10,
            num_axes=2,
        )
        kf1.time[:] = np.arange(30.0)  # type: ignore[index]
        kf1.innov.time[:] = np.arange(10.0, 20.0) + 0.2
        kf1.innov.innov[:] = np.random.rand(2, 10)
        kf2 = space.Kf(
            name="Name 2",
            units="m",
            chan=("a", "b", "c", "d", "e", "f"),
            innov_chan=("X", "Y"),
            num_points=60,
            num_states=6,
            time_dtype=float,
            num_innovs=30,
            num_axes=2,
        )
        kf2.time[:] = np.arange(30.0, 90.0)  # type: ignore[index]
        kf2.innov.time[:] = np.arange(40.0, 70.0) + 0.2
        kf2.innov.innov[:] = np.random.rand(2, 30)
        kf3 = kf1.combine(kf2)
        self.assertEqual(kf3.name, "Name 1")
        self.assertEqual(kf3.chan, ("a", "b", "c", "d", "e", "f"))
        np.testing.assert_array_equal(kf3.time, np.arange(90.0))  # type: ignore[arg-type]
        self.assertEqual(kf3.att.shape, (4, 90))  # type: ignore[union-attr]
        self.assertEqual(kf3.pos.shape, (3, 90))  # type: ignore[union-attr]
        self.assertEqual(kf3.vel.shape, (3, 90))  # type: ignore[union-attr]
        self.assertEqual(kf3.state.shape, (6, 90))  # type: ignore[union-attr]
        self.assertEqual(kf3.istate.shape, (6,))  # type: ignore[union-attr]
        self.assertEqual(kf3.innov.chan, ("X", "Y"))
        np.testing.assert_array_equal(kf3.innov.time, np.hstack((kf1.innov.time, kf2.innov.time)))
        self.assertEqual(kf3.innov.innov.shape, (2, 40))

    def test_combine_to_empty(self) -> None:
        kf1 = space.Kf()
        kf1.name = "Name 1"
        kf1.chan = ("dx", "dy", "dz")
        kf1.innov.units = "m"
        kf1.innov.chan = ("X", "Y")
        kf2 = space.Kf(
            name="Name 2",
            units="m",
            chan=("dx", "dy", "dz"),
            innov_chan=("X", "Y"),
            num_points=60,
            num_states=3,
            num_innovs=60,
            num_axes=2,
            time_dtype=float,
        )
        kf3 = kf1.combine(kf2)
        self.assertEqual(kf3.name, "Name 2")
        self.assertEqual(kf3.chan, ("dx", "dy", "dz"))
        self.assertEqual(kf3.innov.units, "m")
        self.assertEqual(kf3.innov.chan, ("X", "Y"))
        self.assertEqual(kf3.time.shape, (60,))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(kf3.time.dtype, np.floating))  # type: ignore[union-attr]
        self.assertEqual(kf3.att.shape, (4, 60))  # type: ignore[union-attr]
        self.assertEqual(kf3.innov.innov.shape, (2, 60))

    def test_combine_inplace(self) -> None:
        kf1 = space.Kf(
            name="Name 1",
            units="rad",
            chan=("a", "b", "c", "d", "e", "f"),
            innov_chan=("X", "Y", "Z"),
            num_points=30,
            num_states=6,
            num_innovs=10,
            num_axes=3,
            time_dtype=NP_DATETIME_FORM,
        )
        kf1.time[:] = self.date_zero + (10**9 * np.arange(30)).astype(NP_TIMEDELTA_FORM)  # type: ignore[index]
        kf1.innov.time[:] = self.date_zero + (10**9 * np.arange(10, 20) + 2 * 10**8).astype(NP_TIMEDELTA_FORM)
        kf1.innov.innov[:] = np.random.rand(3, 10)
        kf2 = space.Kf(
            name="Name 2",
            units="rad",
            chan=("a", "b", "c", "d", "e", "f"),
            innov_chan=("X", "Y", "Z"),
            num_points=60,
            num_states=6,
            num_innovs=30,
            num_axes=3,
            time_dtype=NP_DATETIME_FORM,
        )
        kf2.time[:] = self.date_zero + (10**9 * np.arange(30, 90)).astype(NP_TIMEDELTA_FORM)  # type: ignore[index]
        kf2.innov.time[:] = self.date_zero + (10**9 * np.arange(40, 70) + 2 * 10**8).astype(NP_TIMEDELTA_FORM)
        kf2.innov.innov[:] = np.random.rand(3, 30)
        kf3 = kf1.combine(kf2, inplace=True)
        self.assertIs(kf1, kf3)
        self.assertEqual(kf3.name, "Name 1")
        self.assertEqual(kf3.chan, ("a", "b", "c", "d", "e", "f"))
        self.assertEqual(kf3.innov.units, "rad")
        self.assertEqual(kf3.innov.chan, ("X", "Y", "Z"))
        self.assertEqual(kf3.time.shape, (90,))  # type: ignore[union-attr]
        self.assertTrue(np.issubdtype(kf3.time.dtype, np.datetime64))  # type: ignore[union-attr]
        self.assertEqual(kf3.att.shape, (4, 90))  # type: ignore[union-attr]
        self.assertEqual(kf3.innov.innov.shape, (3, 40))

    def test_chop(self) -> None:
        kf = space.Kf(
            name="Gnd",
            chan=("dx", "dy", "dz"),
            innov_chan=("X", "Y"),
            units="m",
            num_points=60,
            num_states=3,
            num_innovs=110,
            num_axes=2,
            time_dtype=float,
        )
        kf.time[:] = np.arange(60.0)  # type: ignore[index]
        kf.innov.time[:] = np.arange(5.0, 60.0, 0.5)
        kf.innov.innov[:] = np.random.rand(2, 110)
        kf2 = kf.chop(ti=10, tf=20, include_last=False)
        kf3 = kf.chop(ti=20, tf=40, include_last=True)
        self.assertIsNot(kf, kf2)
        self.assertEqual(kf2.name, "Gnd")
        self.assertEqual(kf2.chan, ("dx", "dy", "dz"))
        self.assertEqual(kf2.innov.units, "m")
        self.assertEqual(kf2.innov.chan, ("X", "Y"))
        self.assertEqual(kf3.name, "Gnd")
        self.assertEqual(kf3.chan, ("dx", "dy", "dz"))
        self.assertEqual(kf3.innov.units, "m")
        self.assertEqual(kf3.innov.chan, ("X", "Y"))
        np.testing.assert_array_equal(kf2.time, np.arange(10.0, 20.0))  # type: ignore[arg-type]
        np.testing.assert_array_equal(kf2.innov.time, np.arange(10.0, 20.0, 0.5))
        np.testing.assert_array_equal(kf3.time, np.arange(20.0, 41.0))  # type: ignore[arg-type]
        np.testing.assert_array_equal(kf3.innov.time, np.arange(20.0, 40.5, 0.5))
        self.assertEqual(kf2.att.shape, (4, 10))  # type: ignore[union-attr]
        self.assertEqual(kf2.innov.innov.shape, (2, 20))
        self.assertEqual(kf3.att.shape, (4, 21))  # type: ignore[union-attr]
        self.assertEqual(kf3.innov.innov.shape, (2, 41))

    def test_chop_inplace(self) -> None:
        kf = space.Kf(
            name="Gnd",
            chan=("dx", "dy", "dz"),
            innov_chan=("X", "Y"),
            units="m",
            num_points=60,
            num_states=3,
            num_innovs=110,
            num_axes=2,
            time_dtype=float,
        )
        kf.time[:] = np.arange(60.0)  # type: ignore[index]
        kf.innov.time[:] = np.arange(5.0, 60.0, 0.5)
        kf.innov.innov[:] = np.random.rand(2, 110)
        kf2 = kf.chop(ti=10, tf=20, inplace=True)
        self.assertIs(kf, kf2)
        self.assertEqual(kf.name, "Gnd")
        self.assertEqual(kf.chan, ("dx", "dy", "dz"))
        self.assertEqual(kf.innov.units, "m")
        self.assertEqual(kf.innov.chan, ("X", "Y"))
        np.testing.assert_array_equal(kf2.time, np.arange(10.0, 21.0))  # type: ignore[arg-type]
        self.assertEqual(kf2.att.shape, (4, 11))  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf2.innov.time, np.arange(10.0, 20.5, 0.5))
        self.assertEqual(kf2.innov.innov.shape, (2, 21))

    def test_chop_return_ends(self) -> None:
        kf = space.Kf(
            name="Gnd",
            chan=("dx", "dy", "dz"),
            innov_chan=("X", "Y"),
            units="m",
            num_points=60,
            num_states=3,
            num_innovs=110,
            num_axes=2,
            time_dtype=float,
        )
        kf.time[:] = np.arange(60.0)  # type: ignore[index]
        kf.innov.time[:] = np.arange(5.0, 60.0, 0.5)
        kf.innov.innov[:] = np.random.rand(2, 110)
        (kf3, kf2, kf4) = kf.chop(ti=10, tf=20, return_ends=True)
        self.assertIsNot(kf, kf2)
        for this in [kf, kf2, kf3, kf4]:
            self.assertEqual(this.name, "Gnd")
            self.assertEqual(this.chan, ("dx", "dy", "dz"))
            self.assertEqual(this.innov.units, "m")
            self.assertEqual(this.innov.chan, ("X", "Y"))
        np.testing.assert_array_equal(kf2.time, np.arange(10.0, 21.0))  # type: ignore[arg-type]
        self.assertEqual(kf2.att.shape, (4, 11))  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf2.innov.time, np.arange(10.0, 20.5, 0.5))
        self.assertEqual(kf2.innov.innov.shape, (2, 21))
        np.testing.assert_array_equal(kf3.time, np.arange(0.0, 10.0))  # type: ignore[arg-type]
        self.assertEqual(kf3.att.shape, (4, 10))  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf3.innov.time, np.arange(5.0, 10.0, 0.5))
        self.assertEqual(kf3.innov.innov.shape, (2, 10))
        np.testing.assert_array_equal(kf4.time, np.arange(21.0, 60.0))  # type: ignore[arg-type]
        self.assertEqual(kf4.att.shape, (4, 39))  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf4.innov.time, np.arange(20.5, 60.0, 0.5))
        self.assertEqual(kf4.innov.innov.shape, (2, 79))  # fix

    def tearDown(self) -> None:
        self.filename.unlink(missing_ok=True)


#%% aerospace.KfRecord
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_KfRecord(unittest.TestCase):
    r"""
    Tests the aerospace.KfRecord class with the following cases:
        Nominal
        With sizes
        Different time type
        Printing (x2)
        Combine (x3 - nominal, empty, inplace)
        Chopping (x3 - nominal, inplace, return ends)
    """

    def setUp(self) -> None:
        self.fields = ("time", "P", "stm", "H", "Pz", "K", "z")
        self.date_zero = np.datetime64("2021-07-01T12:00:00", NP_DATETIME_UNITS)

    def test_nominal(self) -> None:
        kf_record = space.KfRecord()
        self.assertIsInstance(kf_record, space.KfRecord)
        for key in self.fields:
            self.assertIsNone(getattr(kf_record, key), f"Expected None for field {key}")

    def test_arguments(self) -> None:
        kf_record = space.KfRecord(num_points=30, num_states=6, num_active=3, num_axes=2)
        assert kf_record.time is not None
        assert kf_record.P is not None
        assert kf_record.stm is not None
        assert kf_record.H is not None
        assert kf_record.Pz is not None
        assert kf_record.K is not None
        assert kf_record.z is not None
        self.assertEqual(kf_record.time.shape, (30,), "Time shape mismatch.")
        self.assertEqual(kf_record.P.shape, (3, 3, 30), "P shape mismatch.")
        self.assertEqual(kf_record.stm.shape, (3, 3, 30), "stm shape mismatch.")
        self.assertEqual(kf_record.H.shape, (2, 6, 30), "H shape mismatch.")
        self.assertEqual(kf_record.Pz.shape, (2, 2, 30), "Pz shape mismatch.")
        self.assertEqual(kf_record.K.shape, (3, 2, 30), "K shape mismatch.")
        self.assertEqual(kf_record.z.shape, (2, 30), "z shape mismatch.")

    def test_alternative_time(self) -> None:
        kf_record = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=NP_DATETIME_FORM)
        assert kf_record.time is not None
        assert kf_record.P is not None
        assert kf_record.stm is not None
        assert kf_record.H is not None
        assert kf_record.Pz is not None
        assert kf_record.K is not None
        assert kf_record.z is not None
        self.assertEqual(kf_record.time.dtype, "<M8[ns]")
        self.assertEqual(kf_record.time.shape, (60,), "Time shape mismatch.")
        self.assertEqual(kf_record.P.shape, (6, 6, 60), "P shape mismatch.")
        self.assertEqual(kf_record.stm.shape, (6, 6, 60), "stm shape mismatch.")
        self.assertEqual(kf_record.H.shape, (3, 9, 60), "H shape mismatch.")
        self.assertEqual(kf_record.Pz.shape, (3, 3, 60), "Pz shape mismatch.")
        self.assertEqual(kf_record.K.shape, (6, 3, 60), "K shape mismatch.")
        self.assertEqual(kf_record.z.shape, (3, 60), "z shape mismatch.")

    def test_pprint1(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as ctx:
            kf_record.pprint()
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "KfRecord")
        self.assertEqual(lines[1], " time = [0. 1. 2. 3. 4.]")

    def test_pprint2(self) -> None:
        kf_record = space.KfRecord(num_points=5)
        assert kf_record.time is not None
        kf_record.time[:] = np.arange(5)
        with capture_output() as ctx:
            kf_record.pprint(max_elements=0)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "KfRecord")
        self.assertEqual(lines[1], " time = <ndarray float64 (5,)>")

    def test_combine_nominal(self) -> None:
        kf_record1 = space.KfRecord(num_points=30, num_states=9, num_active=6, num_axes=3, time_dtype=float)
        kf_record1.time[:] = np.arange(30.0)  # type: ignore[index]
        kf_record2 = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=float)
        kf_record2.time[:] = np.arange(30.0, 90.0)  # type: ignore[index]
        kf_record3 = kf_record1.combine(kf_record2)
        np.testing.assert_array_equal(kf_record3.time, np.arange(90.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record3.P.shape, (6, 6, 90), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.stm.shape, (6, 6, 90), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.H.shape, (3, 9, 90), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.Pz.shape, (3, 3, 90), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.K.shape, (6, 3, 90), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.z.shape, (3, 90), "z shape mismatch.")  # type: ignore[union-attr]

    def test_combine_to_empty(self) -> None:
        kf_record1 = space.KfRecord()
        kf_record2 = space.KfRecord(num_points=60, num_states=6, num_active=3, num_axes=2, time_dtype=float)
        kf_record2.time[:] = np.arange(30.0, 90.0)  # type: ignore[index]
        kf_record3 = kf_record1.combine(kf_record2)
        np.testing.assert_array_equal(kf_record3.time, np.arange(30.0, 90.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record3.P.shape, (3, 3, 60), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.stm.shape, (3, 3, 60), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.H.shape, (2, 6, 60), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.Pz.shape, (2, 2, 60), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.K.shape, (3, 2, 60), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.z.shape, (2, 60), "z shape mismatch.")  # type: ignore[union-attr]

    def test_combine_inplace(self) -> None:
        kf_record1 = space.KfRecord(num_points=30, num_states=9, num_active=6, num_axes=3, time_dtype=NP_DATETIME_FORM)
        kf_record1.time[:] = self.date_zero + (10**9 * np.arange(30, dtype=np.int64)).astype(NP_TIMEDELTA_FORM)  # type: ignore[index]
        kf_record2 = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=NP_DATETIME_FORM)
        kf_record2.time[:] = self.date_zero + (10**9 * np.arange(30, 90, dtype=np.int64)).astype(NP_TIMEDELTA_FORM)  # type: ignore[index]
        kf_record3 = kf_record1.combine(kf_record2, inplace=True)
        self.assertIs(kf_record1, kf_record3)
        np.testing.assert_array_equal(
            kf_record3.time,  # type: ignore[arg-type]
            self.date_zero + (10**9 * np.arange(90, dtype=np.int64)).astype(NP_TIMEDELTA_FORM),
        )
        self.assertEqual(kf_record3.P.shape, (6, 6, 90), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.stm.shape, (6, 6, 90), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.H.shape, (3, 9, 90), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.Pz.shape, (3, 3, 90), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.K.shape, (6, 3, 90), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.z.shape, (3, 90), "z shape mismatch.")  # type: ignore[union-attr]

    def test_chop(self) -> None:
        kf_record = space.KfRecord(num_points=60, num_states=9, num_active=6, num_axes=3, time_dtype=float)
        self.assertEqual(kf_record.stm.shape, (6, 6, 60), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record.H.shape, (3, 9, 60), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record.Pz.shape, (3, 3, 60), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record.K.shape, (6, 3, 60), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record.z.shape, (3, 60), "z shape mismatch.")  # type: ignore[union-attr]
        kf_record.time[:] = np.arange(60.0)  # type: ignore[index]
        kf_record2 = kf_record.chop(ti=10, tf=20, include_last=False)
        kf_record3 = kf_record.chop(ti=20, tf=40, include_last=True)
        self.assertIsNot(kf_record, kf_record2)
        np.testing.assert_array_equal(kf_record.time, np.arange(60.0))  # type: ignore[arg-type]
        np.testing.assert_array_equal(kf_record2.time, np.arange(10.0, 20.0))  # type: ignore[arg-type]
        np.testing.assert_array_equal(kf_record3.time, np.arange(20.0, 41.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record2.stm.shape, (6, 6, 10))  # type: ignore[union-attr]
        self.assertEqual(kf_record2.H.shape, (3, 9, 10))  # type: ignore[union-attr]
        self.assertEqual(kf_record2.Pz.shape, (3, 3, 10))  # type: ignore[union-attr]
        self.assertEqual(kf_record2.K.shape, (6, 3, 10))  # type: ignore[union-attr]
        self.assertEqual(kf_record2.z.shape, (3, 10))  # type: ignore[union-attr]
        self.assertEqual(kf_record3.stm.shape, (6, 6, 21))  # type: ignore[union-attr]
        self.assertEqual(kf_record3.H.shape, (3, 9, 21))  # type: ignore[union-attr]
        self.assertEqual(kf_record3.Pz.shape, (3, 3, 21))  # type: ignore[union-attr]
        self.assertEqual(kf_record3.K.shape, (6, 3, 21))  # type: ignore[union-attr]
        self.assertEqual(kf_record3.z.shape, (3, 21))  # type: ignore[union-attr]

    def test_chop_inplace(self) -> None:
        kf_record = space.KfRecord(num_points=30, num_states=6, num_active=3, num_axes=2, time_dtype=float)
        kf_record.time[:] = np.arange(30.0)  # type: ignore[index]
        kf_record2 = kf_record.chop(ti=10, tf=20, inplace=True)
        self.assertIs(kf_record, kf_record2)
        np.testing.assert_array_equal(kf_record2.time, np.arange(10.0, 21.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record2.P.shape, (3, 3, 11), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.stm.shape, (3, 3, 11), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.H.shape, (2, 6, 11), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.Pz.shape, (2, 2, 11), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.K.shape, (3, 2, 11), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.z.shape, (2, 11), "z shape mismatch.")  # type: ignore[union-attr]

    def test_chop_return_ends(self) -> None:
        kf_record = space.KfRecord(num_points=60, num_states=6, num_active=3, num_axes=2, time_dtype=float)
        (kf_record3, kf_record2, kf_record4) = kf_record.chop(ti=10, tf=20, return_ends=True)
        self.assertIsNot(kf_record, kf_record2)
        np.testing.assert_array_equal(kf_record2.time, np.arange(10.0, 21.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record2.P.shape, (3, 3, 11), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.stm.shape, (3, 3, 11), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.H.shape, (2, 6, 11), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.Pz.shape, (2, 2, 11), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.K.shape, (3, 2, 11), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record2.z.shape, (2, 11), "z shape mismatch.")  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf_record3.time, np.arange(0.0, 10.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record3.P.shape, (3, 3, 10), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.stm.shape, (3, 3, 10), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.H.shape, (2, 6, 10), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.Pz.shape, (2, 2, 10), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.K.shape, (3, 2, 10), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record3.z.shape, (2, 10), "z shape mismatch.")  # type: ignore[union-attr]
        np.testing.assert_array_equal(kf_record4.time, np.arange(21.0, 60.0))  # type: ignore[arg-type]
        self.assertEqual(kf_record4.P.shape, (3, 3, 39), "P shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record4.stm.shape, (3, 3, 39), "stm shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record4.H.shape, (2, 6, 39), "H shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record4.Pz.shape, (2, 2, 39), "Pz shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record4.K.shape, (3, 2, 39), "K shape mismatch.")  # type: ignore[union-attr]
        self.assertEqual(kf_record4.z.shape, (2, 39), "z shape mismatch.")  # type: ignore[union-attr]


#%% Simple instantiation tests without numpy
class Test_classes_no_numpy(unittest.TestCase):
    r"""
    Tests instantiation of the following classes, whether numpy exists or not.
        KfInnov
    """

    def test_kfinnov(self) -> None:
        innov = space.KfInnov()
        self.assertTrue(isinstance(innov, space.KfInnov))

    def test_kf(self) -> None:
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf))

    def test_kfrecord(self) -> None:
        kf_record = space.KfRecord()
        self.assertIsInstance(kf_record, space.KfRecord)


#%% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
