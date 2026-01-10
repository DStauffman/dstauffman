r"""
Contains generic quaternion utilities optimized for use by keras (tensorflow, torch and/or jax).

Notes
-----
#.  Written by David C. Stauffer in November 2024.

"""

# %% Imports
from __future__ import annotations

import doctest
from typing import TYPE_CHECKING
import unittest

from dstauffman import HAVE_KERAS, HAVE_NUMPY

if HAVE_KERAS:
    import keras

    if keras.backend.backend() == "jax":
        # Enforces actual double precision calculations
        import jax.numpy as ops  # type: ignore[import-not-found]
    else:
        from keras import ops
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _Q = NDArray[np.floating]  # shape (..., 4)
    _V = NDArray[np.floating]  # shape (..., 3)


# %% Functions - enforce_pos_scalar_keras
def enforce_pos_scalar_keras(quat: _Q) -> _Q:
    """Forces the scalar component to always be positive."""
    ix = quat[..., 3] < 0
    if ops.ndim(ix) == 0:
        return -quat if ix else quat
    return ops.where(ix[:, None], -quat, quat)  # type: ignore[no-any-return]


# %% Functions - qrot_keras
def qrot_keras(axis: int, angle: float) -> _Q:
    r"""Construct a quaternion expressing a rotation about a single axis."""
    c = ops.cos(angle / 2)
    s = ops.sign(c) * ops.sin(angle / 2)
    # TODO: vectorize this (is it possible with tensorflow assign at once?)
    if axis == 1:
        quat = ops.array([s, 0.0, 0.0, ops.abs(c)], dtype="float64")
    elif axis == 2:
        quat = ops.array([0.0, s, 0.0, ops.abs(c)], dtype="float64")
    elif axis == 3:
        quat = ops.array([0.0, 0.0, s, ops.abs(c)], dtype="float64")
    return quat  # type: ignore[no-any-return]


# %% Functions - quat_inv_keras
def quat_inv_keras(quats: _Q) -> _Q:
    r"""Return the inverse of a normalized quaternions."""
    return ops.stack([-quats[..., 0], -quats[..., 1], -quats[..., 2], quats[..., 3]], axis=ops.ndim(quats) - 1)  # type: ignore[no-any-return]


# %% Functions - quat_norm_keras
def quat_norm_keras(quats: _Q) -> _Q:
    r"""Normalize quaternion."""
    return quats / ops.sqrt(ops.sum(quats**2, axis=-1, keepdims=True))  # type: ignore[no-any-return]


# %% Functions - quat_mult_keras
def quat_mult_keras(a: _Q, b: _Q) -> _Q:
    r"""Multiply quaternions together."""
    # fmt: off
    c = ops.transpose(ops.array([
         b[..., 0] * a[..., 3] + b[..., 1] * a[..., 2] - b[..., 2] * a[..., 1] + b[..., 3] * a[..., 0],
        -b[..., 0] * a[..., 2] + b[..., 1] * a[..., 3] + b[..., 2] * a[..., 0] + b[..., 3] * a[..., 1],
         b[..., 0] * a[..., 1] - b[..., 1] * a[..., 0] + b[..., 2] * a[..., 3] + b[..., 3] * a[..., 2],
        -b[..., 0] * a[..., 0] - b[..., 1] * a[..., 1] - b[..., 2] * a[..., 2] + b[..., 3] * a[..., 3],
    ], dtype=a.dtype))
    # fmt: on
    return quat_norm_keras(enforce_pos_scalar_keras(c))


# %% Functions - quat_prop_keras
def quat_prop_keras(quat: _Q, dx: _V) -> _Q:  # TODO: put into dstauffman.aerospace
    r"""Updates a quaternion from a given delta state."""
    # calculate magnitude of propagation
    dq_mag = ops.sqrt(ops.sum(dx**2, axis=-1, keepdims=True))
    non_zero = dq_mag > 1e-14
    den = ops.where(non_zero, dq_mag, ops.ones_like(dq_mag))
    fact = ops.sin(dq_mag / 2) / den
    vec = ops.where(non_zero, fact * dx, ops.zeros_like(dq_mag))
    scalar = ops.where(non_zero, ops.cos(dq_mag / 2), ops.ones_like(dq_mag))
    delta_quat = ops.hstack([vec, scalar])
    quat_new = quat_norm_keras(quat_mult_keras(delta_quat, quat))
    return quat_new


# %% Functions - quat_times_vector_keras
def quat_times_vector_keras(quat: _Q, v: _V) -> _Q:
    r"""Multiply quaternion(s) against vector(s)."""
    # fmt: off
    qv = ops.array([
         quat[..., 1] * v[..., 2] - quat[..., 2] * v[..., 1],
        -quat[..., 0] * v[..., 2] + quat[..., 2] * v[..., 0],
         quat[..., 0] * v[..., 1] - quat[..., 1] * v[..., 0],
    ], dtype=quat.dtype)
    skew_qv = ops.array([
         quat[..., 1] * ( quat[..., 0] * v[..., 1] - quat[..., 1] * v[..., 0]) - quat[..., 2] * (-quat[..., 0] * v[..., 2] + quat[..., 2] * v[..., 0]),
        -quat[..., 0] * ( quat[..., 0] * v[..., 1] - quat[..., 1] * v[..., 0]) + quat[..., 2] * ( quat[..., 1] * v[..., 2] - quat[..., 2] * v[..., 1]),
         quat[..., 0] * (-quat[..., 0] * v[..., 2] + quat[..., 2] * v[..., 0]) - quat[..., 1] * ( quat[..., 1] * v[..., 2] - quat[..., 2] * v[..., 1]),
    ], dtype=quat.dtype)
    # fmt: on
    return v + ops.transpose(2 * (-quat[..., 3] * qv + skew_qv))  # type: ignore[no-any-return]


# %% Functions - quat_angle_diff_keras
def quat_angle_diff_keras(quat1: _Q, quat2: _Q) -> _Q:
    r"""Calculate the angular difference between two quaternions."""
    dq = quat_mult_keras(quat2, quat_inv_keras(quat1))
    dv = dq[..., :3]
    mag2 = ops.sum(dv**2, axis=-1, keepdims=True)
    mag = ops.sqrt(mag2)
    theta_over_2 = ops.arcsin(mag)
    theta = 2 * theta_over_2
    fact = ops.where(mag == 0, ops.ones_like(mag), mag)
    nv = dv / fact
    return nv * theta  # type: ignore[no-any-return]


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_quat_keras", exit=False)
    doctest.testmod(verbose=False)
