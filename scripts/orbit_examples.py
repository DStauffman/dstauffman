"""
Example script that matches the "Introduction to Space Flight" by Francis J Hale.
"""

import numpy as np

import dstauffman as dcs
import dstauffman.aerospace as space

line1 = "1 25544U 98067A   06132.29375000  .00013633  00000-0  92740-4 0  9181"
line2 = "2 25544  51.6383  12.2586 0009556 188.7367 320.5459 15.75215761427503"

oe = space.two_line_elements(line1, line2)

(r, v) = space.oe_2_rv(oe, mu=space.MU_EARTH)
oe2 = space.rv_2_oe(r, v, mu=space.MU_EARTH)

print(r, v)
print(r / space.EARTH["a"], v / space.EARTH["a"])
oe.print_orrery()
oe.pprint()
oe2.pprint()

np.testing.assert_array_almost_equal(oe.a, oe2.a, err_msg="a is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.e, oe2.e, err_msg="e is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.i, oe2.i, err_msg="i is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.w, oe2.w, err_msg="w is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.W, oe2.W, err_msg="W is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.vo, oe2.vo, err_msg="nu is different")  # type: ignore[arg-type]

# %% Intro to Space Flight Example
r = np.array([-0.8, 0.6, 0.5])
v = np.array([-0.4, -0.8, 0.6])
mu = 1.0

r_mag = dcs.magnitude(r)
print(f"{r_mag=}")

v_mag = dcs.magnitude(v)
print(f"{v_mag=}")

E = v_mag**2 / 2 - mu / r_mag
print(f"{E=}")

a = -mu / (2 * E)
print(f"{a=}")

e = 1 / mu * ((v_mag**2 - mu / r_mag) * r - np.sum(r * v, axis=0) * v)
print(f"{e=}")

e_mag = dcs.magnitude(e)
print(f"{e_mag=}")

h = np.cross(r, v)
print(f"{h=}")

h_mag = dcs.magnitude(h)
print(f"{h_mag=}")

p = dcs.magnitude(h) ** 2 / mu
print(f"{p=}")

i = np.arccos(h[2] / h_mag)
print(f"{i=}; {dcs.RAD2DEG*i=}")

n = np.cross(np.array([0, 0, 1]), h)
print(f"{n=}")

n_mag = dcs.magnitude(n)
print(f"{n_mag=}")

W = np.arccos(n[0] / n_mag)
if n[1] < 0:
    W = 2 * np.pi - W
print(f"{W=}; {dcs.RAD2DEG*W=}")

w = np.arccos(np.dot(n, e) / (n_mag * e_mag))
if e[2] < 0:
    w = 2 * np.pi - w
print(f"{w=}; {dcs.RAD2DEG*w=}")

vo = np.arccos(np.dot(e, r) / (e_mag * r_mag))
if np.dot(r, v) < 0:
    vo = 2 * np.pi - vo
print(f"{vo=}; {dcs.RAD2DEG*vo=}")

T = 2 * np.pi * np.sqrt(a**3 / mu)
print(f"{T=}")

u = np.arccos((e_mag + np.cos(vo)) / (1 + e_mag * np.cos(vo)))
print(f"{u=}; {dcs.RAD2DEG*u=}")

M = u - e_mag * np.sin(u)
print(f"{M=}; {dcs.RAD2DEG*M=}")

t = T / (2 * np.pi) * M
print(f"{t=}")

uo = w + vo
print(f"{uo=}; {dcs.RAD2DEG*uo=}")

P = W + w
print(f"{P=}; {dcs.RAD2DEG*P=}")

lo = W + uo
print(f"{lo=}; {dcs.RAD2DEG*lo=}")

oe = space.rv_2_oe(r, v, mu=mu)
oe.pprint()

np.testing.assert_array_almost_equal(oe.a, a, err_msg="a is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.e, e_mag, err_msg="e is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.i, i, err_msg="i is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.w, w, err_msg="w is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.W, W, err_msg="W is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.vo, vo, err_msg="nu is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.p, p, err_msg="p is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.uo, uo, err_msg="uo is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.P, P, err_msg="Pi is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.lo, lo, err_msg="lo is different")  # type: ignore[arg-type]
np.testing.assert_array_almost_equal(oe.T, T, err_msg="T is different")  # type: ignore[arg-type]
# np.testing.assert_array_almost_equal(oe.t, t, err_msg="t is different")  # type: ignore[arg-type]


# %% Example 3
r = np.array([-0.0707033342747565, 0.07070333427475654, 0.00027774964281135])
v = np.array([-2.2585415009051526, -2.2138180776640843, 8.7845325527614843e-05])
oe = space.rv_2_oe(r, v)
