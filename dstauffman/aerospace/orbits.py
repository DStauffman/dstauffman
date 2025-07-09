r"""
Classes and functions related to orbits and orbit determination and propagation.

Notes
-----
#.  Written by David C. Stauffer in July 2021.

"""

# %% Imports
from __future__ import annotations

import copy
import datetime
import doctest
from typing import Any, Literal, overload, TYPE_CHECKING
import unittest

from slog import IntEnumPlus, is_dunder

from dstauffman import DEGREE_SIGN, Frozen, HAVE_NUMPY, NP_DATETIME_FORM, NP_NAT, NP_ONE_DAY, RAD2DEG
from dstauffman.aerospace.orbit_const import JULIAN, MU_EARTH, PI, TAU
from dstauffman.aerospace.orbit_conv import (
    anomaly_eccentric_2_mean,
    anomaly_eccentric_2_true,
    anomaly_hyperbolic_2_mean,
    anomaly_hyperbolic_2_true,
    anomaly_mean_2_eccentric,
    anomaly_mean_2_hyperbolic,
    anomaly_true_2_eccentric,
    anomaly_true_2_hyperbolic,
    anomaly_true_2_mean,
    mean_motion_2_semimajor,
)
from dstauffman.aerospace.orbit_support import cross, d_2_r, dot, jd_to_numpy, norm, r_2_d
from dstauffman.aerospace.quat import quat_times_vector

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _Q = NDArray[np.floating]  # (4,)
    _FN = float | np.floating | _N


# %% Enums - OrbitType
class OrbitType(IntEnumPlus):
    r"""
    Values for the possible orbit types, from {"elliptic", "parabolic", "hyperbolic"}.

    Plus an "uninitialized" value.  Note that "circular" is not an option, but falls under elliptic.

    Examples
    --------
    >>> from dstauffman.aerospace import OrbitType
    >>> print(OrbitType.elliptic)
    OrbitType.elliptic: 1

    """

    uninitialized = 0
    elliptic = 1
    parabolic = 2
    hyperbolic = 3


# %% Classes - Elements
class Elements(Frozen):
    r"""
    Orbital Elements class.

    Attributes
    ----------
    a          : semi-major axis
    e          : eccentricity
    i          : inclination
    W          : longitude of the ascending node
    w          : argument of periapsis
    vo         : true anomaly at epoch
    p          : parameter
    uo         : argument of latitude
    P          : longitude of periapsis
    lo         : true longitude at epoch
    T          : period
    type       : field specifying type of orbit
    circular   : (optional) field specifying true or false if elliptic orbit is also circular or not
    equatorial : (optional) field specifying true or false if elliptic orbit is also in the equatorial plane
    t          : epoch time

    Examples
    --------
    >>> from dstauffman.aerospace import Elements
    >>> elements = Elements()
    >>> elements.pprint()
    Elements
     a          = []
     e          = []
     i          = []
     W          = []
     w          = []
     vo         = []
     p          = []
     uo         = []
     P          = []
     lo         = []
     T          = []
     type       = []
     equatorial = []
     circular   = []
     t          = []

    """

    def __init__(self, num: int = 0):
        is_single = num == 1
        # fmt: off
        self.a: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.e: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.i: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.W: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.w: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.vo: _FN         = np.nan if is_single else np.full(num, np.nan)
        self.p: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.uo: _FN         = np.nan if is_single else np.full(num, np.nan)
        self.P: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.lo: _FN         = np.nan if is_single else np.full(num, np.nan)
        self.T: _FN          = np.nan if is_single else np.full(num, np.nan)
        self.type: OrbitType | _I  = OrbitType.uninitialized if is_single else np.full(num, OrbitType.uninitialized, dtype=int)
        self.equatorial: bool | _B = False if is_single else np.zeros(num, dtype=bool)
        self.circular: bool | _B   = False if is_single else np.zeros(num, dtype=bool)
        self.t: np.datetime64 | _D = NP_NAT if is_single else np.full(num, NP_NAT, dtype=NP_DATETIME_FORM)
        # fmt: on

    def __eq__(self, other: Any) -> bool:
        # if not of the same type, then they are not equal
        if not isinstance(other, type(self)):
            return False
        # loop through the fields, and if any are not equal, then it's not equal
        for key in vars(self):
            x = getattr(self, key)
            y = getattr(other, key)
            if key in {"type", "equatorial", "circular"}:
                if not np.all(x == y):
                    return False
            elif key in {"t"}:
                if not np.all((np.isnat(x) & np.isnat(y)) | (x == y)):
                    return False
            elif not np.allclose(x, y, rtol=1e-6, atol=1e-8, equal_nan=True):
                return False
        # if it made it all the way through the fields, then things must be equal
        return True

    def __len__(self) -> int:
        return np.size(self.a)

    def __getitem__(self, key: int) -> Elements:
        elements = Elements(1)
        for field, value in vars(self).items():
            setattr(elements, field, value[key])
        return elements

    def combine(self, elements2: Elements, /, *, inplace: bool = False) -> Elements:
        r"""Combines two KfInnov structures together."""
        # allow an empty structure to be passed through
        if np.size(self.a) == 0:
            if inplace:
                for key, value in vars(elements2).items():
                    setattr(self, key, value)
            return elements2  # TODO: make a copy?
        # concatenate fields
        if inplace:
            elements = self
        else:
            elements = copy.deepcopy(self)
        if np.size(elements2.a) == 0:
            return elements
        for key, value in vars(self).items():
            setattr(elements, key, np.hstack((value, getattr(elements2, key))))
        return elements

    def print_keplerian(self, index: int | None = None) -> None:
        r"""Prints the keplerian orbital elements in typical km/deg units."""
        this = self if index is None else self[index]
        M = anomaly_true_2_mean(this.vo, this.e)
        print(f"a = {this.a / 1000} km")
        print(f"e = {this.e}")
        print(f"i = {this.i * RAD2DEG} {DEGREE_SIGN}")
        print(f"\N{GREEK CAPITAL LETTER OMEGA} = {this.W * RAD2DEG} {DEGREE_SIGN}")
        print(f"\N{GREEK SMALL LETTER OMEGA} = {this.w * RAD2DEG} {DEGREE_SIGN}")
        print(f"vo = {this.vo * RAD2DEG} {DEGREE_SIGN}")
        print(f"M = {M * RAD2DEG} {DEGREE_SIGN}")

    def print_equinoctial(self, index: int | None = None) -> None:
        r"""Prints the orbital elements in equinoctial form in typical km/deg units."""
        this = self if index is None else self[index]
        M = anomaly_true_2_mean(this.vo, this.e)  # mean anomaly
        p = this.a * (1.0 - this.e**2)  # parameter
        f = this.e * np.cos(this.W + this.w)  # eccentric cosine
        g = this.e * np.sin(this.W + this.w)  # eccentric sine
        h = np.tan(this.i / 2) * np.cos(this.W)  # inclined cosine
        k = np.tan(this.i / 2) * np.sin(this.W)  # inclined sine
        L = np.mod(this.W + this.w + this.vo, TAU)  # true longitude
        lambda_ = np.mod(this.W + this.w + M, TAU)  # mean longitude
        print(f"a = {this.a / 1000} km  # semimajor axis")
        print(f"p = {p / 1000} km  # parameter")
        print(f"f = {f * RAD2DEG} {DEGREE_SIGN}  # eccentric cosine")
        print(f"g = {g * RAD2DEG} {DEGREE_SIGN}  # eccentric sine")
        print(f"h = {h * RAD2DEG} {DEGREE_SIGN}  # inclined cosine")
        print(f"k = {k * RAD2DEG} {DEGREE_SIGN}  # inclined sine")
        print(f"L = {L * RAD2DEG} {DEGREE_SIGN}  # true longitude")
        print(f"\N{GREEK SMALL LETTER LAMDA} = {lambda_ * RAD2DEG} {DEGREE_SIGN}  # mean longitude")


# %% Functions - _zero_divide
def _zero_divide(x: _FN, y: _FN) -> _N:
    """Divide two values, but return zero when the denominator is zero."""
    return np.divide(x, y, where=y != 0, out=np.zeros_like(y))  # type: ignore[no-any-return]


# %% Functions - _inf_divide
def _inf_divide(x: _FN, y: _FN) -> _N:
    """Divde two values, but return positive infinity when the denominator is zero."""
    return np.divide(x, y, where=y != 0, out=np.full_like(y, np.inf))  # type: ignore[no-any-return]


# %% Functions - _inf_multiply
def _inf_multiply(x: _FN, y: _FN) -> _N:
    """Multiply that returns zero for inf * 0 instead of nan."""
    return np.multiply(x, y, where=(x != 0.0) & (y != 0.0), out=np.zeros_like(x))  # type: ignore[no-any-return]


# %% Functions - _fix_instab
def _fix_instab(x: _N, precision: float) -> None:
    """Fix instabilities near positive/negative one before taking an arcsin by holding to +/-1."""
    ix = (x > 1.0) & (x < 1.0 + precision)
    if np.any(ix):
        x[ix] = 1.0
    ix = (x < -1.0) & (x > -1 - precision)
    if np.any(ix):
        x[ix] = -1.0


# %% Functions - two_line_elements
def two_line_elements(line1: str, line2: str) -> Elements:  # noqa: C901
    r"""
    Convert two-line elements to elements structure.

    Parameters
    ----------
    line1 : str
        First line of two line elements
    line2 : str
        Second line of two line elements

    Returns
    -------
    elements : class Elements
        Orbit Elements

    Notes
    -----
    #.  Adapted by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.
    #.  See https://celestrak.com/NORAD/elements/ such as GPS satellites:
        https://celestrak.com/NORAD/elements/gps-ops.txt

    Examples
    --------
    >>> from dstauffman.aerospace import two_line_elements
    >>> line1 = "1 25544U 98067A   06132.29375000  .00013633  00000-0  92740-4 0  9181"
    >>> line2 = "2 25544  51.6383  12.2586 0009556 188.7367 320.5459 15.75215761427503"
    >>> elements = two_line_elements(line1, line2)
    >>> elements.pprint()
    Elements
     a          = 6722154.278502964
     e          = 0.0009556
     i          = 0.9012583551325879
     W          = 0.21395293168497687
     w          = 3.294076834348782
     vo         = 5.593365747043137
     p          = 6722148.1400242
     uo         = 8.88744258139192
     P          = 3.5080297660337587
     lo         = 9.101395513076895
     T          = 5484.96289455295
     type       = OrbitType.elliptic: 1
     equatorial = False
     circular   = False
     t          = 2006-05-12T07:03:00.000016096

    """
    if not line1.startswith("1"):
        raise ValueError(f'line1 must start with "1": {line1}')
    if not line2.startswith("2"):
        raise ValueError(f'line2 must start with "2": {line2}')
    if len(line1) != 69:
        raise ValueError(f"line1 must have length 69: {line1}")
    if len(line2) != 69:
        raise ValueError(f"line2 must have length 69: {line2}")

    try:
        year = int(line1[18:20])
    except Exception as exc:
        raise ValueError(f"Error reading year from line1: {line1}") from exc

    try:
        day = float(line1[20:32])
    except Exception as exc:
        raise ValueError(f"Error reading day from line1: {line1}") from exc

    try:
        i = float(line2[8:16])
    except Exception as exc:
        raise ValueError("fError reading inclination (i) from line2: {line2}") from exc
    i = d_2_r(i)

    try:
        Omega = float(line2[17:25])
    except Exception as exc:
        raise ValueError(f"Error reading longitude of ascending node (Omega) from line2: {line2}") from exc
    Omega = d_2_r(Omega)

    try:
        e = float("0." + line2[26:33])
    except Exception as exc:
        raise ValueError(f"Error reading eccentricity (e) from line2: {line2}") from exc

    try:
        omega = float(line2[34:42])
    except Exception as exc:
        raise ValueError(f"Error reading argument of perigee (omega) from line2: {line2}") from exc
    omega = d_2_r(omega)

    try:
        M = float(line2[43:51])
    except Exception as exc:
        raise ValueError(f"Error reading mean anomaly (M) from line2: {line2}") from exc
    M = d_2_r(M)

    try:
        revs_per_day = float(line2[52:63])
    except Exception as exc:
        raise ValueError(f"Error reading revolutions per day from line2: {line2}") from exc

    n = revs_per_day * TAU / JULIAN["day"]  # [rad/sec]
    a = float(mean_motion_2_semimajor(n, MU_EARTH))
    E = float(anomaly_mean_2_eccentric(M, e))
    nu = float(anomaly_eccentric_2_true(E, e))
    p = float(a * (1.0 - e**2))

    # fmt: off
    time = (
        (datetime.datetime(2000 + year, 1, 1, 0, 0, 0) - datetime.datetime(2000, 1, 1, 0, 0, 0)).days
        + JULIAN["jd_2000_01_01"] - 0.5 + day - 1
    )

    elements            = Elements(1)
    elements.equatorial = False
    elements.circular   = False
    elements.a          = a
    elements.e          = e
    elements.i          = i
    elements.W          = Omega
    elements.w          = omega
    elements.vo         = nu
    elements.p          = p
    elements.uo         = omega + nu
    elements.P          = Omega + omega
    elements.lo         = Omega + omega + nu
    elements.T          = TAU * np.sqrt(a ** 3 / MU_EARTH)
    elements.t          = jd_to_numpy(time)
    elements.type       = OrbitType.elliptic
    # fmt: on

    return elements


# %% Functions - rv_2_oe
def rv_2_oe(r: _N, v: _N, mu: _FN = 1.0, unit: bool = False, precision: float = 1e-12) -> Elements:
    r"""
    Position and Velocity to Orbital Elements.

    Parameters
    ----------
    r : (3, N) ndarray
        Position vector (in ECI)
    v : (3, N) ndarray
        Velocity vector (in ECI)
    mu : float or (N, ) ndarray, optional
        Gravitational constant times the sum of the masses
    unit : bool, optional
        Flag specifying radians (0, default) or degrees (1) for output
    precision : float, optional
        Value before something is considered zero, like eccentricity or inclination

    Returns
    -------
    elements : class Elements
        Orbital elements

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 24 April 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import rv_2_oe
    >>> import numpy as np
    >>> r = np.array([1, 0, 0])
    >>> v = np.array([0, 1, 0])
    >>> elements = rv_2_oe(r, v)
    >>> elements.pprint()
    Elements
     a          = 1.0
     e          = 0.0
     i          = 0.0
     W          = 0.0
     w          = 0.0
     vo         = 0.0
     p          = 1.0
     uo         = 0.0
     P          = 0.0
     lo         = 0.0
     T          = 6.283185307179586
     type       = 1
     equatorial = True
     circular   = True
     t          = NaT

    """
    # calculate angular momentum
    h = cross(r, v)
    norm_r = norm(r)
    norm_v = norm(v)
    norm_h = norm(h)

    # calculations
    z = np.zeros(h.shape)
    z[2, ...] = 1.0
    n = cross(z, h)
    n_mag = norm(n)
    ix = n_mag == 0.0
    if np.any(ix):
        n[..., ix] = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, np.count_nonzero(ix)))  # type: ignore[arg-type]
        n_mag[ix] = 1.0

    e = 1.0 / mu * ((norm_v**2 - _zero_divide(mu, norm_r)) * r - dot(r, v) * v)

    # e
    e_mag = norm(e)

    # get sizing information
    num = e_mag.size

    # p
    p = norm_h**2 / mu

    # a
    # Note: a is infinite when eccentricity is exactly one
    a = np.divide(p, 1.0 - e_mag**2, where=e_mag != 1.0, out=np.full_like(p, np.inf))

    # i
    i = np.arccos(_zero_divide(h[2, ...], norm_h), where=norm_h != 0.0, out=np.zeros_like(norm_h))

    # W
    W = np.asanyarray(np.arccos(n[0, ...] / n_mag))
    W = np.subtract(TAU, W, out=W, where=n[1, ...] < -precision)

    # w
    ix = np.abs(e_mag) >= precision
    w = np.divide(dot(n, e), n_mag * e_mag, where=ix, out=np.zeros_like(e_mag))
    _fix_instab(w, precision=precision)
    w = np.arccos(w, where=ix, out=w)
    w = np.subtract(TAU, w, out=w, where=ix & (e[2, ...] < -precision))
    w[ix & ~np.isreal(w)] = PI

    # vo
    ix &= norm_r > precision
    vo = np.divide(dot(e, r), e_mag * norm_r, where=ix, out=np.zeros_like(e_mag))
    _fix_instab(vo, precision=precision)
    vo = np.arccos(vo, where=ix, out=vo)
    vo = np.subtract(TAU, vo, out=vo, where=ix & (dot(r, v) < -precision))

    # uo
    uo = np.asanyarray(np.mod(w + vo, TAU))
    uo[np.abs(uo - TAU) < precision] = 0.0

    # P
    P = np.asanyarray(np.mod(W + w, TAU, where=~np.isnan(W), out=np.full_like(W, np.nan)))
    P[np.abs(P - TAU) < precision] = 0.0

    # lo
    lo = np.asanyarray(np.mod(W + w + vo, TAU, where=~np.isnan(W), out=np.full_like(W, np.nan)))
    lo[np.abs(lo - TAU) < precision] = 0.0

    # tell if equatorial
    equatorial = (np.abs(i) < precision) | (np.abs(i - PI) < precision)

    # convert to degrees specified
    if unit:
        # fmt: off
        i  = r_2_d(i)
        W  = r_2_d(W)
        w  = r_2_d(w)
        vo = r_2_d(vo)
        uo = r_2_d(uo)
        P  = r_2_d(P)
        lo = r_2_d(lo)
        # fmt: on

    # allocate stuff
    orbit_type = np.full(num, OrbitType.uninitialized, dtype=int)

    # specific energy
    E = norm_v**2 / 2 - _inf_divide(mu, norm_r)
    # test for type of orbit - elliptic
    ix = E < 0.0
    orbit_type[ix] = OrbitType.elliptic
    # T (only defined for elliptic, as parabolic and hyperbolic don't have a period)
    T = TAU * np.sqrt(a**3 / mu, where=ix, out=np.full_like(a, np.nan))
    # dt = np.divide(T / TAU * M, where=ix, out=np.full_like(T, np.nan))
    circular = np.where(ix & (np.abs(e_mag) < precision), True, False)
    # parabolic
    ix = E == 0.0
    orbit_type[ix] = OrbitType.parabolic
    # hyperbolic
    ix = E > 0.0
    orbit_type[ix] = OrbitType.hyperbolic

    # populate elements structure
    # fmt: off
    elements            = Elements(num)
    elements.a          = a
    elements.e          = e_mag
    elements.i          = i
    elements.W          = W
    elements.w          = w
    elements.vo         = vo
    elements.p          = p
    elements.uo         = uo
    elements.P          = P
    elements.lo         = lo
    elements.equatorial = equatorial
    elements.type       = orbit_type if num > 1 else orbit_type[0]
    elements.T          = T
    elements.circular   = circular
    # fmt: on

    return elements


# %% oe_2_rv
@overload
def oe_2_rv(elements: Elements, mu: _FN = ..., unit: bool = ..., *, return_PQW: Literal[False] = ...) -> tuple[_N, _N]: ...
@overload
def oe_2_rv(elements: Elements, mu: _FN = ..., unit: bool = ..., *, return_PQW: Literal[True]) -> tuple[_N, _N, _N, _N, _Q]: ...
def oe_2_rv(
    elements: Elements, mu: _FN = 1.0, unit: bool = False, *, return_PQW: bool = False
) -> tuple[_N, _N] | tuple[_N, _N, _N, _N, _Q]:
    r"""
    Orbital Elements to Position and Velocity.

    This function takes a given elements structure containing subfields for
    a, e, i, W, w, vo scalars and calculates the position and velocity vectors r & v.

    Parameters
    ----------
    elements : class Elements
        Orbital elements
    mu : float, optional
        Gravitational constant times the sum of the masses
    unit : bool, optional, default is False
        Units to return, where True changes to degrees
    return_PQW : bool, optional, default is False
        Whether to return position and velocity in PQW (perifocal coordinate) frame,
        along with the transformation

    Returns
    -------
    r     : 3x1 position vector
    v     : 3x1 velocity vector
    (The next three values are usually treated as optional outputs)
    r_PQW : 3x1 position vector in PQW frame
    v_PQW : 3x1 velocity vector in PQW frame
    Q     : transformation quaternion

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 24 April 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import Elements, oe_2_rv
    >>> elements = Elements(1)
    >>> elements.a  = 1.0
    >>> elements.e  = 0.0
    >>> elements.i  = 0.0
    >>> elements.W  = 0.0
    >>> elements.w  = 0.0
    >>> elements.vo = 0.0
    >>> (r, v) = oe_2_rv(elements)
    >>> print(r)
    [1. 0. 0.]

    >>> print(v)
    [0. 1. 0.]

    """
    # pull out short names from structure
    a = elements.a
    e = elements.e
    i = elements.i
    W = elements.W
    w = elements.w
    nu = elements.vo
    num = np.size(a)

    # adjust if angles are in degrees
    if unit:
        # fmt: off
        i  = d_2_r(i)
        W  = d_2_r(W)
        w  = d_2_r(w)
        nu = d_2_r(nu)
        # fmt: on

    # % calculations
    # parameter
    if np.all(~np.isnan(elements.p)):
        p = elements.p
    else:
        p = _inf_multiply(a, (1.0 - e**2))

    # magnitude of r in PQW frame
    r_mag = _zero_divide(p, (1.0 + e * np.cos(nu)))

    # r in PQW frame
    r_PQW = np.vstack([_inf_multiply(r_mag, np.cos(nu)), _inf_multiply(r_mag, np.sin(nu)), np.zeros(num)])

    # v in PQW frame
    v_PQW = np.sqrt(_zero_divide(mu, p)) * np.vstack([-np.sin(nu), e + np.cos(nu), np.zeros(num)])

    # cosine and sine terms
    ci2 = np.cos(i / 2)
    si2 = np.sin(i / 2)
    cwm = np.cos((W - w) / 2)
    cwp = np.cos((W + w) / 2)
    swm = np.sin((W - w) / 2)
    swp = np.sin((W + w) / 2)

    # transformation quaternion
    # import sympy
    # w, i, W = sympy.symbols('w i W')
    # Q = sympy.Quaternion.from_euler([-w, -i, -W], "ZXZ")
    Q = np.vstack([-si2 * cwm, -si2 * swm, -swp * ci2, +ci2 * cwp])

    # translate r & v into IJK frame
    r = quat_times_vector(Q, r_PQW)
    v = quat_times_vector(Q, v_PQW)

    if num == 1:
        if return_PQW:
            return (np.squeeze(r), np.squeeze(v), np.squeeze(r_PQW), np.squeeze(v_PQW), np.squeeze(Q))
        return (np.squeeze(r), np.squeeze(v))

    if return_PQW:
        return (r, v, r_PQW, v_PQW, Q)
    return (r, v)


# %% Functions - advance_true_anomaly
def advance_true_anomaly(a: _FN, e: _FN, vo: _FN, mu: _FN, time: _FN) -> _FN:
    """
    Advances the true anomaly.

    Takes the semimajor axis and eccentricity, along with mu (gravity constant) to advance the
    given true anomaly by the given time.

    Parameters
    ----------
    a : float or numpy.ndarray
        Semimajor axis
    e : float or numpy.ndarray
        Eccentricity
    vo : float or numpy.ndarray
        True Anomaly
    mu : float or numpy.ndarray
        Gravitational Constant for the large body
    time : float or numpy.ndarray
        Delta time in seconds to advance the true anomaly by

    Examples
    --------
    >>> from dstauffman.aerospace import advance_true_anomaly
    >>> a = 7e6
    >>> e = 0.0
    >>> vo = 0.0
    >>> mu = 3.9863e14
    >>> time = 600.0
    >>> nu = advance_true_anomaly(a, e, vo, mu, time)
    >>> print(nu)  # doctest: +ELLIPSIS
    0.646828549162...

    """
    if np.all(e < 1.0):  # elliptic
        # initial mean anomaly
        Ei = anomaly_true_2_eccentric(vo, e)
        Mi = anomaly_eccentric_2_mean(Ei, e)
        # find new mean anomaly based on delta time
        M = np.mod(np.sqrt(mu / a**3) * time + Mi, TAU)
        # solve transcendental function for E
        E = anomaly_mean_2_eccentric(M, e)
        # calculate the new true anomaly from the eccentric anomaly
        nu = anomaly_eccentric_2_true(E, e)
    elif np.all(e == 1.0):  # parabolic
        Ei = np.array(0.0)
        Mi = anomaly_eccentric_2_mean(Ei, e)  # find new mean anomaly based on delta time
        M = np.mod(np.sqrt(mu / a**3) * time + Mi, TAU)
        # solve transcendental function for E
        E = anomaly_mean_2_eccentric(M, e)
        # calculate the new true anomaly from the eccentric anomaly
        nu = anomaly_eccentric_2_true(E, e)
    elif np.all(e > 1.0):  # hyperbolic
        Fi = anomaly_true_2_hyperbolic(vo, e)
        Mi = anomaly_hyperbolic_2_mean(Fi, e)
        # find new mean anomaly based on delta time
        M = np.mod(np.sqrt(mu / (-a) ** 3) * time + Mi, TAU)
        # solve transcendental function for F
        F = anomaly_mean_2_hyperbolic(M, e)
        # calculate the new true anomaly from the eccentric anomaly
        nu = anomaly_hyperbolic_2_true(F, e)
    else:
        raise ValueError(f'Unexpected orbit type: "{type}"')
    return nu


# %% Functions - advance_elements
def advance_elements(elements: Elements, mu: _FN, time: _FN) -> Elements:
    r"""
    Takes the given orbital elements and advancing them in time.

    Parameters
    ----------
    elements : class Elements
        Orbital Elements to advance
    mu : float
        Gravitational Constant for the large body
    time : float
        Delta time in seconds to advance the elements by

    Notes
    -----
    #.  The only parameters that will change are the true anomaly and the time stamp.

    Examples
    --------
    >>> from dstauffman.aerospace import advance_elements, Elements, OrbitType, PI
    >>> elements = Elements(1)
    >>> elements.a = 7e6
    >>> elements.e = 0.0
    >>> elements.i = PI / 4
    >>> elements.W = 0.0
    >>> elements.w = 0.0
    >>> elements.vo = 0.0
    >>> elements.t = np.datetime64("2023-06-01T00:00:00")
    >>> elements.type = OrbitType.elliptic
    >>> elements.equatorial = True
    >>> elements.circular = True
    >>> mu = 3.9863e14
    >>> time = 600.0
    >>> new_elements = advance_elements(elements, mu, time)
    >>> new_elements.pprint()  # doctest: +ELLIPSIS
    Elements
     a          = 7000000.0
     e          = 0.0
     i          = 0.7853981633974...
     W          = 0.0
     w          = 0.0
     vo         = 0.646828549162...
     p          = nan
     uo         = nan
     P          = nan
     lo         = nan
     T          = nan
     type       = OrbitType.elliptic: 1
     equatorial = True
     circular   = True
     t          = 2023-06-01T00:10:00...

    """
    # initialize output to the same as the input
    new_elements = Elements(len(elements))
    for key, value in vars(elements).items():
        if not is_dunder(key):
            setattr(new_elements, key, value)
    # advance the elements
    nu = advance_true_anomaly(elements.a, elements.e, elements.vo, mu, time)
    # store update variables
    new_elements.vo = nu
    new_elements.t = elements.t + NP_ONE_DAY * (time / JULIAN["day"])
    return new_elements


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_orbits", exit=False)
    doctest.testmod(verbose=False)
