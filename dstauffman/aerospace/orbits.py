r"""
Classes and functions related to orbits and orbit determination and propagation.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

#%% Imports
from __future__ import annotations
import copy
import datetime
import doctest
from typing import Any, ClassVar, Literal, overload, Tuple, TypeVar, TYPE_CHECKING, Union
import unittest

from dstauffman import DEGREE_SIGN, Frozen, HAVE_NUMPY, IntEnumPlus, NP_DATETIME_FORM, NP_DATETIME_UNITS, RAD2DEG

from dstauffman.aerospace.orbit_const import JULIAN, MU_EARTH, PI, TAU
from dstauffman.aerospace.orbit_conv import anomaly_eccentric_2_true, mean_motion_2_semimajor, anomaly_mean_2_eccentric
from dstauffman.aerospace.orbit_support import cross, d_2_r, dot, jd_to_numpy, norm, r_2_d

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _B = Union[None, bool, np.ndarray]
    _I = Union[None, int, np.ndarray]
    _V = Union[None, float, np.ndarray]
    _N = TypeVar('_N', float, np.ndarray)

#%% Enums - OrbitType
class OrbitType(IntEnumPlus):
    r"""
    Values for the possible orbit types, from {'elliptic', 'parabolic', 'hyperbolic'},
    Plus an 'uninitialized' value.  Note that 'circular' is not an option, but falls under elliptic.

    Examples
    --------
    >>> from dstauffman.aerospace import OrbitType
    >>> print(OrbitType.elliptic)
    OrbitType.elliptic: 1

    """
    uninitialized: ClassVar[int] = 0
    elliptic: ClassVar[int] = 1
    parabolic: ClassVar[int] = 2
    hyperbolic: ClassVar[int] = 3


#%% Classes - Elements
class Elements(Frozen):
    r"""
    Orbital Elements class

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
    t          : epoch time [JD]

    Examples
    --------
    >>> from dstauffman.aerospace import Elements
    >>> elements = Elements()
    >>> elements.pprint()
    Elements
     a          = None
     e          = None
     i          = None
     W          = None
     w          = None
     vo         = None
     p          = None
     uo         = None
     P          = None
     lo         = None
     T          = None
     type       = OrbitType.uninitialized: 0
     equatorial = False
     circular   = False
     t          = NaT

    """

    def __init__(self, num: int = None):
        if num is None:
            self.a: _V = None
            self.e: _V = None
            self.i: _V = None
            self.W: _V = None
            self.w: _V = None
            self.vo: _V = None
            self.p: _V = None
            self.uo: _V = None
            self.P: _V = None
            self.lo: _V = None
            self.T: _V = None
            self.type: _I = OrbitType.uninitialized
            self.equatorial: _B = False
            self.circular: _B = False
            self.t: np.datetime64 = np.datetime64('nat', NP_DATETIME_UNITS)
        else:
            self.a = np.full(num, np.nan)
            self.e = np.full(num, np.nan)
            self.i = np.full(num, np.nan)
            self.W = np.full(num, np.nan)
            self.w = np.full(num, np.nan)
            self.vo = np.full(num, np.nan)
            self.p = np.full(num, np.nan)
            self.uo = np.full(num, np.nan)
            self.P = np.full(num, np.nan)
            self.lo = np.full(num, np.nan)
            self.T = np.full(num, np.nan)
            self.type = np.full(num, OrbitType.uninitialized, dtype=int)
            self.equatorial = np.zeros(num, dtype=bool)
            self.circular = np.zeros(num, dtype=bool)
            self.t = np.full(num, np.datetime64('nat'), dtype=NP_DATETIME_FORM)  # type: ignore[assignment]

    def __eq__(self, other: Any) -> bool:
        # if not of the same type, then they are not equal
        if type(other) != type(self):
            return False
        # loop through the fields, and if any are not equal, then it's not equal
        for key in vars(self):
            x = getattr(self, key)
            y = getattr(other, key)
            if key in {'type', 'equatorial', 'circular'}:
                if not np.all(x == y):
                    return False
            elif key in {'t'}:
                if not np.all((np.isnat(x) & np.isnat(y)) | (x == y)):
                    return False
            else:
                if x is None and y is None:
                    pass
                elif x is None or y is None:
                    return False
                elif not np.allclose(x, y, rtol=1e-6, atol=1e-8, equal_nan=True):
                    return False
        # if it made it all the way through the fields, then things must be equal
        return True

    def __getitem__(self, key):
        if self.a is None or isinstance(self.a, float):
            raise IndexError('Elements is only a single instance and can not be indexed')
        elements = Elements()
        for (field, value) in vars(self).items():
            setattr(elements, field, value[key])
        return elements

    def combine(self, elements2: Elements, /, *, inplace: bool = False) -> Elements:
        r"""Combines two KfInnov structures together."""
        # allow an empty structure to be passed through
        if self.a is None:
            if inplace:
                for (key, value) in vars(elements2).items():
                    setattr(self, key, value)
            return elements2  # TODO: make a copy?
        # concatenate fields
        if inplace:
            elements = self
        else:
            elements = copy.deepcopy(self)
        if elements2.a is None:
            return elements
        for (key, value) in vars(self).items():
            setattr(elements, key, np.hstack((value, getattr(elements2, key))))
        return elements

    def print_orrery(self, index: int = None):
        if index is None:
            print(f'a = {self.a / 1000} km')  # type: ignore[operator]
            print(f'e = {self.e}')
            print(f'i = {self.i * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[operator]
            print(f'\N{GREEK CAPITAL LETTER OMEGA} = {self.W * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[operator]
            print(f'\N{GREEK SMALL LETTER OMEGA} = {self.w * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[operator]
            print(f'M = {self.vo * RAD2DEG} {DEGREE_SIGN} (TODO: actually nu, need M)')  # type: ignore[operator]
        else:
            print(f'a = {self.a[index] / 1000} km')  # type: ignore[index]
            print(f'e = {self.e[index]}')  # type: ignore[index]
            print(f'i = {self.i[index] * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[index]
            print(f'\N{GREEK CAPITAL LETTER OMEGA} = {self.W[index] * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[index]
            print(f'\N{GREEK SMALL LETTER OMEGA} = {self.w[index] * RAD2DEG} {DEGREE_SIGN}')  # type: ignore[index]
            print(f'M = {self.vo[index] * RAD2DEG} {DEGREE_SIGN} (TODO: actually nu, need M)')  # type: ignore[index]


#%% Functions - _zero_divide
def _zero_divide(x, y):
    return np.divide(x, y, where=y != 0, out=np.zeros_like(y))


#%% Functions - _inf_divide
def _inf_divide(x, y):
    return np.divide(x, y, where=y != 0, out=np.full(y.shape, np.inf))


def _fix_instab(x, precision):
    ix = (x > 1.0) & (x < 1.0 + precision)
    if np.any(ix):
        x[ix] = 1.0
    ix = (x < -1.0) & (x > -1 - precision)
    if np.any(ix):
        x[ix] = -1.0


#%% Functions - two_line_elements
def two_line_elements(line1: str, line2: str) -> Elements:

    r"""
    Convert two-line elements to elements structure

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
    >>> line1 = '1 25544U 98067A   06132.29375000  .00013633  00000-0  92740-4 0  9181'
    >>> line2 = '2 25544  51.6383  12.2586 0009556 188.7367 320.5459 15.75215761427503'
    >>> elements = two_line_elements(line1,line2)
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
     t          = 2006-05-11T19:03:00.000016096

    """
    if not line1.startswith('1'):
        raise ValueError(f'line1 must start with "1": {line1}')
    if not line2.startswith('2'):
        raise ValueError(f'line2 must start with "2": {line2}')
    if len(line1) != 69:
        ValueError(f'line1 must have length 69: {line1}')
    if len(line2) != 69:
        ValueError(f'line2 must have length 69: {line2}')

    try:
        year = int(line1[18:20])
    except:
        raise ValueError(f'Error reading year from line1: {line1}')

    try:
        day = float(line1[20:32])
    except:
        raise ValueError(f'Error reading day from line1: {line1}')

    try:
        i = float(line2[8:16])
    except:
        raise ValueError('fError reading inclination (i) from line2: {line2}')
    i = d_2_r(i)

    try:
        Omega = float(line2[17:25])
    except:
        raise ValueError(f'Error reading longitude of ascending node (Omega) from line2: {line2}')
    Omega = d_2_r(Omega)

    try:
        e = float('0.' + line2[26:33])
    except:
        raise ValueError(f'Error reading eccentricity (e) from line2: {line2}')

    try:
        omega = float(line2[34:42])
    except:
        raise ValueError(f'Error reading argument of perigee (omega) from line2: {line2}')
    omega = d_2_r(omega)

    try:
        M = float(line2[43:51])
    except:
        raise ValueError(f'Error reading mean anomaly (M) from line2: {line2}')
    M = d_2_r(M)

    try:
        revs_per_day = float(line2[52:63])
    except:
        raise ValueError(f'Error reading revolutions per day from line2: {line2}')

    n = revs_per_day * TAU / JULIAN['day']  # [rad/sec]
    a = mean_motion_2_semimajor(n, MU_EARTH)
    E = anomaly_mean_2_eccentric(M, e)
    nu = anomaly_eccentric_2_true(E, e)
    p = a * (1 - e ** 2)

    time = (
        (datetime.datetime(2000 + year, 1, 1, 0, 0, 0) - datetime.datetime(2000, 1, 1, 0, 0, 0)).days
        + JULIAN['jd_2000_01_01'] - 0.5 + day - 1
    )

    elements            = Elements()
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

    return elements


#%% Functions - rv_2_oe
def rv_2_oe(
    r: np.ndarray, v: np.ndarray, mu: Union[float, np.ndarray] = 1.0, unit: bool = False, precision: float = 1e-12
) -> Elements:
    r"""
    Position and Velocity to Orbital Elements.

    Parameters
    ----------
    r : (3, N) ndarray
        Position vector
    v : (3, N) ndarray
        Velocity vector
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
     type       = OrbitType.elliptic: 1
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
    ix = n_mag == 0
    if np.any(ix):
        n[..., ix] = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, np.count_nonzero(ix)))
        n_mag[ix] = 1.0

    e = 1.0 / mu * ((norm_v ** 2 - _zero_divide(mu, norm_r)) * r - dot(r, v) * v)

    # e
    e_mag = norm(e)

    # get sizing information
    is_scalar = h.ndim == 1
    num = e_mag.size

    # p
    p = norm_h ** 2 / mu

    # a
    # Note: a is infinite when eccentricity is exactly one
    a = np.divide(p, 1.0 - e_mag ** 2, where=e_mag != 1, out=np.full(num, np.inf))

    # i
    i = np.arccos(_zero_divide(h[2, ...], norm_h), where=norm_h != 0, out=np.zeros_like(norm_h))

    # W
    W = np.asanyarray(np.arccos(n[0, ...] / n_mag))
    W = np.subtract(TAU, W, out=W, where=n[1, ...] < -precision)

    # w
    ix = np.abs(e_mag) >= precision
    w = np.divide(dot(n, e), n_mag * e_mag, where=ix, out=np.zeros(num))
    _fix_instab(w, precision=precision)
    w = np.arccos(w, where=ix, out=w)
    w = np.subtract(TAU, w, out=w, where=ix & (e[2, ...] < -precision))
    w[ix & ~np.isreal(w)] = PI

    # vo
    ix &= norm_r > precision
    vo = np.divide(dot(e, r), e_mag * norm_r, where=ix, out=np.zeros(num))
    _fix_instab(vo, precision=precision)
    vo = np.arccos(vo, where=ix, out=vo)
    vo = np.subtract(TAU, vo, out=vo, where=ix & (dot(r, v) < -precision))

    # uo
    uo = np.asanyarray(np.mod(w + vo, TAU))
    uo[np.abs(uo - TAU) < precision] = 0.0

    # P
    P = np.asanyarray(np.mod(W + w, TAU))
    P[np.abs(P - TAU) < precision] = 0.0

    # lo
    lo = np.asanyarray(np.mod(W + w + vo, TAU))
    lo[np.abs(lo - TAU) < precision] = 0.0

    # tell if equatorial
    equatorial = (np.abs(i) < precision) | (np.abs(i - PI) < precision)

    # convert to degrees specified
    if unit:
        i  = r_2_d(i)
        W  = r_2_d(W)
        w  = r_2_d(w)
        vo = r_2_d(vo)
        uo = r_2_d(uo)
        P  = r_2_d(P)
        lo = r_2_d(lo)

    # allocate stuff
    orbit_type = np.full(num, OrbitType.uninitialized, dtype=int)
    T = np.full(num, np.nan)
    # dt = np.full(num, np.nan)
    circular = np.zeros(num, dtype=bool)

    # specific energy
    E = norm_v ** 2 / 2 - _inf_divide(mu, norm_r)
    # test for type of orbit - elliptic
    ix = E < 0
    orbit_type[ix] = OrbitType.elliptic
    # T (only defined for elliptic, as parabolic and hyperbolic don't have a period)
    full_mu = mu if num == 1 or np.size(mu) == 1 else mu[ix]  # type: ignore[index]
    T[ix] = TAU * np.sqrt(a[ix] ** 3 / full_mu)
    # dt[ix] = T[ix] / TAU * M
    circular[ix & (np.abs(e_mag) < precision)] = True
    # parabolic
    ix = E == 0
    orbit_type[ix] = OrbitType.parabolic
    # hyperbolic
    ix = E > 0
    orbit_type[ix] = OrbitType.hyperbolic

    # populate elements structure
    if is_scalar:
        elements = Elements()
        elements.a          = float(a)
        elements.e          = float(e_mag)
        elements.i          = float(i)
        elements.W          = float(W)
        elements.w          = float(w)
        elements.vo         = float(vo)
        elements.p          = p
        elements.uo         = float(uo)
        elements.P          = float(P)
        elements.lo         = float(lo)
        elements.equatorial = bool(equatorial)
        elements.type       = OrbitType(orbit_type)
        elements.T          = float(T)
        elements.circular   = bool(circular)
    else:
        elements               = Elements(num)
        elements.a[:]          = a           # type: ignore[index]
        elements.e[:]          = e_mag       # type: ignore[index]
        elements.i[:]          = i           # type: ignore[index]
        elements.W[:]          = W           # type: ignore[index]
        elements.w[:]          = w           # type: ignore[index]
        elements.vo[:]         = vo          # type: ignore[index]
        elements.p[:]          = p           # type: ignore[index]
        elements.uo[:]         = uo          # type: ignore[index]
        elements.P[:]          = P           # type: ignore[index]
        elements.lo[:]         = lo          # type: ignore[index]
        elements.equatorial[:] = equatorial  # type: ignore[index]
        elements.type[:]       = orbit_type  # type: ignore[index]
        elements.T[:]          = T           # type: ignore[index]
        elements.circular[:]   = circular    # type: ignore[index]

    return elements


#%% oe_2_rv
@overload
def oe_2_rv(
    elements: Elements,
    mu: Union[float, np.ndarray] = ...,
    unit: bool = ...,
    *,
    return_PQW: Literal[False] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...

@overload
def oe_2_rv(
    elements: Elements, mu: Union[float, np.ndarray] = ..., unit: bool = ..., *, return_PQW: Literal[True]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def oe_2_rv(
    elements: Elements, mu: Union[float, np.ndarray] = 1.0, unit: bool = False, *, return_PQW: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Orbital Elements to Position and Velocity.

    This function takes a given elements structure containing subfields for
    a,e,i,W,w,vo scalars and calculates the position and velocity vectors r & v.

    Parameters
    ----------
    elements : class Elements
        Orbital elements
    mu : float, optional
        Gravitational constant times the sum of the masses
    unit : bool, optional, default is False
        Units to return, where True changes to degrees
    return_PQW : bool, optional, default is False
        Whether to return position and velocity in PQW frame, along with the transformation

    Returns
    -------
    r     : 3x1 position vector
    v     : 3x1 velocity vector
    (The next three values are usually treated as optional outputs)
    r_PQW : 3x1 position vector in PQW frame
    v_PQW : 3x1 velocity vector in PQW frame
    T     : transformation matrix

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 24 April 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import Elements, oe_2_rv
    >>> elements = Elements()
    >>> elements.a  = 1
    >>> elements.e  = 0
    >>> elements.i  = 0
    >>> elements.W  = 0
    >>> elements.w  = 0
    >>> elements.vo = 0
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
    assert a is not None
    assert e is not None
    assert i is not None
    assert W is not None
    assert w is not None
    assert nu is not None

    # adjust if angles are in degrees
    if unit:
        i  = d_2_r(i)
        W  = d_2_r(W)
        w  = d_2_r(w)
        nu = d_2_r(nu)

    #% calculations
    # parameter
    if elements.p is not None:
        p = elements.p
    else:
        p = a * (1.0 - e ** 2)
    assert p is not None

    # magnitude of r in PQW frame
    r_mag = p / (1 + e * np.cos(nu))

    # r in PQW frame
    r_PQW = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])

    # v in PQW frame
    v_PQW = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # cosine and sine terms
    cw = np.cos(w)
    sw = np.sin(w)
    cW = np.cos(W)
    sW = np.sin(W)
    ci = np.cos(i)
    si = np.sin(i)

    # transformation matrix
    # TODO: create quaternion and use qrot instead
    T = np.array([
        [+cW * cw - sW * sw * ci, -cW * sw - sW * cw * ci, +sW * si],
        [+sW * cw + cW * sw * ci, -sW * sw + cW * cw * ci, -cW * si],
        [+sw * si,                +cw * si,                     +ci],
    ])

    # translate r & v into IJK frame
    r = T @ r_PQW
    v = T @ v_PQW

    if return_PQW:
        return (r, v, r_PQW, v_PQW, T)
    return (r, v)


#%% Functions - advance_elements
def advance_elements(elements, mu, time):
    pass


#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_orbits', exit=False)
    doctest.testmod(verbose=False)
