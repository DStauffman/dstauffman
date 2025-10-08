r"""
Physical Constants related to orbits.

Notes
-----
#.  Written by David C. Stauffer in July 2021.

"""

# %% Imports
from __future__ import annotations

import doctest
import unittest

from dstauffman import DEG2RAD, HAVE_NUMPY

if HAVE_NUMPY:
    from numpy import pi as PI, sqrt  # noqa: N812
else:
    from math import pi as PI, sqrt  # type: ignore[assignment]  # noqa: N812

# %% Constants
# Written by David C. Stauffer for AA 279 on 28 Apr 2007, constants provided by Professor West.

# Convenient aliases
TAU = 2 * PI

# Gravitational constant
G = 6.67430e-11  # [m**3/(kg*s**2)] (+/- 22ppm) 2018 CODATA

# Solar System Masses
SS_MASSES: dict[str, float] = {
    "sun": 1.9891e30,
    "mercury": 3.3022e23,
    "venus": 4.8690e24,
    "earth": 5.972168494074286e24,  # derived from MU_EARTH / G
    "mars": 6.4191e23,
    "jupiter": 1.8988e27,
    "saturn": 5.6850e26,
    "uranus": 8.6625e25,
    "neptune": 1.0278e26,
    "pluto": 1.314e23,
}  # [kg]

# Sidereal times
SIDEREAL_DAY = 86164.09054  # [s] (23:56:04.09054)
SIDEREAL_YEAR = 365.25636 * SIDEREAL_DAY  # [s] (quasar ref. frame)

# Astronautical Unit
AU = 1.49597870691e11  # [m]

# gravitational constant times large body mass
MU_SUN = G * SS_MASSES["sun"]  # [m**3/s**2]
MU_EARTH = G * SS_MASSES["earth"]  # [m**3/s**2]

# Julian times
JULIAN: dict[str, float] = {}
JULIAN["day"] = 24.0 * 60.0 * 60.0  # [s]
JULIAN["year"] = 365.25 * JULIAN["day"]  # [s]
JULIAN["century"] = 36525.0 * JULIAN["day"]  # [s]
JULIAN["jd_2000_01_01"] = 2451545.0  # Julian Date at 2000-01-01T12:00Z  [day]
JULIAN["tg0_2000"] = 1.753368559  # Julian Date 2451544.5 (2000-01-01T00:00Z) [rad]
JULIAN["tg0_2000_time"] = JULIAN["jd_2000_01_01"] - 0.5  # [rad]
JULIAN["mjd_origin"] = 2400000.5  # Modified Julian Date origin in Julian Days, 1858-11-17T00:00:00Z [day]

# Speed of light (c)
SPEED_OF_LIGHT = 299792458.0  # [m/s]

# Ecliptic inclination
ECLIPTIC = 84381.412 / 3600 * DEG2RAD  # [rad] (+/- 0.005 arcsec)

# Earth model constants
EARTH: dict[str, float] = {}
EARTH["omega"] = TAU / SIDEREAL_DAY  # [rad/s]
EARTH["a"] = 6378137.0  # [m]
EARTH["b"] = 6356752.3  # [m]
EARTH["e"] = sqrt(1.0 - (EARTH["b"] / EARTH["a"]) ** 2)
EARTH["j2"] = -0.00108263  # first aspherical perturbation
EARTH["sf"] = 1353.0  # solar flux [W/m**2]

# Location of Palo Alto, CA, USA
PALO_ALTO: dict[str, float | tuple[float, float, float]] = {}
PALO_ALTO["lat"] = 37.429289 * DEG2RAD  # North [rad]
PALO_ALTO["lng"] = -122.138162 * DEG2RAD  # East [rad]
PALO_ALTO["alt"] = 4.0  # Altitude [m]
PALO_ALTO["geo_loc"] = (PALO_ALTO["lat"], PALO_ALTO["lng"], PALO_ALTO["alt"])  # type: ignore[assignment]

# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_orbit_const", exit=False)
    doctest.testmod(verbose=False)
