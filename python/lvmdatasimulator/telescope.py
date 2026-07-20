# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: telescope.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import math
import os

import astropy.units as u
from astropy.coordinates import EarthLocation
from dataclasses import dataclass, field

from lvmdatasimulator import DATA_DIR

# Sky and extinction templates bundled for the LVM/LCO site.
DEFAULT_SKY_TELESCOPE = 'LVM160'


def _diameter_from_area(area):
    return 2 * math.sqrt(area.to(u.cm ** 2).value / math.pi) * u.cm


def _lco_location():
    return EarthLocation.of_site('lco')


def _mcdonald_location():
    return EarthLocation.of_site('mcdonald')


@dataclass
class Telescope:
    """
    Telescope configuration used by the simulator and exposure-time calculator.

    Parameters:
        name (str):
            Short identifier for this telescope (used in output file names, etc.).
        aperture_area (astropy.Quantity):
            Collecting area of the primary mirror.
        location (EarthLocation):
            Observatory site used for airmass and visibility calculations.
        utcoffset (astropy.Quantity):
            Local-time offset from UTC at the observatory.
        sky_telescope_name (str):
            Name of the telescope whose sky and extinction templates should be used.
            Defaults to the LVM/LCO templates (``LVM160``).
    """

    name: str
    aperture_area: u.Quantity
    location: EarthLocation
    utcoffset: u.Quantity = -3 * u.hour
    sky_telescope_name: str = DEFAULT_SKY_TELESCOPE

    @property
    def primary_diameter(self):
        """Primary mirror diameter derived from the collecting area."""
        return _diameter_from_area(self.aperture_area)

    @property
    def ifu_angular_scale(self):
        """
        Scale factor for on-sky IFU geometry (arcsec offsets and diameters).

        Layout files describe the physical IFU at the focal plane of the LVM160
        reference telescope. For another primary with the same focal ratio, angular
        sizes scale as the ratio of reference to current primary diameter.
        """
        return float(IFU_REFERENCE_DIAMETER / self.primary_diameter)

    @property
    def extinction_file(self):
        return os.path.join(DATA_DIR, 'sky', f'LVM_{self.sky_telescope_name}_KLAM.dat')


@dataclass
class LVM160(Telescope):
    """LVM 16 cm Schmidt unit at Las Campanas Observatory."""

    name: str = 'LVM160'
    aperture_area: u.Quantity = field(default_factory=lambda: 201 * u.cm ** 2)
    location: EarthLocation = field(default_factory=_lco_location)
    utcoffset: u.Quantity = field(default_factory=lambda: -3 * u.hour)
    sky_telescope_name: str = DEFAULT_SKY_TELESCOPE


# On-sky fiber layout files assume the LVM160 primary plate scale.
IFU_REFERENCE_DIAMETER = LVM160().primary_diameter


@dataclass
class McDonald21(Telescope):
    """2.1-m telescope at McDonald Observatory (same IFU/spectrographs as LVM)."""

    name: str = 'mcdonald21'
    aperture_area: u.Quantity = field(
        default_factory=lambda: math.pi * (105 * u.cm) ** 2
    )
    location: EarthLocation = field(default_factory=_mcdonald_location)
    utcoffset: u.Quantity = field(default_factory=lambda: -6 * u.hour)
    sky_telescope_name: str = DEFAULT_SKY_TELESCOPE


TELESCOPE_REGISTRY = {
    'lvm160': LVM160,
    'mcdonald21': McDonald21,
    'mcdonald': McDonald21,
}


def get_telescope(name='lvm160'):
    """
    Return a telescope instance by name.

    Args:
        name (str):
            Telescope identifier. Supported values: ``lvm160``, ``mcdonald21``,
            ``mcdonald``.

    Returns:
        Telescope:
            Configured telescope instance.
    """
    if not isinstance(name, str):
        raise TypeError(f'telescope name must be a string, not {type(name)}')

    key = name.lower()
    try:
        telescope_cls = TELESCOPE_REGISTRY[key]
    except KeyError as exc:
        available = ', '.join(sorted(TELESCOPE_REGISTRY))
        raise ValueError(
            f"Unknown telescope {name!r}. Available: {available}"
        ) from exc

    return telescope_cls()


def available_telescopes():
    """Return the canonical names of supported telescopes."""
    return sorted({cls().name for cls in TELESCOPE_REGISTRY.values()})
