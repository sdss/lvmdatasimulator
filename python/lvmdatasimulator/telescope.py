# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u

from dataclasses import dataclass

from abc import ABC, abstractmethod


@dataclass
class Telescope(ABC):

    @property
    @abstractmethod
    def aperture_area(self):
        pass


@dataclass
class LVM160(Telescope):
    """
    Class summarizing the properties of the telescope. Pretty empty, but for the ETC we need just
    this
    """
    aperture_area: u.cm**2 = 201 * u.cm**2
