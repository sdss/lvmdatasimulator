# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu



class LVMField(object):
    """Main container for objects in field of view of LVM.

    This is the main class of the simulator of a sources that contains all the functions and reproduces
    the data as it is on the "fake" sky.

    Parameters:
        name (str):
            Name of the current field
        RA (str or None):
            Right Ascension of the center of the field
        Dec (str or None):
            Declination of the center of the field

    Attributes:
        name (str): Name of the current field.
        RA (str or None):
            Right Ascension of the center of the field
        Dec (str or None):
            Declination of the center of the field

    """

    def __init__(self, name='LVM_field', RA=None, Dec=None):
        self.name = name

    def add_stars(self, RA=None, Dec=None):
        """Add stars from GAIA catalogue for current RA and Dec.
        """
        pass





class Cloud(object):
    """Add a cloud to the current field.

    """
    pass


class Bubble(Cloud):
    """Add a bubble to the current field.

    """
    pass