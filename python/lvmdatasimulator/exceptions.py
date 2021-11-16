# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import absolute_import, division, print_function


class LvmdatasimulatorError(Exception):
    """A custom core Lvmdatasimulator exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(LvmdatasimulatorError, self).__init__(message)


class LvmdatasimulatorNotImplemented(LvmdatasimulatorError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(LvmdatasimulatorNotImplemented, self).__init__(message)


class LvmdatasimulatorAPIError(LvmdatasimulatorError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Lvmdatasimulator API'
        else:
            message = 'Http response error from Lvmdatasimulator API. {0}'.format(message)

        super(LvmdatasimulatorAPIError, self).__init__(message)


class LvmdatasimulatorApiAuthError(LvmdatasimulatorAPIError):
    """A custom exception for API authentication errors"""
    pass


class LvmdatasimulatorMissingDependency(LvmdatasimulatorError):
    """A custom exception for missing dependencies."""
    pass


class LvmdatasimulatorWarning(Warning):
    """Base warning for Lvmdatasimulator."""


class LvmdatasimulatorUserWarning(UserWarning, LvmdatasimulatorWarning):
    """The primary warning class."""
    pass


class LvmdatasimulatorSkippedTestWarning(LvmdatasimulatorUserWarning):
    """A warning for when a test is skipped."""
    pass


class LvmdatasimulatorDeprecationWarning(LvmdatasimulatorUserWarning):
    """A warning for deprecated features."""
    pass
