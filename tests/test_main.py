# encoding: utf-8
#
# main.py

from pytest import mark

from lvmdatasimulator.main import math


class TestMath(object):
    """Tests for the ``math`` function in main.py."""

    @mark.parametrize(('arg1', 'arg2', 'operator', 'result'),
                      [(1, 2, '+', 3), (2, 2, '-', 0), (3, 5, '*', 15), (10, 2, '/', 5)])
    def test_math(self, arg1, arg2, operator, result):

        assert math(arg1, arg2, arith_operator=operator) == result
