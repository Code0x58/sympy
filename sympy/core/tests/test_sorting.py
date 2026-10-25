from __future__ import annotations
import pytest
from decimal import Decimal
from fractions import Fraction
from functools import partial
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.cache import clear_cache
from sympy.core.function import Function
from sympy.core.numbers import Float
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.testing.pytest import raises

from sympy.abc import x


def test_default_sort_key():
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:
        def __repr__(self):
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]


def test_class_sort_keys():
    for first, second in [
        (sin, cos),
        (Function('a', nargs=1), Function('a', nargs=2)),
        (Function('a', positive=True), Function('a', negative=True)),
        (Function('a', value=Symbol('s', positive=True)),
         Function('a', value=Symbol('s', negative=True))),
        (Function('a', value=Float(1, 10)), Function('a', value=Float(1, 20))),
        (Function('a', _imp_=partial(pow, 2)), Function('a', _imp_=partial(pow, 2))),
        (type('a', (Function,), {}), type('a', (Function,), {})),
    ]:
        first_key, second_key = default_sort_key(first), default_sort_key(second)
        assert first_key != second_key
        assert (first_key < second_key) != (second_key < first_key)
        assert sorted([Tuple(first), Tuple(second)], key=default_sort_key) == \
            sorted([Tuple(second), Tuple(first)], key=default_sort_key)

    first = Function('a', nargs=1, value=Symbol('s', positive=True))
    first_key = default_sort_key(first)
    clear_cache()
    second = Function('a', nargs=1, value=Symbol('s', positive=True))
    assert first == second
    assert first_key == default_sort_key(second)


def test_equal_numeric_metadata_sort_keys():
    first = Function('a', value=2)
    second = Function('a', value=Fraction(2))
    assert first == second
    assert default_sort_key(first) == default_sort_key(second)


@pytest.mark.parametrize('value, equivalent', [
    (1, 1.0), (True, 1), (Fraction(1, 10), Decimal('0.1')), (1, 1 + 0j),
])
def test_equal_metadata_numeric_fields(value, equivalent):
    first = Function('a', value=Basic(value, 'left'))
    clear_cache()
    second = Function('a', value=Basic(equivalent, 'left'))
    other = Function('a', value=Basic(equivalent, 'right'))
    assert first == second
    assert hash(first) == hash(second)
    assert default_sort_key(first) == default_sort_key(second)
    assert first(x) + other(x) == other(x) + first(x)
    assert first(x) + other(x) == second(x) + other(x)


def test_equal_metadata_string_fields():
    for base, constructor, value in ((str, Symbol, 's'), (bytes, Basic, b's')):
        class Subclass(base):
            __slots__ = ()

        first = Function('a', value=constructor(Subclass(value)))
        clear_cache()
        second = Function('a', value=constructor(value))
        assert first == second
        assert hash(first) == hash(second)
        assert default_sort_key(first) == default_sort_key(second)


@pytest.mark.parametrize('constructor', [float, complex, Decimal])
def test_metadata_nan_fields(constructor):
    first = Function('a', value=Basic(constructor('nan')))
    second = Function('a', value=Basic(constructor('nan')))
    assert first != second
    assert default_sort_key(first) != default_sort_key(second)
    assert first(x) + second(x) == second(x) + first(x)


def test_class_sort_keys_with_none_module():
    first = type('F', (Function,), {'__module__': None})
    second = type('F', (Function,), {'__module__': 'example'})
    assert len(sorted([first, second], key=default_sort_key)) == 2
    assert sorted([Tuple(first), Tuple(second)], key=default_sort_key) == \
        sorted([Tuple(second), Tuple(first)], key=default_sort_key)


def test_ordered():
    # Issue 7210 - this had been failing with python2/3 problems
    assert (list(ordered([{1:3, 2:4, 9:10}, {1:3}])) == \
               [{1: 3}, {1: 3, 2: 4, 9: 10}])
    # warnings should not be raised for identical items
    l = [1, 1]
    assert list(ordered(l, warn=True)) == l
    l = [[1], [2], [1]]
    assert list(ordered(l, warn=True)) == [[1], [1], [2]]
    raises(ValueError, lambda: list(ordered(['a', 'ab'], keys=[lambda x: x[0]],
        default=False, warn=True)))
