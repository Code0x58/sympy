from __future__ import annotations
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
