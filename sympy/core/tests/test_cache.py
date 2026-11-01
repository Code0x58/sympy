from __future__ import annotations
from pathlib import Path
from runpy import run_path
import sys
from sympy.core.cache import (USE_CACHE, cacheit, cacheit_by_identity,
    cached_property, clear_cache, lazy_function)
from sympy.testing.pytest import raises

def test_cacheit_doc():
    @cacheit
    def testfn():
        "test docstring"
        pass

    assert testfn.__doc__ == "test docstring"
    assert testfn.__name__ == "testfn"

def test_cacheit_unhashable():
    @cacheit
    def testit(x):
        return x

    assert testit(1) == 1
    assert testit(1) == 1
    a = {}
    assert testit(a) == {}
    a[1] = 2
    assert testit(a) == {1: 2}


def test_cacheit_by_identity():
    class Equal:
        def __eq__(self, other):
            return isinstance(other, Equal)

        def __hash__(self):
            return 0

        @cacheit_by_identity
        def capture(self, amount=0, *, identity=None):
            """Capture the actual receiver and arguments."""
            return self, amount, identity

    first, second = Equal(), Equal()
    left, right = first.capture(), second.capture()
    assert left[0] is first and right[0] is second
    assert first.capture(2, identity=second) == (first, 2, second)
    assert first.capture.__name__ == 'capture'
    assert first.capture.__doc__ == 'Capture the actual receiver and arguments.'
    assert Equal.capture.__wrapped__(first, 3) == (first, 3, None)
    assert Equal.capture(self=first, amount=3) == (first, 3, None)
    if USE_CACHE == 'yes':
        assert first.capture() is left
        assert second.capture() is right
        clear_cache()
        assert first.capture.cache_info().currsize == 0
        assert first.capture() is not left


def test_cacheit_by_identity_keyword_forwarding():
    class Receiver:
        @cacheit_by_identity
        def capture(receiver, *, self=None, identity=None):
            return receiver, self, identity

    value = Receiver()
    result = value.capture(self='self', identity='identity')
    assert result == (value, 'self', 'identity')
    assert Receiver.capture(receiver=value, self='self', identity='identity') == result
    if USE_CACHE == 'yes':
        assert value.capture(self='self', identity='identity') is result


def test_cacheit_by_identity_policies(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            decorate = scope['cacheit_by_identity']

            class Equal:
                def __eq__(self, other):
                    return isinstance(other, Equal)

                def __hash__(self):
                    return 0

                @decorate
                def capture(self, value=0):
                    return id(self), value

            first, second = Equal(), Equal()
            left, right = first.capture(), second.capture()
            assert left == (id(first), 0) and right == (id(second), 0)
            repeated = first.capture()
            if mode == 'yes' and size != '0':
                assert repeated is left
            else:
                assert repeated is not left and repeated == left
            assert Equal.capture(self=first, value=3) == (id(first), 3)
            assert Equal.capture.__wrapped__(first, 4) == (id(first), 4)
            if mode == 'yes':
                expected_size = None if size == 'None' else int(size)
                assert first.capture.cache_info().maxsize == expected_size
                scope['clear_cache']()
                assert first.capture.cache_info().currsize == 0

            calls = []

            @decorate
            def length(receiver, values):
                calls.append(id(receiver))
                return len(values)

            for receiver, values in ((object(), []), ([], ()), ([], [])):
                calls.clear()
                assert length(receiver, values) == length(receiver, values) == 0
                assert len(calls) == (4 if mode == 'debug' else 2)
                assert all(value == id(receiver) for value in calls)


def test_cacheit_by_identity_exceptions():
    calls = []

    @cacheit_by_identity
    def fail(receiver, value):
        calls.append(receiver)
        raise TypeError('identity cache failure')

    for receiver, value in ((object(), 0), ([], 0), (object(), [])):
        calls.clear()
        raises(TypeError, lambda: fail(receiver, value))
        assert len(calls) == 1 and calls[0] is receiver


def test_cachit_exception():
    # Make sure the cache doesn't call functions multiple times when they
    # raise TypeError

    a = []

    @cacheit
    def testf(x):
        a.append(0)
        raise TypeError

    raises(TypeError, lambda: testf(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf([]))
    assert len(a) == 1

    @cacheit
    def testf2(x):
        a.append(0)
        raise TypeError("Error")

    a.clear()
    raises(TypeError, lambda: testf2(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf2([]))
    assert len(a) == 1

def test_cached_property():
    class A:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        @cached_property
        def prop(self):
            self.calls = self.calls + 1
            return self.value

    a = A(2)
    assert a.calls == 0
    assert a.prop == 2
    assert a.calls == 1
    assert a.prop == 2
    assert a.calls == 1
    b = A(None)
    assert b.prop is None


def test_lazy_function():
    module_name='xmlrpc.client'
    function_name = 'gzip_decode'
    lazy = lazy_function(module_name, function_name)
    assert lazy(b'') == b''
    assert module_name in sys.modules
    assert function_name in str(lazy)
    repr_lazy = repr(lazy)
    assert 'LazyFunction' in repr_lazy
    assert function_name in repr_lazy

    lazy = lazy_function('sympy.core.cache', 'cheap')
