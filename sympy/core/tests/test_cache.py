from __future__ import annotations
from functools import partial
from inspect import unwrap
from pathlib import Path
from runpy import run_path
import sys
from sympy.core.cache import (USE_CACHE, cacheit, cacheit_by_identity,
    cached_property, clear_cache, lazy_function)
from sympy.testing.pytest import XFAIL, raises

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

            class Meta(type):
                def __hash__(self):
                    raise ValueError('receiver class hashing is unavailable')

            class Equal(metaclass=Meta):
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

            class BadHash:
                def __hash__(self):
                    raise ValueError('receiver hashing is unavailable')

            @decorate
            def identify(receiver):
                return id(receiver)

            receiver = BadHash()
            if mode == 'no' or size == '0':
                assert identify(receiver) == id(receiver)
            else:
                raises(ValueError, lambda: identify(receiver))


def test_cacheit_by_identity_owner_type():
    class Meta(type):
        def __eq__(self, other):
            return isinstance(other, Meta)

        def __hash__(self):
            return 0

    class Receiver(metaclass=Meta):
        @cacheit_by_identity
        def kind(self):
            return type(self).__name__

    class First(Receiver):
        pass

    class Second(Receiver):
        pass

    assert First == Second and First is not Second
    value = First()
    assert value.kind() == 'First'
    value.__class__ = Second
    assert value.kind() == 'Second'
    value.__class__ = First
    assert value.kind() == 'First'


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


def test_cacheit_nonstandard_typeerror(monkeypatch):
    class Message(str):
        __slots__ = ()

        def startswith(self, *args):
            raise AssertionError('exception messages must not supply cache hooks')

    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for name in ('cacheit', 'cacheit_by_identity'):
                for arguments in ((), (None,), (123,), ([],), ('error',),
                                  (Message('error'),), (Message('unhashable type: custom'),),
                                  ('unhashable type: custom', None),
                                  ('unhashable type: custom', 123),
                                  ('unhashable type: custom', [])):
                    error = TypeError(*arguments)
                    calls = []

                    @scope[name]
                    def fail(receiver):
                        calls.append(receiver)
                        raise error

                    receiver = object()
                    caught = raises(TypeError, lambda: fail(receiver))
                    assert caught.value is error
                    assert calls == [receiver]

                    class BadHash:
                        def __hash__(self):
                            raise error

                    @scope[name]
                    def identify(receiver):
                        calls.append(receiver)
                        return id(receiver)

                    receiver = BadHash()
                    calls.clear()
                    if mode == 'no' or size == '0':
                        assert identify(receiver) == id(receiver)
                        assert calls == [receiver]*(2 if mode == 'debug' else 1)
                    else:
                        caught = raises(TypeError, lambda: identify(receiver))
                        assert caught.value is error
                        assert calls == ([receiver] if mode == 'debug' else [])


def test_cacheit_typeerror_subclass(monkeypatch):
    class CacheTypeError(TypeError):
        pass

    class OpaqueTypeError(TypeError):
        @property
        def args(self):
            raise AssertionError('exception arguments must not supply cache hooks')

    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for name in ('cacheit', 'cacheit_by_identity'):
                for error_type in (CacheTypeError, OpaqueTypeError):
                    error = error_type('unhashable type: intentional')
                    calls = []

                    @scope[name]
                    def fail(receiver):
                        calls.append(receiver)
                        raise error

                    receiver = object()
                    caught = raises(TypeError, lambda: fail(receiver))
                    assert caught.value is error
                    assert calls == [receiver]

                    class BadHash:
                        def __hash__(self):
                            raise error

                    @scope[name]
                    def identify(receiver):
                        return id(receiver)

                    receiver = BadHash()
                    if mode == 'no' or size == '0':
                        assert identify(receiver) == id(receiver)
                    else:
                        caught = raises(TypeError, lambda: identify(receiver))
                        assert caught.value is error


def test_cacheit_by_identity_function_typeerror(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for kind in ('function', 'partial', 'callable', 'bound'):
                for arguments in ((object(), 0), ([], 0), (object(), [])):
                    for recover in (False, True):
                        calls = []
                        error = TypeError('unhashable type: intentional')

                        def fail(receiver, value):
                            calls.append(receiver)
                            if recover and len(calls) > 1:
                                return 42
                            raise error

                        class Failure:
                            def __call__(self, receiver, value):
                                return fail(receiver, value)

                            @property
                            def __code__(self):
                                raise AssertionError('user code must not be inspected')

                        function = {'function': fail, 'partial': partial(fail),
                                    'callable': Failure(), 'bound': Failure().__call__}[kind]
                        cached = scope['cacheit_by_identity'](function)
                        assert unwrap(cached) is unwrap(function)
                        caught = raises(TypeError, lambda: cached(*arguments))
                        assert caught.value is error
                        assert calls == [arguments[0]]


def test_cacheit_by_identity_builtin_typeerror(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for recover in (False, True):
                calls = []
                error = TypeError('unhashable type: intentional')

                class Value:
                    def __abs__(self):
                        calls.append(self)
                        if recover and len(calls) > 1:
                            return 42
                        raise error

                cached = scope['cacheit_by_identity'](abs)
                assert unwrap(cached) is abs
                value = Value()
                caught = raises(TypeError, lambda: cached(value))
                assert caught.value is error
                assert calls == [value]


def test_cacheit_hash_function_shared_code(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for name in ('cacheit', 'cacheit_by_identity'):
                def make(result):
                    def implementation(value):
                        if result is None:
                            raise TypeError('unhashable type: intentional')
                        return result
                    return implementation

                class Value:
                    __hash__ = make(None)

                function = make(42)
                assert Value.__hash__.__code__ is function.__code__
                assert scope[name](function)(Value()) == 42


def test_cacheit_receiver_adaptation_typeerror(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            for phase in ('construction', 'adaptation'):
                error = TypeError('unhashable type: intentional')
                calls = []

                class Key:
                    def __init__(self, receiver):
                        if phase == 'construction':
                            raise error

                    @property
                    def value(self):
                        raise error

                def identify(receiver):
                    calls.append(receiver)
                    return 42

                cached = scope['cacheit_by_identity'](identify, key=Key)
                caught = raises(TypeError, lambda: cached(object()))
                assert caught.value is error
                assert not calls


def test_cacheit_by_identity_python_lookup_wrapper(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            make_cache = scope['lru_cache']
            lookups = []

            def framed_cache(maxsize, typed=False):
                def decorate(function):
                    cached = make_cache(maxsize, typed=typed)(function)

                    def lookup(*args, **kwargs):
                        lookups.append(function)
                        return cached(*args, **kwargs)

                    lookup.cache_info = cached.cache_info
                    lookup.cache_clear = cached.cache_clear
                    return lookup
                return decorate

            monkeypatch.setitem(scope['__cacheit'].__globals__, 'lru_cache', framed_cache)
            error = TypeError('unhashable type: intentional')
            calls = []

            def fail(receiver):
                calls.append(receiver)
                raise error

            receiver = object()
            caught = raises(TypeError, lambda: scope['cacheit_by_identity'](fail)(receiver))
            assert caught.value is error
            assert calls == [receiver]

            class Unhashable:
                def __hash__(self):
                    raise error

            def identify(receiver):
                return 42

            assert scope['cacheit_by_identity'](identify)(Unhashable()) == 42
            assert bool(lookups) == (mode != 'no')


def test_cacheit_by_identity_debug_verification_typeerror(monkeypatch):
    monkeypatch.setenv('SYMPY_USE_CACHE', 'debug')
    for size in ('0', '2', 'None'):
        monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
        scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
        for phase in ('body', 'adaptation'):
            for kind in ('function', 'partial', 'callable', 'bound'):
                for arguments in ((object(), 0), ([], 0), (object(), [])):
                    for recover in (False, True):
                        error = TypeError('unhashable type: intentional')
                        constructions, accesses, calls = [], [], []

                        class Key:
                            def __init__(self, receiver, *args):
                                constructions.append(receiver)
                                self.receiver = receiver

                            def __hash__(self):
                                return hash(self.receiver)

                            @property
                            def value(self):
                                accesses.append(self.receiver)
                                if (phase == 'adaptation' and len(accesses) >= 2
                                        and (not recover or len(accesses) == 2)):
                                    raise error
                                return self.receiver

                        def fail(receiver, value):
                            calls.append(receiver)
                            if (phase == 'body' and len(calls) >= 2
                                    and (not recover or len(calls) == 2)):
                                raise error
                            return 42

                        class Failure:
                            def __call__(self, receiver, value):
                                return fail(receiver, value)

                            @property
                            def __code__(self):
                                raise AssertionError('user code must not be inspected')

                        function = {'function': fail, 'partial': partial(fail),
                                    'callable': Failure(), 'bound': Failure().__call__}[kind]
                        cached = scope['cacheit_by_identity'](function, key=Key)
                        assert unwrap(cached) is unwrap(function)
                        caught = raises(TypeError, lambda: cached(*arguments))
                        assert caught.value is error
                        assert constructions == [arguments[0]]
                        assert accesses == [arguments[0]]*2
                        assert calls == [arguments[0]]*(2 if phase == 'body' else 1)


def test_cacheit_by_identity_debug_builtin_typeerror(monkeypatch):
    monkeypatch.setenv('SYMPY_USE_CACHE', 'debug')
    for size in ('0', '2', 'None'):
        monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
        scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
        for hashable in (False, True):
            for recover in (False, True):
                error = TypeError('unhashable type: intentional')
                calls = []

                class Value:
                    __hash__ = object.__hash__ if hashable else None

                    def __abs__(self):
                        calls.append(self)
                        if len(calls) >= 2 and (not recover or len(calls) == 2):
                            raise error
                        return 42

                cached = scope['cacheit_by_identity'](abs)
                assert unwrap(cached) is abs
                receiver = Value()
                caught = raises(TypeError, lambda: cached(receiver))
                assert caught.value is error
                assert calls == [receiver]*2


def test_cacheit_by_identity_metadata_access(monkeypatch):
    for mode in ('yes', 'no', 'debug'):
        for size in ('0', '2', 'None'):
            monkeypatch.setenv('SYMPY_USE_CACHE', mode)
            monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
            scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
            reads = []
            marker = object()

            class Callable:
                @property
                def __name__(self):
                    reads.append('name')
                    return 'identify'

                @property
                def __dict__(self):
                    reads.append('dict')
                    return {'marker': marker}

                def __call__(self, receiver):
                    return 42

            function = Callable()
            cached = scope['cacheit_by_identity'](function)
            assert reads == ['name', 'dict']
            assert cached.__name__ == 'identify'
            assert cached.marker is marker
            assert unwrap(cached) is function
            assert cached(object()) == 42
            assert reads == ['name', 'dict']


def test_cacheit_by_identity_debug_cache_consistency(monkeypatch):
    monkeypatch.setenv('SYMPY_USE_CACHE', 'debug')
    for size in ('0', '2', 'None'):
        monkeypatch.setenv('SYMPY_CACHE_SIZE', size)
        scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
        constructions = []

        class Key:
            def __init__(self, receiver):
                constructions.append(receiver)
                self.value = receiver

            def __hash__(self):
                return id(self.value)

            def __eq__(self, other):
                return type(self) is type(other) and self.value is other.value

        state = [0]
        unexpected = []

        def current(receiver):
            return state[0]

        current.cache_clear = lambda: unexpected.append('cache_clear')
        current.cache_info = lambda: unexpected.append('cache_info')
        cached = scope['cacheit_by_identity'](current, key=Key)
        assert cached.cache_clear is current.cache_clear
        assert cached.cache_info is current.cache_info
        receiver = object()
        assert cached(receiver) == 0
        state[0] = 1
        if size == '0':
            assert cached(receiver) == 1
        else:
            assert raises(RuntimeError,
                lambda: cached(receiver)).value.args == ('Returned values are not the same',)
        assert constructions == [receiver]*2
        scope['clear_cache']()
        assert not unexpected
        assert cached(receiver) == 1
        assert constructions == [receiver]*3


@XFAIL
def test_cacheit_function_unhashable_typeerror(monkeypatch):
    monkeypatch.setenv('SYMPY_USE_CACHE', 'yes')
    monkeypatch.setenv('SYMPY_CACHE_SIZE', '2')
    scope = run_path(str(Path(__file__).parents[1] / 'cache.py'))
    error = TypeError('unhashable type: intentional')
    calls = []

    @scope['cacheit']
    def fail(receiver):
        calls.append(receiver)
        if len(calls) == 1:
            raise error
        return 42

    receiver = object()
    caught = raises(TypeError, lambda: fail(receiver))
    assert caught.value is error
    assert calls == [receiver]


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
