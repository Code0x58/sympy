""" Caching facility for SymPy """
from __future__ import annotations
from importlib import import_module
from typing import Callable

class _cache(list):
    """ List of cached functions """

    def print_cache(self):
        """print cache info"""

        for item in self:
            name = item.__name__
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_info'):
                    info = myfunc.cache_info()
                    break
                else:
                    myfunc = myfunc.__wrapped__
            else:
                info = None

            print(name, info)

    def clear_cache(self):
        """clear cache content"""
        for item in self:
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_clear'):
                    myfunc.cache_clear()
                    break
                else:
                    myfunc = myfunc.__wrapped__


# global cache registry:
CACHE = _cache()
# make clear and print methods available
print_cache = CACHE.print_cache
clear_cache = CACHE.clear_cache

from functools import lru_cache, wraps


class _IdentityKey:
    """Keep a value alive while using its identity as a key."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return type(self) is type(other) and self.value is other.value

    def __reduce__(self):
        return type(self), (self.value,)


class _ReceiverIdentityKey(_IdentityKey):
    """Preserve hashability and concrete-type checks for cached receivers."""
    __slots__ = ('owner_type',)

    def __init__(self, value, *args):
        self.value = value
        self.owner_type = type(value)

    def __hash__(self):
        # Unhashable receivers bypass caching.
        hash(self.value)
        return id(self.value)

    def __eq__(self, other):
        return (type(self) is type(other) and self.value is other.value
                and self.owner_type is other.owner_type)


def _is_unhashable_error(error, cached_code=None, lookup_code=None):
    """Recognize standard key errors, optionally excluding a cached call."""
    if not (type(error) is TypeError and len(error.args) == 1
            and type(error.args[0]) is str
            and error.args[0].startswith('unhashable type:')):
        return False
    if cached_code is None:
        return True
    trace = error.__traceback__
    if trace is None:
        return False
    origin = trace.tb_next
    # A Python implementation of lru_cache contributes its own frame.
    if (origin is not None and lookup_code is not None
            and lookup_code is not cached_code
            and origin.tb_frame.f_code is lookup_code):
        origin = origin.tb_next
    return origin is None or origin.tb_frame.f_code is not cached_code


def __cacheit(maxsize, *, by_identity=False):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
    def func_wrapper(func, *, _identity_key=_ReceiverIdentityKey):
        if by_identity:
            # This frame distinguishes lookup failures from errors in the call.
            def cached(self_key, /, *args, **kwargs):
                return func(self_key.value, *args, **kwargs)
        else:
            cached = func
        cfunc = lru_cache(maxsize, typed=True)(cached)

        if by_identity:
            cached_code = cached.__code__
            lookup_code = getattr(cfunc.__call__, '__code__', None)
            if lookup_code is None:
                lookup_code = getattr(cfunc, '__code__', None)

            @wraps(func)
            def wrapper(*args, **kwargs):
                if not args:
                    return func(**kwargs)
                self_key = _identity_key(*args)
                try:
                    return cfunc(self_key, *args[1:], **kwargs)
                except TypeError as e:
                    if not _is_unhashable_error(e, cached_code, lookup_code):
                        raise
                    return func(*args, **kwargs)
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    retval = cfunc(*args, **kwargs)
                except TypeError as e:
                    if not _is_unhashable_error(e):
                        raise
                    retval = func(*args, **kwargs)
                return retval

        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear

        CACHE.append(wrapper)
        return wrapper

    if not by_identity:
        func_wrapper._by_identity = __cacheit(maxsize, by_identity=True)
    return func_wrapper
########################################


def __cacheit_nocache(func):
    return func


def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""
    def func_wrapper(func):
        cfunc = __cacheit(maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kw_args):
            # always call function itself and compare it with cached version
            r1 = func(*args, **kw_args)
            r2 = cfunc(*args, **kw_args)

            # try to see if the result is immutable
            #
            # this works because:
            #
            # hash([1,2,3])         -> raise TypeError
            # hash({'a':1, 'b':2})  -> raise TypeError
            # hash((1,[2,3]))       -> raise TypeError
            #
            # hash((1,2,3))         -> just computes the hash
            hash(r1), hash(r2)

            # also see if returned values are the same
            if r1 != r2:
                raise RuntimeError("Returned values are not the same")
            return r1
        return wrapper
    return func_wrapper


def _getenv(key, default=None):
    from os import getenv
    return getenv(key, default)

# SYMPY_USE_CACHE=yes/no/debug
USE_CACHE = _getenv('SYMPY_USE_CACHE', 'yes').lower()
# SYMPY_CACHE_SIZE=some_integer/None
# special cases :
#  SYMPY_CACHE_SIZE=0    -> No caching
#  SYMPY_CACHE_SIZE=None -> Unbounded caching
scs = _getenv('SYMPY_CACHE_SIZE', '1000')
if scs.lower() == 'none':
    SYMPY_CACHE_SIZE = None
else:
    try:
        SYMPY_CACHE_SIZE = int(scs)
    except ValueError:
        raise RuntimeError(
            'SYMPY_CACHE_SIZE must be a valid integer or None. ' + \
            'Got: %s' % SYMPY_CACHE_SIZE)

if USE_CACHE == 'no':
    cacheit = __cacheit_nocache
elif USE_CACHE == 'yes':
    cacheit = __cacheit(SYMPY_CACHE_SIZE)
elif USE_CACHE == 'debug':
    cacheit = __cacheit_debug(SYMPY_CACHE_SIZE)   # a lot slower
else:
    raise RuntimeError(
        'unrecognized value for SYMPY_USE_CACHE: %s' % USE_CACHE)


def cacheit_by_identity(func, *, key=_ReceiverIdentityKey):
    """Cache a method using cacheit's policies and a receiver identity key.

    The key receives all positional arguments and exposes the receiver as ``value``.
    """
    identity_cache = getattr(cacheit, '_by_identity', None)
    if identity_cache is not None:
        return identity_cache(func, _identity_key=key)
    @cacheit
    @wraps(func)
    def cached(self_key, /, *args, **kwargs):
        return func(self_key.value, *args, **kwargs)

    @wraps(cached)
    def wrapper(*args, **kwargs):
        if args:
            return cached(key(*args), *args[1:], **kwargs)
        return func(**kwargs)

    wrapper.__wrapped__ = func
    return wrapper


def cached_property(func):
    '''Decorator to cache property method'''
    attrname = '__' + func.__name__
    _cached_property_sentinel = object()
    def propfunc(self):
        val = getattr(self, attrname, _cached_property_sentinel)
        if val is _cached_property_sentinel:
            val = func(self)
            setattr(self, attrname, val)
        return val
    return property(propfunc)


def lazy_function(module : str, name : str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """
    func = None

    def _get_function():
        nonlocal func
        if func is None:
            func = getattr(import_module(module), name)
        return func

    # The metaclass is needed so that help() shows the docstring
    class LazyFunctionMeta(type):
        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

    class LazyFunction(metaclass=LazyFunctionMeta):
        def __call__(self, *args, **kwargs):
            # inline get of function for performance gh-23832
            nonlocal func
            if func is None:
                func = getattr(import_module(module), name)
            return func(*args, **kwargs)

        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

        def __str__(self):
            return _get_function().__str__()

        def __repr__(self):
            return f"<{__class__.__name__} object at 0x{id(self):x}>: wrapping '{module}.{name}'"

    return LazyFunction()
