from __future__ import annotations
import pytest
import gc
import os
from decimal import Decimal
from fractions import Fraction
from functools import partial
from itertools import permutations
from pickle import dumps, loads
from threading import Event, Thread
from weakref import ref
import sympy.core.function as function_module
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.cache import clear_cache
from sympy.core.function import Function, Subs
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.testing.pytest import raises

from sympy.abc import x

if hasattr(os, 'fork'):
    from multiprocessing import get_context


def test_default_sort_key():
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:
        def __repr__(self):
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]


@pytest.mark.parametrize('order', [None, 'rev-lex'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('convert', [int, S])
def test_frozenset_sort_key(order, nested, convert):
    first, second, other = [frozenset(map(convert, items))
                            for items in ((1, 9), (9, 1), (8, 1))]
    if nested:
        first, second, other = [frozenset([value]) for value in (first, second, other)]
    key = lambda value: default_sort_key(value, order=order)
    assert first == second
    assert key(first) == key(second)
    assert key(first) != key(other)
    assert sorted([first, other], key=key) == sorted([other, second], key=key)


@pytest.mark.parametrize('lower, upper', [
    (1, 1 + 2j), (1j, 1), (1 + 2j, 2),
    (Decimal('0.5'), 0.5 + 1j), (Fraction(1, 2), 0.5 + 1j),
    (frozenset([1, 2]), frozenset([2, 3])),
    (frozenset([1]), frozenset([2])),
    (frozenset([S.One]), frozenset([1])),
    (frozenset([frozenset([1])]), frozenset([frozenset([2])])),
    ((1j,), (2j,)),
    ((frozenset([1]),), (frozenset([2]),)),
])
def test_native_argument_sort_keys(lower, upper):
    first, second = Basic(lower), Basic(upper)
    assert first.compare(second) == -1
    for values in ([first, second], [second, first]):
        clear_cache()
        assert sorted(values, key=default_sort_key) == [first, second]
    left, right = default_sort_key(first), default_sort_key(second)
    assert left < right and left <= right and right > left and right >= left
    assert not (left > right or left >= right or right < left or right <= left)


@pytest.mark.parametrize('value, equivalent', [
    (1 + 0j, 1), (0j, False), (0.5 + 0j, Fraction(1, 2)),
    (frozenset([1]), frozenset([True])),
    ((1,), (True,)), ((1,), (1.0,)),
])
def test_native_argument_sort_key_reconstruction(value, equivalent):
    first = default_sort_key(Basic(value))
    clear_cache()
    second = default_sort_key(Basic(equivalent))
    restored = loads(dumps(first))
    assert first == second == restored
    assert hash(first) == hash(second) == hash(restored)
    if isinstance(value, complex):
        other = default_sort_key(Basic(2 + 1j))
    elif isinstance(value, tuple):
        other = default_sort_key(Basic((2,)))
    else:
        other = default_sort_key(Basic(frozenset([2])))
    assert first < other and second < other and restored < other


@pytest.mark.parametrize('value, other', [
    (complex('nan'), 1), (complex(0, float('nan')), 1),
    (1j, float('nan')), (1j, Decimal('NaN')),
])
def test_native_argument_nan_sort_keys(value, other):
    first = default_sort_key(Basic(value))
    clear_cache()
    assert first == default_sort_key(Basic(value))
    second = default_sort_key(Basic(other))
    assert first != second
    with raises(TypeError):
        assert first < second
    with raises(TypeError):
        assert second < first


def test_native_argument_sort_key_storage():
    class StoredSet(frozenset):
        def __iter__(self):
            return iter([2])

    first = default_sort_key(Basic(StoredSet([1, 9])))
    clear_cache()
    second = default_sort_key(Basic(frozenset([1, 9])))
    other = default_sort_key(Basic(frozenset([1, 8])))
    assert first == second
    assert hash(first) == hash(second)
    assert first > other and second > other


@pytest.mark.parametrize('hide_protocol', [False, True])
@pytest.mark.parametrize('lower, upper', [
    (frozenset([None]), frozenset([None, 1])),
    (frozenset([frozenset([None])]),
     frozenset([frozenset([None]), frozenset([1])])),
    (frozenset([x]), frozenset([None, 0])),
    (frozenset(), frozenset([None, 1])),
    (frozenset([1]), frozenset([1, 2])),
])
def test_frozenset_argument_cardinality(lower, upper, hide_protocol):
    class StoredSet(frozenset):
        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    container = StoredSet if hide_protocol else frozenset
    first, second = Basic(container(lower)), Basic(container(upper))
    assert first != second
    assert first.compare(second) == -1
    assert second.compare(first) == 1
    left, right = default_sort_key(first), default_sort_key(second)
    assert left < right and not right < left
    for values in ([first, second], [second, first]):
        assert sorted(values, key=default_sort_key) == [first, second]


@pytest.mark.parametrize('base, value', [(complex, 1j), (frozenset, [1]), (tuple, [1])])
def test_native_argument_sort_key_hooks(base, value):
    class Tagged(base):
        def __new__(cls, tag):
            obj = super().__new__(cls, value)
            obj.tag = tag
            return obj

        def __eq__(self, other):
            return isinstance(other, Tagged) and self.tag == other.tag

        def __hash__(self):
            return hash(self.tag)

        def __lt__(self, other):
            return self.tag < other.tag

        def __gt__(self, other):
            return self.tag > other.tag

    first, second = Basic(Tagged(1)), Basic(Tagged(2))
    assert first.compare(second) == -1
    assert second.compare(first) == 1
    assert default_sort_key(first) < default_sort_key(second)


@pytest.mark.parametrize('container', [tuple, lambda value: (value,),
    lambda value: frozenset([value]), lambda value: (frozenset(value),)])
@pytest.mark.parametrize('wrap', [lambda value: value,
    lambda value: Function('tuple_field', value=value)])
def test_tuple_field_equivalence(container, wrap):
    fields = [Basic(container((value,))) for value in (1, 1.0, S.One)]
    groups = (0, 0, 1)
    for indices in permutations(range(3)):
        items = []
        keys = []
        for i in indices:
            clear_cache()
            items.append(wrap(fields[i]))
            clear_cache()
            keys.append(default_sort_key(items[-1]))
        expressions = [item if isinstance(item, Basic) else item(x) for item in items]
        for i, left in enumerate(items):
            for j, right in enumerate(items):
                equal = groups[indices[i]] == groups[indices[j]]
                assert (left == right) == equal
                assert (keys[i] == keys[j]) == equal
                comparison = expressions[i].compare(expressions[j])
                assert (comparison == 0) == equal
                assert comparison == -expressions[j].compare(expressions[i])
                assert comparison == (keys[i] > keys[j]) - (keys[i] < keys[j])
                if equal:
                    assert hash(left) == hash(right)
                    assert hash(keys[i]) == hash(keys[j])


def test_tuple_field_partial_order():
    class TupleValue:
        def __eq__(self, other):
            return other == (1,) if type(other) is tuple else NotImplemented

        def __hash__(self):
            return hash((1,))

        def __lt__(self, other):
            return False

        __gt__ = __lt__

    first, second = Basic((1,)), Basic(TupleValue())
    assert first != second
    expressions = (first, second)
    keys = [default_sort_key(value) for value in (first, second)]
    assert keys[0] != keys[1]
    for a, b in ((0, 1), (1, 0)):
        with raises(TypeError, match='Cannot order tuple fields with custom equality'):
            expressions[a].compare(expressions[b])
        with raises(TypeError, match='Cannot order tuple fields with custom equality'):
            assert keys[a] < keys[b]


def test_tuple_field_custom_ordering():
    class ReverseTuple(tuple):
        __slots__ = ()
        __lt__ = tuple.__gt__
        __gt__ = tuple.__lt__

    values = [Basic(ReverseTuple((value,))) for value in (1, 2, 3)]
    assert sorted(values, key=default_sort_key) == list(reversed(values))
    first = default_sort_key(values[0])
    second = default_sort_key(values[1])
    assert first > second and second < first
    assert first >= second and second <= first
    assert first == default_sort_key(Basic(ReverseTuple((1.0,))))
    symbolic = default_sort_key(Basic(ReverseTuple((S.One,))))
    assert first != symbolic
    assert (first > symbolic) != (first < symbolic)

    class FailingTuple(tuple):
        __slots__ = ()

        def __lt__(self, other):
            raise ValueError('tuple ordering failed')

        __gt__ = __lt__

    first, second = Basic(FailingTuple((1,))), Basic(FailingTuple((2,)))
    assert first.compare(Basic(FailingTuple((1,)))) == 0
    with raises(ValueError, match='tuple ordering failed'):
        first.compare(second)
    with raises(ValueError, match='tuple ordering failed'):
        assert default_sort_key(first) < default_sort_key(second)


def test_native_argument_boolean_comparison():
    class Value:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return type(other) is Value and self.value == other.value

        def __hash__(self):
            return hash(self.value)

        def __lt__(self, other):
            return S(self.value < other.value)

        def __gt__(self, other):
            return S(self.value > other.value)

    first, second = Basic(Value(1)), Basic(Value(2))
    assert first.compare(second) == -1
    assert second.compare(first) == 1
    assert default_sort_key(first) < default_sort_key(second)
    assert default_sort_key(second) > default_sort_key(first)


@pytest.mark.parametrize('native, symbolic', [(1, S.One), (1.0, Float(1))])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('protocol', ['subclass', 'proxy'])
@pytest.mark.parametrize('wrap', [lambda value: value,
    lambda value: Function('protocol_field', value=value)])
def test_frozenset_field_protocol_identity(native, symbolic, nested, protocol, wrap):
    class ForwardSet(frozenset):
        def __eq__(self, other):
            if type(other) not in (ForwardSet, frozenset):
                return NotImplemented
            return super().__eq__(other)

        __hash__ = frozenset.__hash__

    class SetValue:
        def __init__(self, values):
            self.values = frozenset(values)

        def __eq__(self, other):
            if isinstance(other, SetValue):
                other = other.values
            if type(other) is frozenset:
                return self.values == other
            return NotImplemented

        def __hash__(self):
            return hash(self.values)

    constructor = ForwardSet if protocol == 'subclass' else SetValue
    values = [frozenset([native]), constructor([native]),
              frozenset([symbolic]), constructor([symbolic])]
    if nested:
        values = [frozenset([value]) for value in values]
    fields = list(map(Basic, values))
    groups = (0, 1, 2, 1)
    for indices in permutations(range(4)):
        clear_cache()
        items = [wrap(fields[i]) for i in indices]
        expressions = [item if isinstance(item, Basic) else item(x) for item in items]
        keys = []
        for item in items:
            clear_cache()
            keys.append(default_sort_key(item))
        for i, left in enumerate(items):
            for j, right in enumerate(items):
                equal = groups[indices[i]] == groups[indices[j]]
                assert (left == right) == equal
                assert (keys[i] == keys[j]) == equal
                comparison = expressions[i].compare(expressions[j])
                assert (comparison == 0) == equal
                assert comparison == -expressions[j].compare(expressions[i])
                assert comparison == (keys[i] > keys[j]) - (keys[i] < keys[j])
                if equal:
                    assert hash(left) == hash(right)
                    assert hash(keys[i]) == hash(keys[j])


def test_frozenset_field_ordering_errors():
    class UnorderedSet(frozenset):
        def __eq__(self, other):
            return super().__eq__(other)

        __hash__ = frozenset.__hash__

        def __gt__(self, other):
            raise RuntimeError('custom ordering failed')

    first, second = [Basic(UnorderedSet([value])) for value in (1, 2)]
    with raises(RuntimeError, match='custom ordering failed'):
        first.compare(second)
    clear_cache()
    with raises(RuntimeError, match='custom ordering failed'):
        assert default_sort_key(first) < default_sort_key(second)


def test_frozenset_field_partial_order():
    class ForwardSet(frozenset):
        def __eq__(self, other):
            return super().__eq__(other)

        __hash__ = frozenset.__hash__

    first, second = [Basic(ForwardSet([value])) for value in (1, 2)]
    with raises(TypeError, match='Cannot order frozenset fields with custom equality'):
        first.compare(second)
    clear_cache()
    with raises(TypeError, match='Cannot order frozenset fields with custom equality'):
        assert default_sort_key(first) < default_sort_key(second)


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


@pytest.mark.parametrize('order', [None, 'rev-lex'])
@pytest.mark.parametrize('wrap', [lambda value: value, Basic,
    lambda value: Tuple(Basic(value))])
def test_equal_class_literal_sort_keys(order, wrap):
    class Meta(type):
        def __eq__(cls, other):
            if not isinstance(other, Meta):
                return NotImplemented
            return cls.label == other.label

        def __hash__(cls):
            return 0

    class SubMeta(Meta):
        pass

    first = Meta('SharedName', (), {'label': 1})
    second = SubMeta('Alias', (), {'label': 1})
    other = Meta('SharedName', (), {'label': 2})
    keys = []
    for value in (first, second, other):
        clear_cache()
        keys.append(default_sort_key(wrap(value), order=order))
    assert first == second and second == first
    assert keys[0] == keys[1]
    assert hash(keys[0]) == hash(keys[1])
    assert keys[0] != keys[2]
    assert (keys[0] < keys[2]) == (keys[1] < keys[2])
    assert (keys[0] > keys[2]) == (keys[1] > keys[2])
    assert Basic(first).compare(Basic(second)) == 0
    assert Basic(first).compare(Basic(other)) == Basic(second).compare(Basic(other)) != 0
    key_hash = hash(keys[0])
    del keys
    clear_cache()
    alias_key = default_sort_key(wrap(second), order=order)
    assert hash(default_sort_key(wrap(first), order=order)) == key_hash == hash(alias_key)


def test_class_literal_sort_key_identity():
    class UnhashableMeta(type):
        __hash__ = None

    for metaclass in (type, UnhashableMeta):
        first = metaclass('SharedName', (), {})
        second = metaclass('SharedName', (), {})
        clear_cache()
        forward = sorted([first, second], key=default_sort_key)
        clear_cache()
        backward = sorted([second, first], key=default_sort_key)
        assert forward == backward


def _class_alias_metaclass():
    class Meta(type):
        def __eq__(cls, other):
            return isinstance(other, Meta) and cls.label == other.label

        def __hash__(cls):
            return 0

    return Meta


def test_equal_class_literal_cold_comparison():
    Meta = _class_alias_metaclass()
    first = Meta('Alpha', (), {'label': 1})
    second = Meta('Zulu', (), {'label': 1})
    other = Meta('Middle', (), {'label': 2})
    expressions = list(map(Basic, (first, second, other)))
    for indices in permutations(range(3)):
        clear_cache()
        left, alias, third = [expressions[i] for i in indices]
        if left != alias:
            continue
        comparisons = []
        for a, b in ((left, third), (alias, third), (third, left), (third, alias)):
            clear_cache()
            comparisons.append(a.compare(b))
        assert comparisons[0] == comparisons[1] != 0
        assert comparisons[2] == comparisons[3] == -comparisons[0]


def test_class_order_survives_representative_collection():
    Meta = _class_alias_metaclass()
    first = Meta('Alpha', (), {'label': 1})
    alias = Meta('Zulu', (), {'label': 1})
    other = Basic(Meta('Middle', (), {'label': 2}))
    first_ref = ref(first)
    expected = Basic(first).compare(other)
    assert Basic(alias).compare(other) == expected != 0
    del first
    clear_cache()
    gc.collect()
    assert first_ref() is None
    assert Basic(alias).compare(other) == expected


@pytest.mark.parametrize('wrap', [lambda value: value, Basic])
def test_class_order_survives_interned_representative(wrap):
    Meta = _class_alias_metaclass()
    first = Meta('Alpha', (), {'label': 1})
    alias = Meta('Zulu', (), {'label': 1})
    other = Basic(Meta('Middle', (), {'label': 2}))
    reference = ref(first)
    first_key = default_sort_key(wrap(first))
    alias_key = default_sort_key(wrap(alias))
    assert first_key == alias_key
    expected = Basic(alias).compare(other)
    del first, first_key, alias_key
    clear_cache()
    gc.collect()
    assert reference() is None
    assert Basic(alias).compare(other) == expected != 0


@pytest.mark.parametrize('order', [None, 'rev-lex'])
@pytest.mark.parametrize('wrap', [Basic, Expr, lambda value: Basic(Basic(value)),
    lambda value: Subs(Basic(value), x, 1)])
def test_class_order_survives_cached_wrapper(wrap, order):
    Meta = _class_alias_metaclass()
    first = Meta('Alpha', (), {'label': 1})
    alias = Meta('Zulu', (), {'label': 1})
    other = Meta('Middle', (), {'label': 2})
    reference = ref(first)
    first_expr, alias_expr, other_expr = map(wrap, (first, alias, other))
    first_key = default_sort_key(first_expr, order=order)
    alias_key = default_sort_key(alias_expr, order=order)
    other_key = default_sort_key(other_expr, order=order)
    assert first_key == alias_key
    expected = alias_key < other_key
    del first, first_expr, first_key, alias_key, other_key
    clear_cache()
    gc.collect()
    assert reference() is None
    alias_key = default_sort_key(alias_expr, order=order)
    other_key = default_sort_key(other_expr, order=order)
    assert (alias_key < other_key) == expected
    assert (other_key < alias_key) != expected


def test_class_order_reused_identity(monkeypatch):
    Meta = _class_alias_metaclass()
    monkeypatch.setattr(function_module, '_class_order_key', lambda cls: ('class', 0))
    first = Meta('Alpha', (), {'label': 1})
    alias = Meta('Zulu', (), {'label': 1})
    reference = ref(first)
    key = function_module._get_class_order_key(first, hash(first))
    assert function_module._get_class_order_key(alias, hash(alias)) == key
    del first
    clear_cache()
    gc.collect()
    assert reference() is None
    other = Meta('Alpha', (), {'label': 2})
    assert function_module._get_class_order_key(other, hash(other)) != key
    assert Basic(alias).compare(Basic(other)) != 0


def test_class_order_does_not_retain_names():
    class Name(str):
        __slots__ = ('owner',)

    class Meta(type):
        def __getattribute__(cls, name):
            if name == '__name__':
                return type.__getattribute__(cls, 'owned_name')
            return type.__getattribute__(cls, name)

    value = Meta('Original', (), {})
    name = Name('Alpha')
    name.owner = value
    value.owned_name = name
    reference = ref(value)
    assert Basic(value).compare(Basic(type('Omega', (), {}))) != 0
    del value, name
    clear_cache()
    gc.collect()
    assert reference() is None


def test_class_order_reentrant_equality():
    base = _class_alias_metaclass()
    retained = []
    armed = [False]

    class Meta(base):
        def __eq__(cls, other):
            if armed[0] and cls.label == 2 and isinstance(other, base) and other.label == 1:
                armed[0] = False
                alias = Meta('Zulu', (), {'label': 1})
                retained.append(alias)
                function_module._get_class_order_key(alias, hash(alias))
            return super().__eq__(other)

        __hash__ = base.__hash__

    other = Meta('Middle', (), {'label': 2})
    function_module._get_class_order_key(other, hash(other))
    value = Meta('Alpha', (), {'label': 1})
    armed[0] = True
    key = function_module._get_class_order_key(value, hash(value))
    assert not armed[0]
    assert key == function_module._get_class_order_key(retained[0], hash(retained[0]))
    assert Basic(value).compare(Basic(other)) == Basic(retained[0]).compare(Basic(other))


def test_class_order_equality_registers_unrelated_classes():
    base = _class_alias_metaclass()
    retained = []

    class Meta(base):
        def __eq__(cls, other):
            if cls.label == 2 and isinstance(other, base) and other.label == 1:
                if len(retained) == 20:
                    raise RuntimeError('unrelated registrations prevent progress')
                unrelated = type('Unrelated', (), {})
                retained.append(unrelated)
                function_module._get_class_order_key(unrelated, hash(unrelated))
            return super().__eq__(other)

        __hash__ = base.__hash__

    other = Meta('Middle', (), {'label': 2})
    function_module._get_class_order_key(other, hash(other))
    value = Meta('Alpha', (), {'label': 1})
    key = function_module._get_class_order_key(value, hash(value))
    assert retained
    assert key == function_module._get_class_order_key(value, hash(value))


def test_class_order_equality_collects_an_earlier_entry():
    base = _class_alias_metaclass()
    discarded = []
    armed = [False]

    class Meta(base):
        def __eq__(cls, other):
            if armed[0] and cls.label == 2 and isinstance(other, base) and other.label == 1:
                armed[0] = False
                discarded.clear()
                gc.collect()
            return super().__eq__(other)

        __hash__ = base.__hash__

    discarded.append(Meta('Discard', (), {'label': 0}))
    reference = ref(discarded[0])
    other = Meta('Middle', (), {'label': 2})
    alias = Meta('Zulu', (), {'label': 1})
    for value in (discarded[0], other, alias):
        function_module._get_class_order_key(value, hash(value))
    value = Meta('Alpha', (), {'label': 1})
    armed[0] = True
    key = function_module._get_class_order_key(value, hash(value))
    assert reference() is None
    assert key == function_module._get_class_order_key(alias, hash(alias))


def test_class_literal_sort_key_reconstruction():
    for value in (int, Symbol, cos):
        key = default_sort_key(value)
        clear_cache()
        restored = loads(dumps(key))
        assert key == restored == default_sort_key(value)
        assert hash(key) == hash(restored)


@pytest.mark.parametrize('value', [Symbol, cos, Function('converted_class', nargs=1)])
def test_class_literal_sort_key_conversion(value):
    class Convertible:
        def _sympy_(self):
            return value

    assert default_sort_key(Convertible()) == default_sort_key(value)


@pytest.mark.parametrize('order', [None, 'lex', 'rev-lex', 'grlex', 'grevlex'])
def test_sympified_sort_key_order(order):
    expr = x**2 + x + Symbol('y')

    class Convertible:
        def _sympy_(self):
            return expr

    assert default_sort_key(Convertible(), order=order) == default_sort_key(expr, order=order)


def test_class_literal_sort_key_collection():
    value = type('CollectedSortKey', (), {})
    reference = ref(value)
    key = default_sort_key(value)
    del value
    gc.collect()
    assert reference() is not None
    del key
    clear_cache()
    gc.collect()
    assert reference() is None


def test_class_and_metadata_comparison_key_order():
    values = [type('Omega', (), {}), ('opaque',), type('Alpha', (), {})]
    keys = [function_module._get_comparison_key(value) for value in values]
    expected = sorted(keys)
    for items in permutations(keys):
        assert sorted(items) == expected
    assert all(left < right for left, right in zip(expected, expected[1:]))


@pytest.mark.parametrize('order', [None, 'lex', 'grlex', 'grevlex', 'rev-lex'])
@pytest.mark.parametrize('wrap', [lambda value: value, Tuple,
    lambda value: Function('sort_key_parent')(value)])
def test_equal_subs_sort_keys(order, wrap):
    y, z = Symbol('y'), Symbol('z')
    f = Function('f')
    first = wrap(Subs(f(x), x, 0))
    clear_cache()
    first_key = default_sort_key(first, order=order)
    clear_cache()
    second = wrap(Subs(f(z), z, 0))
    second_key = default_sort_key(second, order=order)
    assert first == second
    assert first_key == second_key

    other = wrap(Subs(f(y), y, 1))
    clear_cache()
    left = sorted([first, other], key=lambda value: default_sort_key(value, order))
    clear_cache()
    right = sorted([second, other], key=lambda value: default_sort_key(value, order))
    assert left == right


def test_equal_numeric_metadata_sort_keys():
    first = Function('a', value=2)
    second = Function('a', value=Fraction(2))
    assert first == second
    assert default_sort_key(first) == default_sort_key(second)


def test_function_subclass_sort_keys():
    base = Function('sort_subclass_base')
    functions = [type('F', (base,), {}) for _ in range(2)] + [Function('F')]
    keys = [default_sort_key(function) for function in functions]
    assert len(set(keys)) == 3
    first, second = functions[:2]
    assert first(x).compare(second(x)) == -second(x).compare(first(x)) != 0
    assert first(x) + second(x) == second(x) + first(x)


def test_function_subclass_sort_key_collection():
    function = type('F', (Function('collect_subclass_base'),), {})
    reference = ref(function)
    default_sort_key(function)
    clear_cache()
    del function
    gc.collect()
    assert reference() is None


def test_equal_opaque_metadata_sort_keys():
    class Value:
        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            return isinstance(other, Value) and self.label == other.label

        def __hash__(self):
            return 0

    first = Function('a', value=Value(1))
    first_key = default_sort_key(first)
    clear_cache()
    second = Function('a', value=Value(1))
    other = Function('a', value=Value(2))
    assert first == second
    assert hash(first) == hash(second)
    assert first_key == default_sort_key(second)
    assert first_key != default_sort_key(other)
    assert first(x) + other(x) == second(x) + other(x) == other(x) + first(x)
    assert first(x).compare(other(x)) == -other(x).compare(second(x)) != 0
    del first, second
    clear_cache()
    gc.collect()
    rebuilt = Function('a', value=Value(1))
    assert first_key == default_sort_key(rebuilt)


@pytest.mark.parametrize('factory', ['key', 'function'])
@pytest.mark.parametrize('trigger', [1, 2, 3, 4])
@pytest.mark.thread_unsafe(reason='temporarily replaces the process-global comparison-key registry')
def test_metadata_sort_key_reentrant_equality(monkeypatch, factory, trigger):
    monkeypatch.setattr(function_module, '_comparison_keys',
        type(function_module._comparison_keys)())
    retained = []

    def create(value):
        if factory == 'key':
            return function_module._get_comparison_key(value)
        return function_module.UndefinedFunction('reentrant_field', value=value)

    def key(value):
        return value if factory == 'key' else default_sort_key(value)

    class Value:
        count = 0
        active = False

        def __init__(self, label):
            self.label = label

        def __hash__(self):
            return 0

        def __eq__(self, other):
            if not isinstance(other, Value):
                return NotImplemented
            if {self.label, other.label} == {0, 1} and not Value.active:
                Value.count += 1
                if Value.count == trigger:
                    Value.active = True
                    retained.extend((create(Value(2)), create(Value(1))))
                    Value.active = False
            return self.label == other.label

    values = [create(Value(0))]
    values.extend(create(Value(1)) for _ in range(trigger))
    assert len(retained) == 2
    other, inner = map(key, retained)
    for value in values[1:]:
        outer = key(value)
        assert outer == inner
        assert hash(outer) == hash(inner)
        assert not outer < inner and not inner < outer
        assert (outer < other) == (inner < other)
        assert (other < outer) == (other < inner)


@pytest.mark.thread_unsafe(reason='temporarily replaces the process-global comparison-key registry')
def test_metadata_sort_key_collected_during_lookup(monkeypatch):
    retained = []

    class Registry(type(function_module._comparison_keys)):
        def setdefault(self, key, default=None):
            reference = super().setdefault(key, default)
            retained.clear()
            return reference

    registry = Registry()
    monkeypatch.setattr(function_module, '_comparison_keys', registry)
    value = Symbol('collected_comparison_key')
    first = function_module._get_comparison_key(value)
    old_reference = ref(first)
    retained.append(first)
    del first
    replacement = function_module._get_comparison_key(value)
    assert old_reference() is None
    assert replacement.value is value
    assert function_module._get_comparison_key(value) is replacement
    assert len(registry) == 1


@pytest.mark.thread_unsafe(reason='temporarily replaces the process-global comparison-key registry')
def test_metadata_sort_key_equality_error(monkeypatch):
    monkeypatch.setattr(function_module, '_comparison_keys',
        type(function_module._comparison_keys)())

    class Value:
        def __hash__(self):
            return 0

        def __eq__(self, other):
            if self is other:
                return True
            raise RuntimeError('metadata equality failed')

    first = function_module._get_comparison_key(Value())
    with raises(RuntimeError, match='metadata equality failed'):
        function_module._get_comparison_key(Value())
    assert function_module._comparison_keys.get(first)() is first


def test_metadata_sort_key_collection():
    class Value:
        pass

    value = Value()
    function = Function('a', value=value)
    value.owner = function
    value_ref, function_ref = ref(value), ref(function)
    default_sort_key(function)
    clear_cache()
    del value, function
    gc.collect()
    assert value_ref() is None
    assert function_ref() is None


def test_metadata_sort_key_reconstruction():
    value = Symbol('pickle_key_value')
    first = Function('a', value=value)
    saved_key = loads(dumps(default_sort_key(first)))
    other = Function('a', value=Symbol('other_pickle_key_value'))
    other_key = default_sort_key(other)
    assert saved_key < other_key
    del first
    clear_cache()
    gc.collect()
    rebuilt = Function('a', value=value)
    assert saved_key == default_sort_key(rebuilt)
    assert saved_key < other_key
    assert default_sort_key(rebuilt) < other_key


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
@pytest.mark.thread_unsafe(reason='forks while another thread holds a lock')
@pytest.mark.filterwarnings('ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning')
def test_metadata_comparison_after_fork():
    value = Symbol('fork_key_value')
    first = Function('a', value=value)
    other_key = default_sort_key(Function('a', value=Symbol('other_fork_key_value')))
    saved_key = default_sort_key(first)
    acquired, release = Event(), Event()

    def hold_lock():
        with function_module._comparison_key_lock:
            acquired.set()
            release.wait()

    def construct():
        clear_cache()
        rebuilt = Function('a', value=value)
        key = default_sort_key(rebuilt)
        assert first == rebuilt
        assert key == saved_key
        assert (key < other_key) == (saved_key < other_key)
        assert hash(Subs(x, x, 0)) == hash(Subs(Symbol('y'), Symbol('y'), 0))

    thread = Thread(target=hold_lock, daemon=True)
    process = get_context('fork').Process(target=construct)
    thread.start()
    try:
        assert acquired.wait(5)
        process.start()
    finally:
        release.set()
        thread.join(5)
    try:
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)


def test_metadata_comparison_lock_owned_reset():
    lock = function_module._comparison_key_lock
    with lock, lock:
        function_module._reset_comparison_key_lock()
        assert function_module._comparison_key_lock is lock


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
