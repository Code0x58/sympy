"""This tests sympy/core/basic.py with (ideally) no reference to subclasses
of Basic or Atom."""
from __future__ import annotations
import collections
from decimal import Decimal
from fractions import Fraction
from typing import TypeVar, Generic

from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
    _atomic, _aresame)
from sympy.core.cache import clear_cache
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi, Float
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.relational import Eq


b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)
T = TypeVar('T')


def test__aresame():
    assert not _aresame(Basic(Tuple()), Basic())
    for i, j in [(S(2), S(2.)), (1., Float(1))]:
        for do in range(2):
            assert not _aresame(Basic(i), Basic(j))
            assert not _aresame(i, j)
            i, j = j, i


def test_structure():
    assert b21.args == (b2, b1)
    assert b21.func(*b21.args) == b21
    assert bool(b1)


def test_immutable():
    assert not hasattr(b1, '__dict__')
    with raises(AttributeError):
        b1.x = 1


def test_compare_function_classes():
    f, g = Function('f'), Function('g')
    assert Tuple(f).compare(Tuple(g)) == -1
    assert Tuple(g).compare(Tuple(f)) == 1
    assert Tuple(sin).compare(Tuple(cos)) == 1
    assert Tuple(cos).compare(Tuple(sin)) == -1
    assert Tuple(f).compare(Tuple(f)) == 0
    x = Symbol('x')
    assert Tuple(f).compare(Tuple(x)) == -Tuple(x).compare(Tuple(f)) != 0
    named_like_metaclass = Function('UndefinedFunction')(x)
    assert named_like_metaclass.compare(f) == -1
    assert Tuple(f).compare(Tuple(named_like_metaclass)) == 1
    assert Tuple(named_like_metaclass).compare(Tuple(f)) == -1

    for first, second in [
        (Function('a', nargs=1), Function('a', nargs=2)),
        (Function('a', positive=True), Function('a', negative=True)),
    ]:
        assert first != second
        assert Tuple(first).compare(Tuple(second)) == -Tuple(second).compare(Tuple(first)) != 0
    first = Function('a', positive=True)(x)
    second = Function('a', negative=True)(x)
    assert first.compare(second) == -second.compare(first) != 0
    assert first + second == second + first


def test_compare_class_and_native_arguments():
    for cls in (Function('f'), Symbol, int):
        first = Basic(cls)
        for value in (None, 1, 1.0, False, 'f', (1,), frozenset([1])):
            second = Basic(value)
            assert first.compare(second) == -second.compare(first) != 0


def test_compare_class_and_native_transitivity():
    first = Basic(frozenset([S.One]))
    second = Basic('f')
    third = Basic(int)
    assert first.compare(second) < 0
    assert second.compare(third) < 0
    assert first.compare(third) < 0


def test_compare_frozenset_arguments():
    for convert in (int, S):
        values = [frozenset(map(convert, items))
                  for items in ((1, 9), (9, 1), (8, 1))]
        for nested in (False, True):
            first, second, other = [Basic(frozenset([value]) if nested else value)
                                    for value in values]
            assert first == second
            assert first.compare(second) == second.compare(first) == 0
            assert first.compare(other) == second.compare(other) != 0
            assert first.compare(other) == -other.compare(first)


def test_custom_frozenset_ordering_alias():
    class Label:
        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            return isinstance(other, Label) and self.label == other.label

        def __hash__(self):
            return hash(self.label)

        def __lt__(self, other):
            return self.label < other.label if isinstance(other, Label) else NotImplemented

        def __gt__(self, other):
            return self.label > other.label if isinstance(other, Label) else NotImplemented

    class LabelSet(Label, frozenset):
        def __new__(cls, label):
            return frozenset.__new__(cls, [0])

    for wrap in (lambda value: value, lambda value: (value,),
                 lambda value: frozenset([value])):
        first, alias, other = [Basic(wrap(value))
            for value in (LabelSet(0), Label(0), LabelSet(1))]
        assert first == alias
        assert hash(first) == hash(alias)
        assert first.compare(alias) == alias.compare(first) == 0
        assert first.compare(other) == alias.compare(other) == -1
        assert other.compare(first) == other.compare(alias) == 1
        keys = []
        for item in (first, alias, other):
            clear_cache()
            keys.append(default_sort_key(item))
        assert keys[0] == keys[1]
        assert hash(keys[0]) == hash(keys[1])
        assert keys[0] < keys[2] and keys[1] < keys[2]
        assert keys[2] > keys[0] and keys[2] > keys[1]

    class FailingLabel(Label):
        def __lt__(self, other):
            raise RuntimeError('label ordering failed')

    with raises(RuntimeError, match='label ordering failed'):
        Basic(FailingLabel(0)).compare(Basic(LabelSet(1)))


def test_frozenset_argument_kinds():
    for native, symbolic in [(1, S.One), (1.0, Float(1)), (True, S.true)]:
        for nested in (False, True):
            left, right = frozenset([native]), frozenset([symbolic])
            if nested:
                left, right = frozenset([left]), frozenset([right])
            first, second = Basic(left), Basic(right)
            assert first != second
            assert first.compare(second) == -second.compare(first) != 0


def test_frozenset_native_numeric_equivalence():
    for one in (1, True, 1.0, Fraction(1), Decimal(1)):
        for nested in (False, True):
            values = [frozenset(items) for items in ((1, 2), (one, 2), (0, 2), (1, 3))]
            if nested:
                values = [frozenset([value]) for value in values]
            first, equivalent, lower, upper = map(Basic, values)
            assert first == equivalent
            assert hash(first) == hash(equivalent)
            assert first.compare(equivalent) == 0
            assert first.compare(lower) == equivalent.compare(lower) == 1
            assert first.compare(upper) == equivalent.compare(upper) == -1


def test_tuple_argument_identity():
    for wrap in (tuple, lambda value: (tuple(value),),
                 lambda value: frozenset([tuple(value)]),
                 lambda value: (frozenset(value),)):
        first = Basic(wrap((1, 2)))
        for alias in (True, 1.0, Fraction(1), Decimal(1)):
            equivalent = Basic(wrap((alias, 2)))
            assert first == equivalent
            assert hash(first) == hash(equivalent)
            assert first.compare(equivalent) == equivalent.compare(first) == 0
        for symbolic in (S.One, Float(1), S.true):
            other = Basic(wrap((symbolic, 2)))
            assert first != other
            assert first.compare(other) == -other.compare(first) != 0


def test_tuple_argument_storage():
    class StoredTuple(tuple):
        __slots__ = ()

        def __iter__(self):
            return iter([2])

        def __len__(self):
            raise TypeError('stored contents do not use overridden iteration or length')

    first = Basic(StoredTuple((S.One, 9)))
    equivalent = Basic((S.One, 9))
    other = Basic((1, 9))
    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert first != other
    assert first.compare(equivalent) == 0
    assert first.compare(other) == equivalent.compare(other) != 0


def test_tuple_argument_ordering():
    for left, right in [((1, 2), (2,)), ((1,), (1, 0)), ((1, 2), (1, 3))]:
        first, second = Basic(left), Basic(right)
        assert first.compare(second) == -1
        assert second.compare(first) == 1
        assert default_sort_key(first) < default_sort_key(second)


def test_frozenset_argument_storage():
    class StoredSet(frozenset):
        def __iter__(self):
            return iter([2])

        def __len__(self):
            raise TypeError('stored contents do not use overridden iteration or length')

    first = Basic(StoredSet([1, 9]))
    equivalent = Basic(frozenset([1, 9]))
    other = Basic(frozenset([1, 8]))
    assert first == equivalent
    assert first.compare(equivalent) == 0
    assert first.compare(other) == equivalent.compare(other) == 1


def test_compare_equal_unordered_frozenset_arguments():
    first = Basic(frozenset([None, 'a', 1j]))
    second = Basic(frozenset([1j, 'a', None]))
    assert first == second
    assert first.compare(second) == second.compare(first) == 0


def test_frozenset_custom_equality():
    class EquivalentSet(frozenset):
        def __eq__(self, other):
            return isinstance(other, EquivalentSet)

        def __hash__(self):
            return 0

    first, second = Basic(EquivalentSet([1])), Basic(EquivalentSet([2]))
    assert first == second
    assert hash(first) == hash(second)
    assert first.compare(second) == second.compare(first) == 0


def test_frozenset_metaclass_equality_attribute():
    class Meta(type):
        __dict__ = property(lambda cls: {})

        def __getattribute__(cls, name):
            if name == '__eq__':
                return frozenset.__eq__
            return super().__getattribute__(name)

    class EquivalentSet(frozenset, metaclass=Meta):
        def __eq__(self, other):
            return isinstance(other, EquivalentSet)

        def __hash__(self):
            return 0

    first, second = Basic(EquivalentSet([1])), Basic(EquivalentSet([2]))
    assert first == second
    assert first.compare(second) == second.compare(first) == 0


def test_compare_frozenset_sort_key_ties():
    class CollidingSymbol(Symbol):
        def __hash__(self):
            return 0

    positive = CollidingSymbol('a', positive=True)
    negative = CollidingSymbol('a', negative=True)
    zero = CollidingSymbol('a', zero=True)
    assert default_sort_key(positive) == default_sort_key(negative)
    first, second, other = [Basic(frozenset(items)) for items in (
        (positive, negative), (negative, positive), (positive, zero))]
    assert first == second
    assert first.compare(second) == second.compare(first) == 0
    assert first.compare(other) == second.compare(other) != 0
    assert first.compare(other) == -other.compare(first)


def test_compare_frozenset_and_native_arguments():
    first = Basic(frozenset([S.One]))
    for value in (None, 1, 1.0, 1 + 2j, False, 'f', (1,)):
        second = Basic(value)
        assert first.compare(second) == -second.compare(first) != 0


def test_compare_frozenset_complex_arguments():
    for convert in (int, float, Fraction, Decimal):
        one = convert(1)
        first = Basic(frozenset([one, 2j]))
        second = Basic(frozenset([2j, one]))
        lower = Basic(frozenset([0, 3j]))
        upper = Basic(frozenset([one, 3j]))
        assert first == second
        assert first.compare(lower) == second.compare(lower) == 1
        assert lower.compare(first) == -1
        assert first.compare(upper) == second.compare(upper) == -1
        assert upper.compare(first) == 1


def test_native_complex_argument_ordering():
    groups = [
        (-2, -2 + 0j), (-2 + 1j,),
        (False, 0, 0.0, 0j, Decimal(0), Fraction(0)), (1j,),
        (Fraction(1, 2), Decimal('0.5'), 0.5, 0.5 + 0j),
        (True, 1, 1.0, 1 + 0j, Decimal(1), Fraction(1)),
        (1 + 2j,), (2, 2 + 0j),
    ]
    for i, left in enumerate(groups):
        for j, right in enumerate(groups):
            for a in left:
                for b in right:
                    assert Basic(a).compare(Basic(b)) == (i > j) - (i < j)


def test_native_complex_nan_ordering():
    for left, right in [
        (complex('nan'), 1),
        (complex(0, float('nan')), 0),
        (complex('nan'), 1j),
        (complex(0, float('nan')), 1j),
        (1j, float('nan')),
        (1j, Decimal('NaN')),
    ]:
        with raises(TypeError):
            Basic(left).compare(Basic(right))
        with raises(TypeError):
            Basic(right).compare(Basic(left))


def test_compare_shared_native_nan():
    for constructor in (float, complex, Decimal):
        value = constructor('nan')
        first, second = Basic(value), Basic(value)
        assert first == second
        assert first.compare(second) == second.compare(first) == 0
        assert Basic(value, 0).compare(Basic(value, 1)) == -1


def test_native_complex_subclass_ordering():
    class TaggedComplex(complex):
        def __new__(cls, value, tag):
            obj = super().__new__(cls, value)
            obj.tag = tag
            return obj

        def __eq__(self, other):
            return type(other) is type(self) and self.tag == other.tag

        def __hash__(self):
            return hash(self.tag)

        def __lt__(self, other):
            return self.tag < other.tag

        def __gt__(self, other):
            return self.tag > other.tag

    first = Basic(TaggedComplex(1 + 2j, 1))
    second = Basic(TaggedComplex(1 + 2j, 2))
    assert first != second
    assert first.compare(second) == -1
    assert second.compare(first) == 1


def test_sort_key_class_and_native_arguments():
    for cls in (Function('f'), Symbol, int):
        first = Basic(cls)
        for value in (None, 1, 1.0, False, 'f', (1,), frozenset([1]), Symbol('x')):
            second = Basic(value)
            assert sorted([first, second], key=default_sort_key) == [second, first]
            assert sorted([second, first], key=default_sort_key) == [second, first]
    first = Basic(1)
    first_key = first.sort_key()
    for value in (1.0, True):
        clear_cache()
        second = Basic(value)
        assert first == second
        assert first_key == second.sort_key()


def test_native_and_sympy_argument_ordering():
    x = Symbol('x')
    for symbolic in (S.One, x, Basic(x)):
        first = Basic(symbolic)
        for native in (None, 1, 1.0, False, 'x', (1,), frozenset([1])):
            second = Basic(native)
            assert first != second and second != first
            assert first.compare(second) == -1
            assert second.compare(first) == 1
            clear_cache()
            assert sorted([first, second], key=default_sort_key) == [first, second]
            clear_cache()
            assert sorted([second, first], key=default_sort_key) == [first, second]


def test_native_and_sympy_argument_identity():
    for native, symbolic in [(1, S.One), (1.0, Float(1)),
            (True, S.true), (False, S.false), ((1,), Tuple(1))]:
        first, second = Basic(native), Basic(symbolic)
        assert first.xreplace({native: S(2)}) == first
        assert second.xreplace({native: S(2)}) == Basic(S(2))
        assert first != second
        assert second != first


def test_argument_kinds_before_equality():
    class Value:
        def __eq__(self, other):
            raise RuntimeError('native equality failed')

        __hash__ = object.__hash__

    for native in (Fraction(1), Value()):
        for symbolic in (S.One, Float(1), Symbol):
            for wrap in (lambda value: value, lambda value: (value,),
                         lambda value: ((value,),), lambda value: frozenset([value])):
                first, second = Basic(wrap(native)), Basic(wrap(symbolic))
                assert first != second and second != first
                assert first.compare(second) == -second.compare(first) != 0
                assert default_sort_key(first) != default_sort_key(second)

    for wrap in (lambda value: value, lambda value: (value,)):
        with raises(RuntimeError, match='native equality failed'):
            assert Basic(wrap(Value())) == Basic(wrap(Value()))


def test_argument_equality_short_circuit():
    class Value(Basic):
        def __eq__(self, other):
            raise RuntimeError('unreached equality')

        __hash__ = Basic.__hash__

    first_value, second_value = Value(0), Value(1)
    for first, second in [
        (Basic(0, first_value), Basic(1, second_value)),
        (Basic((0, first_value)), Basic((1, second_value))),
        (Basic(first_value), Basic(second_value, second_value)),
        (Basic((first_value,)), Basic((second_value, second_value))),
    ]:
        assert first != second and second != first


def test_symbol_equality_compatibility():
    class Name(str):
        __slots__ = ()

    class SymbolSubclass(Symbol):
        def _hashable_content(self):
            return ()

    values = [Symbol('x'), Symbol('y'), Symbol('x', positive=True),
              Symbol(Name('name')), Symbol('name'), Dummy('x'),
              SymbolSubclass('x'), SymbolSubclass('y')]
    for first in values:
        for second in [*values, S.One, Float(1), Basic(), 1, 'x', None]:
            assert first.__eq__(second) is Basic.__eq__(first, second)
            if first == second:
                assert hash(first) == hash(second)

    class FailingName(str):
        __slots__ = ()

        def __eq__(self, other):
            raise RuntimeError('symbol name equality failed')

        __hash__ = str.__hash__

    first = Symbol.__xnew__(Symbol, FailingName('first'))
    second = Symbol.__xnew__(Symbol, 'second')
    with raises(RuntimeError, match='symbol name equality failed'):
        assert first == second


def test_native_argument_equality_without_ordering():
    for first, second in [(None, None), (1, 1 + 0j)]:
        first, second = Basic(first), Basic(second)
        assert first == second
        assert first.compare(second) == second.compare(first) == 0
        clear_cache()
        first_key = first.sort_key()
        clear_cache()
        assert first_key == second.sort_key()


def test_native_and_class_argument_identity():
    class Value:
        def __eq__(self, other):
            return other is Symbol or isinstance(other, Value)

        def __hash__(self):
            return hash(Symbol)

    first, second = Basic(Symbol), Basic(Value())
    assert first.xreplace({Symbol: S(2)}) == Basic(S(2))
    assert second.xreplace({Symbol: S(2)}) == second
    assert first != second and second != first


def test_subs_class_arguments():
    x, y = symbols('x y')
    f, g = Function('f'), Function('g')
    for cls in (f, Symbol):
        source = Basic(cls, x)
        assert source.subs(x, y) == Basic(cls, y)
        assert source.subs(cls, g) == Basic(g, x)
        assert source.xreplace({x: y}) == Basic(cls, y)
        assert source.xreplace({cls: g}) == Basic(g, x)
        assert source.replace(lambda node: node == x, lambda node: y) == Basic(cls, y)
        assert source.has(x)
        assert not source.has(y)


def test_free_symbols_class_arguments():
    x, y = symbols('x y')
    function = Function('f')
    for cls in (function, Symbol, int):
        source = Basic(cls, x + y)
        assert source.free_symbols == {x, y}
    for cls in (function, Symbol):
        bound = Lambda(x, Tuple(cls, x, y))
        assert bound.free_symbols == {y}
        assert bound.as_dummy().free_symbols == {y}

    class Value:
        free_symbols = {y}

    assert Basic(Value(), x).free_symbols == {x, y}


def test_rewrite_class_arguments():
    x = Symbol('x')
    for cls in (Function('f'), Symbol, int):
        source = Basic(cls, sin(x))
        expected = Basic(cls, sin(x).rewrite(exp))
        assert source.rewrite(exp) == expected
        assert source.rewrite(sin, exp) == expected
        assert source.rewrite(exp, deep=False) == source


def test_equality():
    instances = [b1, b2, b3, b21, Basic(b1, b1, b1), Basic]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            assert (b_i == b_j) == (i == j)
            assert (b_i != b_j) == (i != j)

    assert Basic() != []
    assert not(Basic() == [])
    assert Basic() != 0
    assert not(Basic() == 0)

    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """

    b = Basic()
    foo = Foo()

    assert b != foo
    assert foo != b
    assert not b == foo
    assert not foo == b

    class Bar:
        """
        Class that considers itself equal to any instance of Basic, and relies
        on Basic returning the NotImplemented singleton in order to achieve
        a symmetric equivalence relation.

        """
        def __eq__(self, other):
            if isinstance(other, Basic):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    assert b == bar
    assert bar == b
    assert not b != bar
    assert not bar != b


def test_matches_basic():
    instances = [Basic(b1, b1, b2), Basic(b1, b2, b1), Basic(b2, b1, b1),
                 Basic(b1, b2), Basic(b2, b1), b2, b1]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            if i == j:
                assert b_i.matches(b_j) == {}
            else:
                assert b_i.matches(b_j) is None
    assert b1.match(b1) == {}


def test_has():
    assert b21.has(b1)
    assert b21.has(b3, b1)
    assert b21.has(Basic)
    assert not b1.has(b21, b3)
    assert not b21.has()
    assert not b21.has(str)
    assert not Symbol("x").has("x")


def test_subs():
    assert b21.subs(b2, b1) == Basic(b1, b1)
    assert b21.subs(b2, b21) == Basic(b21, b1)
    assert b3.subs(b2, b1) == b2

    assert b21.subs([(b2, b1), (b1, b2)]) == Basic(b2, b2)

    assert b21.subs({b1: b2, b2: b1}) == Basic(b2, b2)
    assert b21.subs(collections.ChainMap({b1: b2}, {b2: b1})) == Basic(b2, b2)
    assert b21.subs(collections.OrderedDict([(b2, b1), (b1, b2)])) == Basic(b2, b2)

    raises(ValueError, lambda: b21.subs('bad arg'))
    raises(TypeError, lambda: b21.subs(b1, b2, b3))
    # dict(b1=foo) creates a string 'b1' but leaves foo unchanged; subs
    # will convert the first to a symbol but will raise an error if foo
    # cannot be sympified; sympification is strict if foo is not string
    raises(TypeError, lambda: b21.subs(b1='bad arg'))

    assert Symbol("text").subs({"text": b1}) == b1
    assert Symbol("s").subs({"s": 1}) == 1


def test_subs_with_unicode_symbols():
    expr = Symbol('var1')
    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'

    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'


def test_atoms():
    assert b21.atoms() == {Basic()}


def test_free_symbols_empty():
    assert b21.free_symbols == set()


def test_doit():
    assert b21.doit() == b21
    assert b21.doit(deep=False) == b21


def test_S():
    assert repr(S) == 'S'


def test_xreplace():
    assert b21.xreplace({b2: b1}) == Basic(b1, b1)
    assert b21.xreplace({b2: b21}) == Basic(b21, b1)
    assert b3.xreplace({b2: b1}) == b2
    assert Basic(b1, b2).xreplace({b1: b2, b2: b1}) == Basic(b2, b1)
    assert Atom(b1).xreplace({b1: b2}) == Atom(b1)
    assert Atom(b1).xreplace({Atom(b1): b2}) == b2
    raises(TypeError, lambda: b1.xreplace())
    raises(TypeError, lambda: b1.xreplace([b1, b2]))
    for f in (exp, Function('f')):
        assert f.xreplace({}) == f
        assert f.xreplace({}, hack2=True) == f
        assert f.xreplace({f: b1}) == b1
        assert f.xreplace({f: b1}, hack2=True) == b1


def test_sorted_args():
    x = symbols('x')
    assert b21._sorted_args == b21.args
    raises(AttributeError, lambda: x._sorted_args)

def test_call():
    x, y = symbols('x y')
    # See the long history of this in issues 5026 and 5105.

    raises(TypeError, lambda: sin(x)({ x : 1, sin(x) : 2}))
    raises(TypeError, lambda: sin(x)(1))

    # No effect as there are no callables
    assert sin(x).rcall(1) == sin(x)
    assert (1 + sin(x)).rcall(1) == 1 + sin(x)

    # Effect in the presence of callables
    l = Lambda(x, 2*x)
    assert (l + x).rcall(y) == 2*y + x
    assert (x**l).rcall(2) == x**4
    # TODO UndefinedFunction does not subclass Expr
    #f = Function('f')
    #assert (2*f)(x) == 2*f(x)

    assert (Q.real & Q.positive).rcall(x) == Q.real(x) & Q.positive(x)


def test_rewrite():
    x, y, z = symbols('x y z')
    a, b = symbols('a b')
    f1 = sin(x) + cos(x)
    assert f1.rewrite(cos,exp) == exp(I*x)/2 + sin(x) + exp(-I*x)/2
    assert f1.rewrite([cos],sin) == sin(x) + sin(x + pi/2, evaluate=False)
    f2 = sin(x) + cos(y)/gamma(z)
    assert f2.rewrite(sin,exp) == -I*(exp(I*x) - exp(-I*x))/2 + cos(y)/gamma(z)

    assert f1.rewrite() == f1

def test_literal_evalf_is_number_is_zero_is_comparable():
    x = symbols('x')
    f = Function('f')

    # issue 5033
    assert f.is_number is False
    # issue 6646
    assert f(1).is_number is False
    i = Integral(0, (x, x, x))
    # expressions that are symbolically 0 can be difficult to prove
    # so in case there is some easy way to know if something is 0
    # it should appear in the is_zero property for that object;
    # if is_zero is true evalf should always be able to compute that
    # zero
    assert i.n() == 0
    assert i.is_zero
    assert i.is_number is False
    assert i.evalf(2, strict=False) == 0

    # issue 10268
    n = sin(1)**2 + cos(1)**2 - 1
    assert n.is_comparable is False
    assert n.n(2).is_comparable is False
    assert n.n(2).n(2).is_comparable


def test_as_Basic():
    assert as_Basic(1) is S.One
    assert as_Basic(()) == Tuple()
    raises(TypeError, lambda: as_Basic([]))


def test_atomic():
    g, h = map(Function, 'gh')
    x = symbols('x')
    assert _atomic(g(x + h(x))) == {g(x + h(x))}
    assert _atomic(g(x + h(x)), recursive=True) == {h(x), x, g(x + h(x))}
    assert _atomic(1) == set()
    assert _atomic(Basic(S(1), S(2))) == set()


def test_as_dummy():
    u, v, x, y, z, _0, _1 = symbols('u v x y z _0 _1')
    assert Lambda(x, x + 1).as_dummy() == Lambda(_0, _0 + 1)
    assert Lambda(x, x + _0).as_dummy() == Lambda(_1, _0 + _1)
    eq = (1 + Sum(x, (x, 1, x)))
    ans = 1 + Sum(_0, (_0, 1, x))
    once = eq.as_dummy()
    assert once == ans
    twice = once.as_dummy()
    assert twice == ans
    assert Integral(x + _0, (x, x + 1), (_0, 1, 2)
        ).as_dummy() == Integral(_0 + _1, (_0, x + 1), (_1, 1, 2))
    for T in (Symbol, Dummy):
        d = T('x', real=True)
        D = d.as_dummy()
        assert D != d and D.func == Dummy and D.is_real is None
    assert Dummy().as_dummy().is_commutative
    assert Dummy(commutative=False).as_dummy().is_commutative is False


def test_as_dummy_native_arguments():
    class Node:
        def __init__(self, arg):
            self.args = (arg,)

        @property
        def func(self):
            return type(self)

        _xreplace = Basic._xreplace

    x, canonical = symbols('x _0')
    source = Basic(Node(Integral(x, (x, 0, 1))))
    result = source.as_dummy().args[0].args[0]
    assert result == Integral(canonical, (canonical, 0, 1))


def test_transform_native_leaf_arguments():
    class Children:
        _iterable = False

        def __iter__(self):
            raise AssertionError('iteration is disabled')

    class Leaf:
        def __init__(self, args):
            self.args = args

    x, y, canonical = symbols('x y _0')
    for args in (None, 0, 1, Children()):
        leaf = Leaf(args)
        source = Basic(leaf, Integral(x, (x, 0, 1)), y)
        assert source.as_dummy() == Basic(leaf, Integral(canonical, (canonical, 0, 1)), y)
        assert source.replace(lambda node: node == y,
                              lambda node: x) == Basic(leaf, Integral(x, (x, 0, 1)), x)
        expected = source if args is None else Basic(x, Integral(x, (x, 0, 1)), y)
        assert source.replace(lambda node: node is leaf, lambda node: x) == expected


def test_replace_native_child_sequences():
    x, y, z = symbols('x y z')
    for factory in (tuple, list, iter):
        class Node:
            @property
            def args(self):
                return factory((x,))

        node = Node()
        for source in (Basic(node, y), Basic(y, node)):
            result, changes = source.replace(lambda item: False, lambda item: z, map=True)
            assert result is source
            assert changes == {}
            result = source.replace(lambda item: item is y, lambda item: z)
            assert result.args == tuple(z if item is y else item for item in source.args)


def test_replace_native_child_sequences_simultaneous():
    y, z, t = symbols('y z t')
    for factory in (tuple, list, iter):
        class Projection:
            @property
            def args(self):
                return factory((y, z))

            @staticmethod
            def func(*children):
                return children[0]

        source = Basic(Projection())
        for simultaneous in (False, True):
            for mapping in (False, True):
                result = source.replace(lambda item: item is y or item is z,
                    lambda item: {y: z, z: t}[item], simultaneous=simultaneous, map=mapping)
                if mapping:
                    result, changes = result
                    assert changes == {y: z, z: t}
                assert result == Basic(z if simultaneous else t)


def test_as_dummy_scopes():
    u, v, x, y, _0, _1 = symbols('u v x y _0 _1')

    # Nested scopes must have distinct bound variables.
    a = Lambda(y, Lambda(x, x + y))
    expected = Lambda(_0, Lambda(_1, _0 + _1))
    assert a.as_dummy() == expected

    # Alpha-equivalent expressions have the same canonical form.
    b = Lambda(v, Lambda(u, u + v))
    assert b.as_dummy() == expected

    # Shadowing must not change the meaning.
    assert Lambda(x, Lambda(x, x + y)).as_dummy() == \
        Lambda(_0, Lambda(_1, _1 + y))

    # Independent sibling scopes can reuse canonical variables.
    a = Basic(Lambda(x, x + 1), Lambda(y, y + 2))
    b = Basic(Lambda(v, v + 1), Lambda(u, u + 2))
    expected = Basic(
        Lambda(_0, _0 + 1), Lambda(_0, _0 + 2))
    assert a.as_dummy() == expected
    assert b.as_dummy() == expected

    # Alpha-equivalent Lambdas combine when canonicalized as part of Add.
    a = Lambda(x, Lambda(y, x + y))
    b = Lambda(y, Lambda(x, x + y))
    assert (a + b).as_dummy() == 2*Lambda(
        _0, Lambda(_1, _0 + _1))

    # Canonicalization is idempotent.
    assert expected.as_dummy() == expected


def test_canonical_variables():
    x, i0, i1 = symbols('x _:2')
    assert Integral(x, (x, x + 1)).canonical_variables == {x: i0}
    assert Integral(x, (x, x + 1), (i0, 1, 2)).canonical_variables == {
        x: i0, i0: i1}
    assert Integral(x, (x, x + i0)).canonical_variables == {x: i1}


def test_replace_exceptions():
    from sympy.core.symbol import Wild
    x, y = symbols('x y')
    e = (x**2 + x*y)
    raises(TypeError, lambda: e.replace(sin, 2))
    b = Wild('b')
    c = Wild('c')
    raises(TypeError, lambda: e.replace(b*c, c.is_real))
    raises(TypeError, lambda: e.replace(b.is_real, 1))
    raises(TypeError, lambda: e.replace(lambda d: d.is_Number, 1))


def test_ManagedProperties():
    # ManagedProperties is now deprecated. Here we do our best to check that if
    # someone is using it then it does work in the way that it previously did
    # but gives a deprecation warning.
    from sympy.core.assumptions import ManagedProperties

    myclasses = []

    class MyMeta(ManagedProperties):
        def __init__(cls, *args, **kwargs):
            myclasses.append('executed')
            super().__init__(*args, **kwargs)

    code = """
class MySubclass(Basic, metaclass=MyMeta):
    pass
"""
    with warns_deprecated_sympy():
        exec(code)

    assert myclasses == ['executed']


def test_generic():
    # https://github.com/sympy/sympy/issues/25399
    class A(Symbol, Generic[T]):
        pass

    class B(A[T]):
        pass


def test_rewrite_abs():
    # https://github.com/sympy/sympy/issues/27323
    x = Symbol('x')
    assert sign(x).rewrite(abs) == sign(x).rewrite(Abs)
    assert sign(x).rewrite(abs) == Piecewise((0, Eq(x, 0)), (x / Abs(x), True))
