"""ASV cases for argument identity, metadata, ordering, and substitution."""

from __future__ import annotations

from sympy import Basic, Expr, Float, Function, S, Subs, Symbol, default_sort_key, sin
from sympy.core.cache import clear_cache
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.traversal import preorder_traversal, postorder_traversal


class _Label:
    def __init__(self, label):
        self.label = label

    def __eq__(self, other):
        return isinstance(other, _Label) and self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __lt__(self, other):
        return self.label < other.label if isinstance(other, _Label) else NotImplemented

    def __gt__(self, other):
        return self.label > other.label if isinstance(other, _Label) else NotImplemented


class _LabelSet(_Label, frozenset):
    def __new__(cls, label):
        return frozenset.__new__(cls, [0])


class TimeNativeFields:
    def setup(self):
        self.native = Basic((1, 2))
        self.alias = Basic((1.0, 2.0))
        self.other = Basic((2, 0))
        self.keys = default_sort_key(self.native), default_sort_key(self.other)
        self.typed_keys = (default_sort_key(Basic((S.One, S(2)))),
                           default_sort_key(Basic((S(2), S(3)))))
        self.floats = Float('1.2', 50), Float('1.3', 50)
        self.metadata_value = Symbol('metadata_value')
        self.metadata_reference = UndefinedFunction('metadata_bench', value=self.metadata_value)
        self.custom_plain = Basic(_Label(0))
        self.custom_set = Basic(_LabelSet(1))

    def time_tuple_equality(self):
        self.native == self.alias

    def time_tuple_comparison(self):
        self.native.compare(self.other)

    def time_tuple_key_comparison(self):
        self.keys[0] < self.keys[1]

    def time_typed_tuple_key_comparison(self):
        self.typed_keys[0] < self.typed_keys[1]

    def time_float_comparison(self):
        self.floats[0].compare(self.floats[1])

    def time_metadata_factory(self):
        UndefinedFunction('metadata_bench', value=self.metadata_value)

    def time_custom_frozenset_comparison(self):
        self.custom_plain.compare(self.custom_set)


class TimeArgumentEquality:
    def setup(self):
        self.symbols = Symbol('equality_x'), Symbol('equality_y')
        function = Function('equality_bench')
        self.expressions = tuple(map(function, self.symbols))
        self.expression_key = default_sort_key(self.expressions[0])

    def time_symbol_equality(self):
        self.symbols[0] == self.symbols[1]

    def time_expression_equality(self):
        self.expressions[0] == self.expressions[1]

    def time_expression_sort_key(self):
        default_sort_key(self.expressions[0])


class TimeRecreatedFactoryAliases:
    def setup(self):
        argument = Symbol('factory_alias_argument')
        self.first = Function('factory_alias_bench', value=1)(argument)
        clear_cache()
        self.second = Function('factory_alias_bench', value=S.One)(argument)
        assert type(self.first) is not type(self.second)
        assert self.first == self.second and hash(self.first) == hash(self.second)
        assert self.first.compare(self.second) == 0

    def time_structural_equality(self):
        Basic.is_same(self.first, self.second)

    def time_equality(self):
        self.first == self.second

    def time_comparison(self):
        self.first.compare(self.second)


class _NativeTree:
    def __init__(self, *children):
        self.children = children

    def __eq__(self, other):
        return isinstance(other, _NativeTree) and self.children == other.children

    def __hash__(self):
        return hash(self.children)

    @property
    def args(self):
        return list(self.children)

    @property
    def func(self):
        return type(self)


class TimeNativeReplace:
    def setup(self):
        self.old, self.new = Symbol('replace_y'), Symbol('replace_z')
        self.source = Basic(_NativeTree(Symbol('replace_x')), self.old)
        self.query = lambda node: node is self.old
        self.value = lambda node: self.new

    def time_replace(self):
        self.source.replace(self.query, self.value)


class TimeNativeSortKeys:
    def setup(self):
        self.frozenset_value = frozenset([1])
        self.scalar_value = _Label(0)
        self.frozensets = tuple(default_sort_key(Basic(value))
            for value in (frozenset([1]), frozenset([1, 2])))
        self.scalars = tuple(default_sort_key(Basic(_Label(value))) for value in (0, 1))

    def time_frozenset_comparison(self):
        self.frozensets[0] < self.frozensets[1]

    def time_scalar_comparison(self):
        self.scalars[0] < self.scalars[1]

    def time_frozenset_creation(self):
        default_sort_key(Basic(self.frozenset_value))

    def time_scalar_creation(self):
        default_sort_key(Basic(self.scalar_value))


class TimeEqualSizedFrozensets:
    def setup(self):
        self.first, self.second = (Basic(frozenset([value])) for value in (1, 2))
        self.keys = default_sort_key(self.first), default_sort_key(self.second)
        assert self.first != self.second
        assert self.first.compare(self.second) < 0 and self.keys[0] != self.keys[1]

    def time_equality(self):
        self.first == self.second

    def time_comparison(self):
        self.first.compare(self.second)

    def time_key_equality(self):
        self.keys[0] == self.keys[1]


    def time_cold_equality(self):
        Basic(frozenset([1])) == Basic(frozenset([2]))

    def time_cold_comparison(self):
        Basic(frozenset([1])).compare(Basic(frozenset([2])))


class TimeStructuralQueries:
    params = ['symbol', 'expression']
    param_names = ['kind']

    def setup(self, kind):
        self.pattern, other = Symbol('query_x'), Symbol('query_y')
        self.source = (self.pattern if kind == 'symbol' else
                       (sin(self.pattern) + sin(other))**2)
        assert self.source.has(self.pattern) and self.source.has_free(self.pattern)

    def time_has(self, kind):
        self.source.has(self.pattern)

    def time_has_free(self, kind):
        self.source.has_free(self.pattern)


class _ClassMeta(type):
    def __eq__(cls, other):
        return isinstance(other, _ClassMeta) and cls.label == other.label

    def __hash__(cls):
        return hash(cls.label)

    def __lt__(cls, other):
        return cls.label < other.label

    def __gt__(cls, other):
        return cls.label > other.label


class TimeClassLiteralKeys:
    def setup(self):
        self.symbol = default_sort_key(Symbol)
        first = _ClassMeta('Alpha', (), {'label': 1})
        second = _ClassMeta('Omega', (), {'label': 2})
        self.first = Basic(first)
        self.keys = default_sort_key(self.first), default_sort_key(Basic(second))

    def time_creation(self):
        default_sort_key(Symbol)

    def time_comparison(self):
        self.keys[0] < self.keys[1]

    def time_wrapped_creation(self):
        default_sort_key(self.first)


class _DeferredSubsExpr(Expr):
    _first = Symbol('deferred_first')
    _second = Symbol('deferred_second')

    def _eval_subs(self, old, new):
        if old == self._second and new == 0:
            return Subs(self, (self._second, self._first), (0, self._second))


class TimeDeferredSubs:
    params = [False, True]
    param_names = ['deep']

    def setup(self, deep):
        expr = _DeferredSubsExpr()
        first, second = expr._first, expr._second
        self.source = Subs(expr, (first, second), (second, 0))
        expected = Subs(expr, (second, first), (0, second))
        assert self.source.doit(deep=deep).args == expected.args

    def time_doit(self, deep):
        self.source.doit(deep=deep)


class TimeUnmatchedSubs:
    params = [[False, True], ['sum', 'power']]
    param_names = ['point_change', 'expression']

    def setup(self, point_change, expression):
        x, self.old = Symbol('unmatched_x'), Symbol('unmatched_y')
        expr = Function('unmatched_function')(x)
        expr = expr + 1 if expression == 'sum' else expr**2
        self.new = x
        self.source = Subs(expr, x, self.old if point_change else 0)
        expected = Subs(expr, x, x if point_change else 0)
        assert self.source._eval_subs(self.old, self.new).args == expected.args

    def time_eval_subs(self, point_change, expression):
        self.source._eval_subs(self.old, self.new)


class _Bases(tuple):
    __slots__ = ()


class _Namespace(dict):
    pass


class _EvaluatingMeta(UndefinedFunction):
    def __new__(mcl, *args, **kwargs):
        cls = super().__new__(mcl, *args, **kwargs)
        cls.eval = classmethod(lambda cls, arg: arg + 1)
        return cls


class TimeFunctionDeclarations:
    def setup(self):
        self.bases = _Bases((AppliedUndef,))
        self.namespace = {'eval': classmethod(lambda cls, arg: arg + 1)}
        assert UndefinedFunction('declared', bases=self.bases)(1).func.__name__ == 'declared'
        assert UndefinedFunction('declared', __dict__=_Namespace(self.namespace))(1) == 2
        assert _EvaluatingMeta('declared')(1) == 2

    def time_tuple_bases(self):
        UndefinedFunction('declared', bases=self.bases)

    def time_dict_namespace(self):
        UndefinedFunction('declared', __dict__=_Namespace(self.namespace))

    def time_custom_metaclass(self):
        _EvaluatingMeta('declared')


class TimePassiveFunctionSubs:
    params = [False, True]
    param_names = ['late_mixin']

    def setup(self, late_mixin):
        class Mixin:
            value = 1
        bases = (Function, Mixin) if late_mixin else (Mixin, Function)
        cls = type('Passive', bases, {})
        x, z = Symbol('passive_x'), Symbol('passive_z')
        self.expr, self.variable = cls(x), x
        first, second = Subs(self.expr, x, 0), Subs(cls(z), z, 0)
        assert first.doit() == second.doit() == cls(0)
        assert first == second and hash(first) == hash(second)

    def time_key(self, late_mixin):
        hash(Subs(self.expr, self.variable, 0))


class TimeFunctionApplications:
    params = ['undefined', 'defined', 'declared']
    param_names = ['kind']

    def setup(self, kind):
        self.argument = Symbol('application_argument')
        self.function = (Function('application_bench') if kind == 'undefined' else
                         sin if kind == 'defined' else type('Declared', (Function,), {}))
        self.value = self.function(self.argument)
        assert type(self.value) is self.function and self.value.args == (self.argument,)

    def time_construction(self, kind):
        self.function(self.argument)



class _PassiveMetaMeta(type):
    pass


class _OwnerMetaMeta(type):
    def __call__(mcl, *args, **kwargs):
        function = super().__call__(*args, **kwargs)
        function.eval = classmethod(lambda cls, arg: S.One if arg == 0 else None)
        return function


class _UnhashableMetaMeta(type):
    def __eq__(cls, other):
        return cls is other


class _PassiveFactoryMeta(UndefinedFunction, metaclass=_PassiveMetaMeta):
    pass


class _OwnerFactoryMeta(UndefinedFunction, metaclass=_OwnerMetaMeta):
    pass


class _UnhashableFactoryMeta(UndefinedFunction, metaclass=_UnhashableMetaMeta):
    pass


class _UnhashableClassMeta(UndefinedFunction):
    def __eq__(cls, other):
        return cls is other


class TimeFactoryMetaclassHierarchy:
    params = ['passive', 'owner', 'unhashable', 'unhashable_class']
    param_names = ['kind']

    def setup(self, kind):
        self.constructor = {'passive': _PassiveFactoryMeta,
                            'owner': _OwnerFactoryMeta,
                            'unhashable': _UnhashableFactoryMeta,
                            'unhashable_class': _UnhashableClassMeta}[kind]
        function = self.constructor('hierarchy_factory', value=1)
        if kind == 'owner':
            assert function(0) == 1
        else:
            assert function(0).func == function

    def time_construction(self, kind):
        self.constructor('hierarchy_factory', value=1)


class _ReportedBasic(Basic):
    @property
    def __class__(self):
        return Basic


class TimeReportedClassComparison:
    def setup(self):
        self.first, self.second = _ReportedBasic(1), _ReportedBasic(2)
        self.equal = _ReportedBasic(1)
        assert self.first.compare(self.second) == -1
        assert Basic.is_same(self.first, self.equal)

    def time_comparison(self):
        self.first.compare(self.second)

    def time_structural_equality(self):
        Basic.is_same(self.first, self.equal)


class TimeNativeTraversal:
    params = ['class', 'tuple']
    param_names = ['field']

    def setup(self, field):
        value = Symbol if field == 'class' else (1, 2)
        self.source = Basic(value, Symbol('traversal_x'))
        self.other = Basic(*self.source.args)
        size = 3 if field == 'class' else 5
        assert len(list(preorder_traversal(self.source))) == size
        assert len(list(postorder_traversal(self.source))) == size
        assert Basic.is_same(self.source, self.other)

    def time_preorder(self, field):
        list(preorder_traversal(self.source))

    def time_postorder(self, field):
        list(postorder_traversal(self.source))

    def time_structural_equality(self, field):
        Basic.is_same(self.source, self.other)


class TimeUnorderedStructure:
    params = [False, True]
    param_names = ['fresh_members']

    def setup(self, fresh_members):
        values = tuple(_Label(index) for index in range(8))
        aliases = tuple(_Label(index) for index in range(8)) if fresh_members else values
        self.first = Basic(frozenset(values))
        self.second = Basic(frozenset(aliases))
        assert self.first == self.second
        assert Basic.is_same(self.first, self.second)

    def time_structural_equality(self, fresh_members):
        Basic.is_same(self.first, self.second)


class TimeDeepStructure:
    params = ['ordered_equal', 'ordered_unequal', 'set_equal', 'set_unequal', 'shared']
    param_names = ['kind']

    def setup(self, kind):
        expected = kind in ('ordered_equal', 'set_equal', 'shared')
        if kind == 'shared':
            left, right = Basic(S.One), Basic(S.One)
            for _ in range(10):
                left, right = Basic(left, left), Basic(right, right)
        else:
            sets = kind.startswith('set')
            left = -1 if sets else Symbol('deep_structure_left')
            right = left if expected else -2 if sets else Symbol('deep_structure_right')
            for _ in range(100):
                left, right = ((frozenset([left]), frozenset([right]))
                               if sets else (Basic(left), Basic(right)))
        self.first, self.second = left, right
        assert Basic.is_same(left, right) is expected

    def time_structural_equality(self, kind):
        Basic.is_same(self.first, self.second)


class _IterableField:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        return isinstance(other, _IterableField) and self.values == other.values


class TimeIterableStructure:
    params = [['list', 'iterable'], [False, True]]
    param_names = ['container', 'equal']

    def setup(self, container, equal):
        first = list(range(256))
        second = first.copy()
        if not equal:
            second[0] = -1
        wrap = list if container == 'list' else _IterableField
        self.first, self.second = wrap(first), wrap(second)
        assert Basic.is_same(self.first, self.second) is equal

    def time_structural_equality(self, container, equal):
        Basic.is_same(self.first, self.second)


class _ReverseTuple(tuple):
    __slots__ = ()
    __lt__ = tuple.__gt__
    __gt__ = tuple.__lt__


class TimeTupleOrdering:
    params = [False, True]
    param_names = ['mixed']

    def setup(self, mixed):
        fields = ((_ReverseTuple((2,)), (1,)) if mixed else
                  (_ReverseTuple((1,)), _ReverseTuple((2,))))
        self.first, self.second = map(Basic, fields)
        self.left, self.right = map(default_sort_key, (self.first, self.second))
        expected = -1 if mixed else 1
        assert self.first.compare(self.second) == expected
        assert (self.left > self.right) == (expected > 0)
        self.metadata = (1,) if mixed else _ReverseTuple((1,))
        self.factory = Function('tuple_order_metadata', value=self.metadata)
        assert self.factory == Function('tuple_order_metadata', value=self.metadata)

    def time_comparison(self, mixed):
        self.first.compare(self.second)

    def time_key_comparison(self, mixed):
        self.left > self.right

    def time_metadata_factory(self, mixed):
        Function('tuple_order_metadata', value=self.metadata)
