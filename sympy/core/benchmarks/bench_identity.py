r"""ASV cases for argument identity, metadata, ordering, and substitution.

Reproduce a comparison with sympy/sympy_benchmarks at commit
3a6f8897c0837c628a0cdb8a20663638b5de53ed, using ASV 0.6.6 and CPython 3.14.4.
Keep the SymPy checkout at ../sympy relative to the corpus and copy this file
to benchmarks/review_native_fields.py. Save asv.pr3.json in the corpus:

    {
      "version": 1, "project": "sympy", "repo": "../sympy",
      "environment_type": "virtualenv", "pythons": ["3.14"],
      "matrix": {"mpmath": ["1.4.1"], "numpy": ["2.5.4"]},
      "build_command": ["python -m pip install build",
                        "python -m build --wheel -o {build_cache_dir} {build_dir}"]
    }

Run ``asv machine --yes --config asv.pr3.json``. With python3.14 on PATH,
substitute the master and PR commit hashes from the PR body for MASTER and PR:

    PYTHONHASHSEED=0 SYMPY_USE_CACHE=yes SYMPY_CACHE_SIZE=1000 \
      asv continuous --config asv.pr3.json --cpu-affinity 8 \
      --attribute rounds=4 --attribute repeat=5 --interleave-rounds \
      --factor 1.1 --only-changed --bench 'SELECTOR' MASTER PR

Choose a local CPU for --cpu-affinity. The PR body provides the full and
confirmation selectors; confirmation uses rounds=6 and repeat=10.
Report a workload only when ASV flags a change above factor 1.1 in the same
direction in both runs, with all four Python controls unflagged.
Workloads perform setup and correctness assertions outside the timed methods.
"""

from __future__ import annotations

from sympy import (Basic, Derivative, Dummy, Expr, Float, Function, Lambda, Rational, S, Subs,
                   Symbol, Tuple, default_sort_key, sin)
from sympy.core import cache
from sympy.core.cache import USE_CACHE, cacheit, clear_cache
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.traversal import iterargs, iterfreeargs, preorder_traversal, postorder_traversal


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


class _TupleLabel(_Label):
    def __lt__(self, other):
        if isinstance(other, tuple):
            return self.label < tuple.__getitem__(other, 0)
        return super().__lt__(other)

    def __gt__(self, other):
        if isinstance(other, tuple):
            return self.label > tuple.__getitem__(other, 0)
        return super().__gt__(other)


class _TupleLabelTuple(_TupleLabel, tuple):
    __slots__ = ()

    def __new__(cls, label):
        return tuple.__new__(cls, (label,))


class _TupleLabelSet(_TupleLabel, frozenset):
    def __new__(cls, label):
        return frozenset.__new__(cls, (label,))


class TimeMixedNativeFields:
    params = ['scalar', 'tuple', 'frozenset']
    param_names = ['other']

    def setup(self, other):
        cls = {'scalar': _TupleLabel, 'tuple': _TupleLabelTuple,
               'frozenset': _TupleLabelSet}[other]
        self.first, self.second = Basic((1,)), Basic(cls(1.5))
        self.left, self.right = map(default_sort_key, (self.first, self.second))
        assert self.first.compare(self.second) == -1
        assert self.left < self.right

    def time_comparison(self, other):
        self.first.compare(self.second)

    def time_key_comparison(self, other):
        self.left < self.right


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


class TimeReplacements:
    params = ['predicate', 'type', 'expression']
    param_names = ['query']

    def setup(self, query):
        if query == 'expression':
            original, replacement = Symbol('replace_x'), Symbol('replace_y')
            self.query, self.value = original, replacement
        else:
            class Literal:
                args = ()

            class Replacement:
                args = ()

            original, replacement = Literal, Replacement
            self.query = (lambda node: node is original) if query == 'predicate' else type
            self.value = (lambda node: replacement) if query == 'predicate' else (lambda: replacement)

        self.source = Basic(original)
        result = self.source.replace(self.query, self.value)
        assert result.args == (replacement,)

    def time_replace(self, query):
        self.source.replace(self.query, self.value)


class TimeCacheFallback:
    params = [['value', 'identity'], ['hashable', 'receiver', 'argument']]
    param_names = ['cache', 'unhashable']

    def setup(self, cache_kind, unhashable):
        decorate = cacheit if cache_kind == 'value' else getattr(cache, 'cacheit_by_identity', cacheit)

        @decorate
        def length(receiver, values):
            return len(values)

        assert USE_CACHE == 'yes' and length.cache_info().maxsize != 0
        self.function = length
        self.receiver, self.values = {'hashable': (object(), ()),
                                     'receiver': ([], ()),
                                     'argument': (object(), [])}[unhashable]
        assert self.function(self.receiver, self.values) == 0

    def time_call(self, cache_kind, unhashable):
        self.function(self.receiver, self.values)


class _ClassMeta(type):
    def __eq__(cls, other):
        return isinstance(other, _ClassMeta) and cls.label == other.label

    def __hash__(cls):
        return hash(cls.label)

    def __lt__(cls, other):
        return cls.label < other.label

    def __gt__(cls, other):
        return cls.label > other.label


class _QueryPattern(Basic):
    def __eq__(self, other):
        return isinstance(other, _QueryPattern)

    def __hash__(self):
        return 0

    def _has_matcher(self):
        return lambda node: False


class TimeMultipleBindingScope:
    params = ['sin', 'noncommutative', 'inactive', 'postprocessor',
              'postprocessor_compound', 'postprocessor_inactive', 'postprocessor_definition']
    param_names = ['kind']

    def setup(self, kind):
        x, y, z = map(Symbol, ('scope_x', 'scope_y', 'scope_z'))
        self.old, self.new = z, x
        if kind == 'sin':
            self.source = Subs(sin(x + y) + z, (x, y), (y, 1))
        elif kind.startswith('postprocessor'):
            if kind == 'postprocessor_definition':
                function = Function('scope_function')
                self.source = Subs(function(x), (x, y), (1, 0))
                self.old, self.new = function, Lambda(z, x*z + y)
            elif kind == 'postprocessor_inactive':
                self.source = Subs(x*z, (x, y), (1, 0))
                self.new = x + y
            else:
                self.source = Subs(x*z, (x, Dummy('scope_unused')), (1, 0))
                if kind == 'postprocessor_compound':
                    self.new = x + 1
            mapping = Basic._constructor_postprocessor_mapping
            self._had_dummy_processors = Dummy in mapping
            self._dummy_processors = mapping.get(Dummy)
            mapping[Dummy] = {'Mul': [lambda expr: S.Zero]}
            clear_cache()
        else:
            function = Function('scope_function')
            point = Symbol('scope_A', commutative=False)
            if kind == 'noncommutative':
                self.source = Subs(function(x, y) + z, (x, y), (point, 1))
            else:
                self.source = Subs(function(x) + z, (x, y), (0, point))
        assert isinstance(self.source._eval_subs(self.old, self.new), Subs)

    def time_eval_subs(self, kind):
        self.source._eval_subs(self.old, self.new)

    def teardown(self, kind):
        if kind.startswith('postprocessor'):
            mapping = Basic._constructor_postprocessor_mapping
            if self._had_dummy_processors:
                mapping[Dummy] = self._dummy_processors
            else:
                del mapping[Dummy]
            clear_cache()


class TimeMultipleValuePatterns:
    params = [[False, True], [False, True]]
    param_names = ['mixed_types', 'free']

    def setup(self, mixed_types, free):
        classes = ([_ClassMeta('PatternAlias', (_QueryPattern,), {'label': 0})
                    for _ in range(2)] if mixed_types else [_QueryPattern]*2)
        self.patterns = tuple(cls() for cls in classes)
        self.source = Basic(Symbol('multiple_pattern_query_x'))
        self.traversal = iterfreeargs if free else iterargs
        assert not self.source._has(self.traversal, *self.patterns)

    def time_has(self, mixed_types, free):
        self.source._has(self.traversal, *self.patterns)


class TimeReceiverQueryCache:
    params = [False, True]
    param_names = ['custom_metaclass']

    def setup(self, custom_metaclass):
        cls = (_ClassMeta('QueryNode', (Basic,), {'label': 0})
               if custom_metaclass else Basic)
        self.pattern = Symbol('receiver_query_x')
        self.source = cls(self.pattern)
        assert self.source.has(self.pattern)
        assert self.source.has_free(self.pattern)
        assert self.source.sort_key() == self.source.sort_key()

    def time_has(self, custom_metaclass):
        self.source.has(self.pattern)

    def time_has_free(self, custom_metaclass):
        self.source.has_free(self.pattern)

    def time_sort_key(self, custom_metaclass):
        self.source.sort_key()


class TimeClassPatternQueries:
    params = [False, True]
    param_names = ['custom_metaclass']

    def setup(self, custom_metaclass):
        cls = (_ClassMeta('PatternNode', (Basic,), {'label': 0})
               if custom_metaclass else Basic)
        self.pattern = cls
        self.source = cls(Symbol('class_pattern_query_x'))
        assert self.source.has(self.pattern)
        assert self.source.has_free(self.pattern)

    def time_has(self, custom_metaclass):
        self.source.has(self.pattern)

    def time_has_free(self, custom_metaclass):
        self.source.has_free(self.pattern)


class TimeClassLiteralFreeQueries:
    params = [['native', 'sympy'], [False, True]]
    param_names = ['pattern', 'cached']

    def setup(self, pattern, cached):
        self.pattern = int if pattern == 'native' else Basic
        self.source = Basic(int) if pattern == 'native' else Basic()
        query = Basic.has_free if cached else Basic.has_free.__wrapped__
        self.query = lambda: query(self.source, self.pattern)
        assert self.query()

    def time_has_free(self, pattern, cached):
        self.query()


class TimeReportedClassData:
    params = [False, True]
    param_names = ['reports_type']

    def setup(self, reports_type):
        reported = type if reports_type else Basic
        cls = type('ReportedNode', (Basic,), {
            '__class__': property(lambda self: reported),
            'func': property(lambda self: type(self)),
        })
        self.x, self.y = Symbol('reported_data_x'), Symbol('reported_data_y')
        self.node = cls(self.x)
        self.source = Basic(self.node)
        self.rule = {self.x: self.y}
        expected = Basic(cls(self.y))
        assert self.source.free_symbols == {self.x}
        assert self.source.has_free(self.node)
        assert self.source.xreplace(self.rule) == expected
        assert self.source.replace(self.x, self.y) == expected

    def time_free_symbols(self, reports_type):
        self.source.free_symbols

    def time_has_free(self, reports_type):
        self.source.has_free(self.node)

    def time_xreplace(self, reports_type):
        self.source.xreplace(self.rule)

    def time_replace(self, reports_type):
        self.source.replace(self.x, self.y)


class TimeClassPatternAliases:
    def setup(self):
        self.source = Basic(Symbol('class_alias_query_x'))
        self.patterns = [_ClassMeta('PatternAlias', (Basic,), {'label': 0})
                         for _ in range(128)]
        assert all(not self.source.has(pattern) and not self.source.has_free(pattern)
                   for pattern in self.patterns)

    def time_has_aliases(self):
        for pattern in self.patterns:
            self.source.has(pattern)

    def time_has_free_aliases(self):
        for pattern in self.patterns:
            self.source.has_free(pattern)


_aliased_rationals = tuple(_ClassMeta('PassiveRational', (Rational,),
    {'label': 0, '__module__': __name__}) for _ in range(2))


class TimeAtomicHookCache:
    params = [False, True]
    param_names = ['alias_classes']

    def setup(self, alias_classes):
        classes = _aliased_rationals if alias_classes else (Rational, Rational)
        first, second = classes[0](1, 3), classes[1](2, 5)
        x, self.old = Symbol('hook_cache_x'), Symbol('hook_cache_y')
        self.new = x
        function = Function('hook_cache_function')
        primer = Subs(Tuple(function(x), first), x, 0)
        self.source = Subs(Tuple(function(x), second), x, 0)
        primer._eval_subs(self.old, self.new)
        assert self.source._eval_subs(self.old, self.new).doit() == Tuple(function(0), second)

    def time_eval_subs(self, alias_classes):
        self.source._eval_subs(self.old, self.new)


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


class TimeFunctionDefinition:
    params = ['scalar', 'derivative', 'unhashable']
    param_names = ['kind']

    def setup(self, kind):
        variable = Symbol('definition_variable')
        self.function = Function('definition_function')
        metaclass = (type('DefinitionMeta', (type(Function),), {'__hash__': None})
                     if kind == 'unhashable' else type(Function))

        class Square(Function, metaclass=metaclass):
            @classmethod
            def eval(cls, arg):
                if isinstance(arg, Dummy):
                    raise ValueError('unexpected dummy argument')
                return arg**2

        self.definition = Square
        body = self.function(variable)
        if kind == 'derivative':
            body = Derivative(body, variable)
        self.source = Subs(body, variable, 1)
        expected = 2 if kind == 'derivative' else 1
        assert self.source._eval_subs(self.function, self.definition).doit() == expected

    def time_definition(self, kind):
        self.source._eval_subs(self.function, self.definition).doit()


class TimeBindingQueries:
    def setup(self):
        variable = Symbol('binding_variable')
        self.missing = Symbol('absent_variable')
        function = Function('binding_function')
        self.source = Subs(function(variable), variable, 0)
        assert self.source.as_dummy().doit() == function(0)
        assert self.source.canonical_variables == {variable: Symbol('_0')}
        assert not self.source._has(iterfreeargs, self.missing)

    def time_as_dummy(self):
        self.source.as_dummy()

    def time_canonical_variables(self):
        self.source.canonical_variables

    def time_has_free(self):
        self.source._has(iterfreeargs, self.missing)
