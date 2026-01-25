# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Binary expressions for arithmetic, logic, and comparison operations.

## Syntax

It's recommended to employ static type analysis tools like `mypy` to provide
realtime feedback on the correctness of your expressions as you work.

### Mapping Context Variables <-> `Var`s

The rule is that the `str` name of the `Var`, it's `name` attribute, must match
the corresponding field name in the context object. For example, this will work:
>>> from typing import NamedTuple
>>> class Context(NamedTuple):
... 	my_name: str
>>> any_name_works_here = Var[int, Context]("my_name")
>>> any_name_works_here.to_string()
'my_name'
>>> any_name_works_here.to_string(Context(my_name='jp'))
'my_name:jp'

But this will not, when it's evaluated:
>>> wrong_name = Var[int, Context]("wrong_name")
>>> wrong_name.to_string()
'wrong_name'
>>> wrong_name.to_string(Context(my_name='jp'))
Traceback (most recent call last):
	...
mahonia.EvalError: Variable 'wrong_name' not found in context

This occurs because when `wrong_name` is evaluated, it looks for its' value at
the `wrong_name` attribute of the `Context` object, which does not exist.

Note that `Var`s are tied to their context type, so you cannot use a `Var` from
one context in another context and will get a static type error if you try.

### Type Coercion

Coercible types (like `int`, `float`, `bool`) can appear on either side of
a binary operator with a Mahonia `Expr`. The coercible type will be wrapped
in an unnamed `Const`.

>>> X = Const("X", 41)
>>> X + 1
Add(left=Const(name='X', value=41), right=Const(name=None, value=1))

>>> 1 + X
Add(left=Const(name=None, value=1), right=Const(name='X', value=41))

This works because `BinaryOperationOverloads` implements both the standard
operators (`__add__`, etc.) and their right-associative counterparts
(`__radd__`, etc.). When Python's `int.__add__` doesn't know how to handle
a Mahonia `Expr`, it falls back to the Expr's `__radd__` method.

Comparison operators also work with literals on the left, but Python reflects
them to the swapped operator (e.g., `5 < x` becomes `x > 5`):

>>> from typing import Any
>>> x = Var[int, Any]("x")
>>> 5 < x
Gt(left=Var(name='x'), right=Const(name=None, value=5))

For `==` and `!=`, static type checkers incorrectly infer `bool` because they
don't know that `int.__eq__` returns `NotImplemented` for unknown types. At
runtime, Python correctly falls back to the Expr's `__eq__` method. To avoid
type checker errors, prefer placing the Expr on the left side: `x == 5`.

### Examples

For more exhaustive examples, see the tests.

>>> from typing import NamedTuple, assert_type
...
>>> class Ctx(NamedTuple):
... 	x: int
... 	y: int
...
>>> x = Var[int, Ctx]("x")
>>> y = Var[int, Ctx]("y")
>>> MAX = Const("Max", 10)
>>> x
Var(name='x')
>>> MAX
Const(name='Max', value=10)
>>> # Create an expression that compares a Var and a Const
>>> x < MAX
Lt(left=Var(name='x'), right=Const(name='Max', value=10))
>>> # Serialize
>>> (x < MAX).to_string()
'(x < Max:10)'
>>> # Evaluate
>>> (x < MAX).unwrap(Ctx(x=5, y=10))
True
>>> # Serialize the evaluation
>>> (x < MAX).to_string(Ctx(x=5, y=10))
'(x:5 < Max:10 -> True)'
>>> # Assign an expression to a variable
>>> sum_expr = x + y
>>> sum_expr
Add(left=Var(name='x'), right=Var(name='y'))
>>> # Serialize the expression to a string
>>> sum_expr.to_string()
'(x + y)'
>>> # Evaluate the expression with concrete values
>>> sum_result = sum_expr(Ctx(x=1, y=2))
>>> sum_result
Const(name=None, value=3)
>>> # Serialize the evaluated expression to a string
>>> sum_expr.to_string(Ctx(x=1, y=2))
'(x:1 + y:2 -> 3)'
>>> # Access the value of the result
>>> sum_result.value
3
>>> # Shorthand for getting the value
>>> sum_expr.unwrap(Ctx(x=1, y=2))
3
>>> # Bind an expression to a context
>>> summed = sum_expr.bind(Ctx(x=1, y=2))
>>> str(summed)
'(x:1 + y:2 -> 3)'
>>> summed.unwrap()
3
>>> # Combine expressions
>>> zero = x + y - x - y
>>> zero.to_string()
'(((x + y) - x) - y)'
>>> zero.to_string(Ctx(x=1, y=2))
'(((x:1 + y:2 -> 3) - x:1 -> 2) - y:2 -> 0)'
>>> zero.unwrap(Ctx(x=1, y=2))
0
>>> # Compose expressions
>>> four = sum_expr + x
>>> four.to_string()
'((x + y) + x)'
>>> four.to_string(Ctx(x=1, y=2))
'((x:1 + y:2 -> 3) + x:1 -> 4)'
>>> four.unwrap(Ctx(x=1, y=2))
4
>>> # Create a predicate
>>> is_valid = (x > 0) & (y < MAX)
>>> is_valid.to_string()
'((x > 0) & (y < Max:10))'
>>> is_valid.to_string(Ctx(x=1, y=2))
'((x:1 > 0 -> True) & (y:2 < Max:10 -> True) -> True)'
>>> is_valid.unwrap(Ctx(x=1, y=2))
True
>>> # Define the predicate
>>> cate = Predicate("x is pos, y less than Max", is_valid)
>>> cate.to_string()
'x is pos, y less than Max: ((x > 0) & (y < Max:10))'
>>> cate.to_string(Ctx(x=1, y=2))
'x is pos, y less than Max: True ((x:1 > 0 -> True) & (y:2 < Max:10 -> True) -> True)'
>>> cate.unwrap(Ctx(x=1, y=2))
True
>>> # Other abstractions similar to Predicate include
>>> # Approximately and PlusMinus. PlusMinus is a kind
>>> # of Const that implements the ConstToleranceProtocol.
>>> # These can be use as reference for custom convenience
>>> # abstractions.
>>> class Ctx(NamedTuple):
... 	voltage: float
>>> voltage = Var[float, Ctx]("voltage")
>>> voltage_check = Approximately(voltage, PlusMinus("V", 5.0, 0.1))
>>> voltage_check.to_string(Ctx(voltage=5.05))
'(voltage:5.05 ≈ V:5.0 ± 0.1 -> True)'
"""

import operator
from dataclasses import dataclass, field
from difflib import get_close_matches
from types import SimpleNamespace
from typing import (
	TYPE_CHECKING,
	Any,
	Callable,
	ClassVar,
	Final,
	Generic,
	Iterable,
	Protocol,
	Self,
	Sized,
	TypeVar,
	TypeVarTuple,
	overload,
	runtime_checkable,
)

if TYPE_CHECKING:
	from mahonia.match import Match, MatchExpr

T = TypeVar("T")
"""The type of the expression's operands."""

U = TypeVar("U")
"""A generic type parameter for mapped values."""

R = TypeVar("R")
"""The type returned by eval() and unwrap()."""


class ContextProtocol(Protocol):
	def __getattribute__(self, name: str, /) -> Any: ...


S = TypeVar("S", bound=ContextProtocol)
"""The type of the expression's context."""

T_co = TypeVar("T_co", covariant=True)


Ss = TypeVarTuple("Ss")


class MergeContextProtocol(Protocol[*Ss]):
	def __getattribute__(self, name: str, /) -> Any: ...


def merge[*Ss](*contexts: *Ss) -> MergeContextProtocol[*Ss]:
	"""Merge values from multiple context instances into a single namespace.

	>>> from typing import NamedTuple
	>>> class A(NamedTuple):
	... 	a: int
	>>> class B(NamedTuple):
	... 	b: int
	>>> ctx = merge(A(a=1), B(b=2))
	>>> ctx.a
	1
	>>> ctx.b
	2
	"""

	values: dict[str, Any] = {}
	for context in contexts:
		# https://typing.python.org/en/latest/spec/generics.html#variance-type-constraints-and-type-bounds-not-yet-supported
		annotations: dict[str, Any] = getattr(type(context), "__annotations__")
		for field_name in annotations:
			if field_name in values:
				raise TypeError(f"Duplicate field '{field_name}' found when merging contexts.")
			values[field_name] = getattr(context, field_name)
	return SimpleNamespace(**values)


S_contra = TypeVar("S_contra", bound=ContextProtocol, contravariant=True)
"""The contravariant type of the expression's context."""


class SizedIterable(Sized, Iterable[T_co], Protocol[T_co]):
	"""Protocol for containers that are both sized and iterable with indexing support."""

	def __getitem__(self, index: int, /) -> T_co: ...


class _SupportsArithmetic(Protocol):
	def __add__(self, other: Any, /) -> Any: ...
	def __sub__(self, other: Any, /) -> Any: ...
	def __mul__(self, other: Any, /) -> Any: ...
	def __truediv__(self, other: Any, /) -> Any: ...
	def __mod__(self, other: Any, /) -> Any: ...
	def __abs__(self, /) -> Any: ...
	def __pow__(self, power: Any, /) -> Any: ...


TSupportsArithmetic = TypeVar("TSupportsArithmetic", bound=_SupportsArithmetic)


class _SupportsEquality(Protocol):
	def __eq__(self, other: Any, /) -> bool: ...
	def __ne__(self, other: Any, /) -> bool: ...


TSupportsEquality = TypeVar("TSupportsEquality", bound=_SupportsEquality)


class _SupportsComparison(Protocol):
	def __lt__(self, other: Any, /) -> bool: ...
	def __le__(self, other: Any, /) -> bool: ...
	def __gt__(self, other: Any, /) -> bool: ...
	def __ge__(self, other: Any, /) -> bool: ...


TSupportsComparison = TypeVar("TSupportsComparison", bound=_SupportsComparison)


class _SupportsLogic(Protocol):
	def __and__(self, other: Any, /) -> Any: ...
	def __or__(self, other: Any, /) -> Any: ...
	def __invert__(self, /) -> Any: ...


TSupportsLogic = TypeVar("TSupportsLogic", bound=_SupportsLogic)


class EvalError(Exception):
	"""Raised when variable evaluation fails due to missing context attributes."""

	pass


R_Eval = TypeVar("R_Eval")
"""Result type for the Eval protocol (no default, since T is not in scope)."""


@runtime_checkable
class Eval(Protocol[R_Eval, S_contra]):
	"""Protocol for objects that can evaluate themselves with a context.

	This protocol defines the interface for expression evaluation. All Mahonia
	expressions implement this protocol. R_Eval is the result type returned by eval().

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> isinstance(x, Eval)
	True
	>>> result = x.eval(Ctx(x=42))
	>>> result.value
	42
	"""

	def eval(self, ctx: S_contra) -> "Const[R_Eval]": ...


@runtime_checkable
class ToString(Protocol[S_contra]):
	"""Protocol for objects that can serialize themselves to strings.

	This protocol defines the interface for string serialization, with optional
	context for showing evaluated values.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> isinstance(x, ToString)
	True
	>>> x.to_string()
	'x'
	>>> x.to_string(Ctx(x=42))
	'x:42'
	"""

	def to_string(self, ctx: S_contra | None = None) -> str: ...


@runtime_checkable
class Expr(Protocol[T, S, R]):
	"""Base class for all Mahonia expressions.

	Provides core evaluation, serialization, and binding functionality.
	Users typically work with concrete expression types like Var, Const, Add, etc.

	Type parameters:
	- T: The operand type (for binary operations, the type of left/right)
	- S: The context type (bound by ContextProtocol)
	- R: The result type from eval() and unwrap(). Defaults to T for backward compat.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> isinstance(x, Expr)
	True
	>>> x.unwrap(Ctx(x=10))
	10
	>>> bound = x.bind(Ctx(x=10))
	>>> bound.unwrap()
	10
	"""

	def eval(self, ctx: S) -> "Const[R]": ...

	def to_string(self, ctx: S | None = None) -> str: ...

	def __call__(self, ctx: S) -> "Const[R]":
		return self.eval(ctx)

	def to_func(self) -> "Func[R, S]":
		return Func(_extract_vars((), self), self)

	def map(self, container: "Expr[SizedIterable[Any], Any, Any]") -> "MapExpr[Any, R, Any]":
		"""Apply this expression as a function to each element in a container.

		This is shorthand for: self.to_func() -> MapExpr(func, container)

		>>> from typing import NamedTuple
		>>> class ElemCtx(NamedTuple):
		... 	x: int
		>>> class ContainerCtx(NamedTuple):
		... 	numbers: list[int]
		>>> x = Var[int, ElemCtx]("x")
		>>> numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
		>>> mapped = (x * 2).map(numbers)
		>>> mapped.to_string()
		'(map x -> (x * 2) numbers)'
		"""
		func = self.to_func()
		return MapExpr(func, container)

	def unwrap(self, ctx: S) -> R:
		return self.eval(ctx).value

	def bind(self, ctx: S) -> "BoundExpr[T, S, R]":
		return BoundExpr(self, ctx)

	def partial(self, ctx: Any) -> "Expr[T, Any, R]":
		"""Partially apply a context, resolving variables that exist in it.

		Variables whose names exist as attributes in ctx are replaced with Const.
		Variables not found in ctx remain as Var. The expression structure is
		preserved (no eager evaluation of operations).

		>>> from typing import NamedTuple, Any
		>>> class XCtx(NamedTuple):
		... 	x: int
		>>> class YCtx(NamedTuple):
		... 	y: int
		>>> x = Var[int, Any]("x")
		>>> y = Var[int, Any]("y")
		>>> expr = x + y
		>>> partial_expr = expr.partial(XCtx(x=5))
		>>> partial_expr.to_string()
		'(x:5 + y)'
		>>> partial_expr.unwrap(YCtx(y=10))
		15
		>>> partial_expr.to_string(YCtx(y=10))
		'(x:5 + y:10 -> 15)'
		"""
		...


def _extract_vars(
	vars: tuple["Var[Any, Any]", ...], expr: "Expr[Any, Any, Any]"
) -> tuple["Var[Any, Any]", ...]:
	"""Extract all unique variables from an expression, preserving order.

	Note: pyright ignores on pattern match cases below are necessary because pattern
	matching on generic dataclasses loses type parameter information. This is a known
	pyright limitation with structural pattern matching on Generic types.
	"""
	match expr:
		case Var() as v:
			if id(v) not in (id(var) for var in vars):
				vars += (v,)
		case BinaryOp(left, right):
			vars = _extract_vars(vars, left)
			vars = _extract_vars(vars, right)
		case UnaryOpEval(left=left):  # pyright: ignore[reportUnnecessaryComparison, reportUnknownVariableType]
			vars = _extract_vars(vars, left)  # pyright: ignore[reportUnknownArgumentType]
		case Contains(element=element, container=container):  # pyright: ignore[reportUnknownVariableType]
			vars = _extract_vars(vars, element)  # pyright: ignore[reportUnknownArgumentType]
			vars = _extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case _ if hasattr(expr, "branches") and hasattr(expr, "default"):
			for condition, value in expr.branches:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
				vars = _extract_vars(vars, condition)  # pyright: ignore[reportUnknownArgumentType]
				vars = _extract_vars(vars, value)  # pyright: ignore[reportUnknownArgumentType]
			vars = _extract_vars(vars, expr.default)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
		case (
			AnyExpr(container)
			| AllExpr(container)
			| MinExpr(container)
			| MaxExpr(container)
			| FoldLExpr(container=container)  # pyright: ignore[reportUnknownVariableType]
		):
			vars = _extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case MapExpr(func=func, container=container):  # pyright: ignore[reportUnknownVariableType]
			for arg in func.args:
				if isinstance(arg, Var) and id(arg) not in (id(var) for var in vars):
					vars += (arg,)
			vars = _extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case FilterExpr(predicate=predicate, container=container):  # pyright: ignore[reportUnknownVariableType]
			for arg in predicate.args:
				if isinstance(arg, Var) and id(arg) not in (id(var) for var in vars):
					vars += (arg,)
			vars = _extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case _:
			pass
	return vars


@dataclass(frozen=True, eq=False, slots=True)
class Func(Generic[T, S]):
	args: tuple[Expr[Any, S, Any], ...]
	expr: Expr[Any, S, T]

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			if len(self.args) == 0:
				return f"() -> {self.expr.to_string()}"
			elif len(self.args) == 1:
				return f"{self.args[0].to_string()} -> {self.expr.to_string()}"
			else:
				args_str = ", ".join(arg.to_string() for arg in self.args)
				return f"({args_str}) -> {self.expr.to_string()}"
		else:
			if len(self.args) == 0:
				return f"() -> {self.expr.to_string()} -> {self.expr.unwrap(ctx)}"
			elif len(self.args) == 1:
				return f"{self.args[0].unwrap(ctx)} -> {self.expr.to_string()} -> {self.expr.unwrap(ctx)}"
			else:
				args_str = ", ".join(f"{arg.unwrap(ctx)}" for arg in self.args)
				return f"({args_str}) -> {self.expr.to_string()} -> {self.expr.unwrap(ctx)}"


@runtime_checkable
class BoolExpr(Expr[T, S, bool], Protocol[T, S]):
	"""Base class for boolean expressions that support logical operations.

	Extends Expr with support for logical operators like & (and), | (or), and ~ (not).
	T is the operand type, S is the context type, and the result is always bool.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> bool_expr = x > 5
	>>> isinstance(bool_expr, BoolExpr)
	True
	>>> bool_expr.unwrap(Ctx(x=10))
	True
	>>> combined = bool_expr & (x < 20)
	>>> combined.unwrap(Ctx(x=10))
	True
	"""

	def eval(self, ctx: S) -> "Const[bool]": ...


class UnaryOperationOverloads(Expr[bool, S, bool]):
	def __invert__(self) -> "Not[S]":
		return Not(self)


class BooleanBinaryOperationOverloads(Generic[T, S]):
	def __and__(self, other: "Expr[Any, S, Any]") -> "And[bool, S]":
		return And(self, other)  # type: ignore[arg-type]

	def __rand__(self, other: bool) -> "And[bool, S]":
		return And(Const(None, other), self)  # type: ignore[arg-type]

	def __or__(self, other: "Expr[Any, S, Any]") -> "Or[bool, S]":
		return Or(self, other)  # type: ignore[arg-type]

	def __ror__(self, other: bool) -> "Or[bool, S]":
		return Or(Const(None, other), self)  # type: ignore[arg-type]

	def __invert__(self) -> "Not[S]":
		return Not(self)  # type: ignore[arg-type]


class BinaryOperationOverloads(Expr[T, S, T]):
	@overload  # type: ignore[override]
	def __eq__(
		self, other: "ConstTolerance[TSupportsArithmetic]"
	) -> "Approximately[TSupportsArithmetic, S]": ...

	@overload
	def __eq__(
		self, other: Expr[TSupportsEquality, S, TSupportsEquality]
	) -> "Eq[TSupportsEquality, S]": ...

	@overload
	def __eq__(self, other: TSupportsEquality) -> "Eq[TSupportsEquality, S]": ...

	def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
		self,
		other: Expr[TSupportsEquality, S, TSupportsEquality]
		| TSupportsEquality
		| "ConstTolerance[TSupportsArithmetic]",
	) -> "Eq[TSupportsEquality, S] | Approximately[TSupportsArithmetic, S]":
		if isinstance(other, ConstTolerance):
			return Approximately(self, other)  # type: ignore[arg-type]
		elif isinstance(other, Expr):
			return Eq(self, other)  # type: ignore[arg-type]
		else:
			return Eq(self, Const(None, other))  # type: ignore[arg-type]

	@overload  # type: ignore[override]
	def __ne__(self, other: Expr[T, S, T]) -> "Ne[T, S]": ...

	@overload
	def __ne__(self, other: T) -> "Ne[T, S]": ...

	def __ne__(self, other: Expr[T, S, T] | T) -> "Ne[T, S]":  # pyright: ignore[reportIncompatibleMethodOverride]
		if isinstance(other, Expr):
			return Ne(self, other)  # pyright: ignore[reportUnknownArgumentType]
		else:
			return Ne(self, Const[T](None, other))

	@overload
	def __lt__(self, other: TSupportsComparison) -> "Lt[TSupportsComparison, S]": ...

	@overload
	def __lt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "Lt[TSupportsComparison, S]": ...

	def __lt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison] | TSupportsComparison
	) -> "Lt[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Lt(self, other)  # type: ignore[arg-type]
		else:
			return Lt(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __le__(self, other: TSupportsComparison) -> "Le[TSupportsComparison, S]": ...

	@overload
	def __le__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "Le[TSupportsComparison, S]": ...

	def __le__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison] | TSupportsComparison
	) -> "Le[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Le(self, other)  # type: ignore[arg-type]
		else:
			return Le(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __gt__(self, other: TSupportsComparison) -> "Gt[TSupportsComparison,S]": ...

	@overload
	def __gt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "Gt[TSupportsComparison, S]": ...

	def __gt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison] | TSupportsComparison
	) -> "Gt[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Gt(self, other)  # type: ignore[arg-type]
		else:
			return Gt(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __ge__(self, other: TSupportsComparison) -> "Ge[TSupportsComparison, S]": ...

	@overload
	def __ge__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "Ge[TSupportsComparison, S]": ...

	def __ge__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison] | TSupportsComparison
	) -> "Ge[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Ge(self, other)  # type: ignore[arg-type]
		else:
			return Ge(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __add__(self, other: TSupportsArithmetic) -> "Add[TSupportsArithmetic, S]": ...

	@overload
	def __add__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Add[TSupportsArithmetic, S]": ...

	def __add__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Add[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Add(self, other)  # type: ignore[arg-type]
		else:
			return Add(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __radd__(self, other: TSupportsArithmetic) -> "Add[TSupportsArithmetic, S]": ...

	@overload
	def __radd__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Add[TSupportsArithmetic, S]": ...

	def __radd__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Add[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Add(other, self)  # type: ignore[arg-type]
		else:
			return Add(Const(None, other), self)  # type: ignore[arg-type]

	@overload
	def __sub__(self, other: TSupportsArithmetic) -> "Sub[TSupportsArithmetic, S]": ...

	@overload
	def __sub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Sub[TSupportsArithmetic, S]": ...

	def __sub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Sub[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Sub(self, other)  # type: ignore[arg-type]
		else:
			return Sub(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __rsub__(self, other: TSupportsArithmetic) -> "Sub[TSupportsArithmetic, S]": ...

	@overload
	def __rsub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Sub[TSupportsArithmetic, S]": ...

	def __rsub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Sub[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Sub(other, self)  # type: ignore[arg-type]
		else:
			return Sub(Const(None, other), self)  # type: ignore[arg-type]

	@overload
	def __mul__(self, other: TSupportsArithmetic) -> "Mul[TSupportsArithmetic, S]": ...

	@overload
	def __mul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Mul[TSupportsArithmetic, S]": ...

	def __mul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Mul[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Mul(self, other)  # type: ignore[arg-type]
		else:
			return Mul(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __rmul__(self, other: TSupportsArithmetic) -> "Mul[TSupportsArithmetic, S]": ...

	@overload
	def __rmul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Mul[TSupportsArithmetic, S]": ...

	def __rmul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Mul[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Mul(other, self)  # type: ignore[arg-type]
		else:
			return Mul(Const(None, other), self)  # type: ignore[arg-type]

	@overload
	def __truediv__(self, other: float) -> "Div[float, S]": ...

	@overload
	def __truediv__(self, other: Expr[float, S, float]) -> "Div[float, S]": ...

	def __truediv__(self, other: Expr[float, S, float] | float) -> "Div[float, S]":
		if isinstance(other, Expr):
			return Div(self, other)  # type: ignore[arg-type]
		else:
			return Div(self, Const[float](None, other))  # type: ignore[arg-type]

	@overload
	def __rtruediv__(self, other: float) -> "Div[float, S]": ...

	@overload
	def __rtruediv__(self, other: Expr[float, S, float]) -> "Div[float, S]": ...

	def __rtruediv__(self, other: Expr[float, S, float] | float) -> "Div[float, S]":
		if isinstance(other, Expr):
			return Div(other, self)  # type: ignore[arg-type]
		else:
			return Div(Const[float](None, other), self)  # type: ignore[arg-type]

	@overload
	def __pow__(self, power: TSupportsArithmetic) -> "Pow[TSupportsArithmetic, S]": ...

	@overload
	def __pow__(
		self, power: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Pow[TSupportsArithmetic, S]": ...

	def __pow__(
		self, power: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Pow[TSupportsArithmetic, S]":
		if isinstance(power, Expr):
			return Pow(self, power)  # type: ignore[arg-type]
		else:
			return Pow(self, Const(None, power))  # type: ignore[arg-type]

	@overload
	def __rpow__(self, other: TSupportsArithmetic) -> "Pow[TSupportsArithmetic, S]": ...

	@overload
	def __rpow__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Pow[TSupportsArithmetic, S]": ...

	def __rpow__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Pow[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Pow(other, self)  # type: ignore[arg-type]
		else:
			return Pow(Const(None, other), self)  # type: ignore[arg-type]

	@overload
	def __mod__(self, other: TSupportsArithmetic) -> "Mod[TSupportsArithmetic, S]": ...

	@overload
	def __mod__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Mod[TSupportsArithmetic, S]": ...

	def __mod__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Mod[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Mod(self, other)  # type: ignore[arg-type]
		else:
			return Mod(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __rmod__(self, other: TSupportsArithmetic) -> "Mod[TSupportsArithmetic, S]": ...

	@overload
	def __rmod__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Mod[TSupportsArithmetic, S]": ...

	def __rmod__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Mod[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Mod(other, self)  # type: ignore[arg-type]
		else:
			return Mod(Const(None, other), self)  # type: ignore[arg-type]

	def __neg__(self) -> "Neg[T, S]":
		return Neg(self)


@dataclass(frozen=True, eq=False, slots=True)
class BoundExpr(
	BinaryOperationOverloads[R, Any],
	BooleanBinaryOperationOverloads[R, Any],
	Generic[T, S, R],
):
	"""An immutable expression bound to a specific context.

	BoundExpr satisfies the Expr protocol as a "closed term" - it ignores
	any context passed to eval/to_string/partial and uses its captured context.
	This makes it composable with other expressions.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> bound = (x > 5).bind(Ctx(x=10))
	>>> bound.unwrap()
	True
	>>> str(bound)
	'(x:10 > 5 -> True)'
	>>> isinstance(bound, Expr)
	True
	>>> bound.eval(()).value
	True
	"""

	expr: Expr[T, S, R]
	ctx: S

	def eval(self, ctx: Any) -> "Const[R]":  # noqa: ARG002
		return Const(None, self.expr.unwrap(self.ctx))

	def __call__(self, ctx: Any) -> "Const[R]":  # noqa: ARG002
		return self.eval(ctx)

	def to_string(self, ctx: Any | None = None) -> str:  # noqa: ARG002
		return self.expr.to_string(self.ctx)

	def partial(self, ctx: Any) -> "BoundExpr[R, Any, R]":  # noqa: ARG002
		return self  # type: ignore[return-value]

	def unwrap(self, ctx: Any = None) -> R:  # noqa: ARG002
		return self.expr.unwrap(self.ctx)

	def __str__(self) -> str:
		return self.expr.to_string(self.ctx)

	def bind(self, ctx: Any) -> "BoundExpr[R, Any, R]":  # noqa: ARG002
		return self  # type: ignore[return-value]

	def to_func(self) -> "Func[R, Any]":
		return Func((), self)

	def map(self, container: Expr[SizedIterable[Any], Any, Any]) -> "MapExpr[Any, R, Any]":
		return MapExpr(self.to_func(), container)


@dataclass(frozen=True, eq=False, slots=True)
class Const(BinaryOperationOverloads[T, Any], BooleanBinaryOperationOverloads[T, Any]):
	"""A constant that evaluates to itself.

	>>> MY_CONST = Const("My Const", 42)
	>>> MY_CONST.eval(None)
	Const(name='My Const', value=42)
	>>> MY_CONST.to_string()
	'My Const:42'
	>>> MY_CONST.unwrap(None)
	42

	You can create an unnamed constant by passing `None` as the name, but in
	this case, literals should be preferred.

	>>> MY_UNNAMED_CONST = Const(None, 42)
	>>> MY_UNNAMED_CONST.to_string()
	'42'
	>>> MY_LITERAL_CONST = 42
	>>> str(MY_LITERAL_CONST)
	'42'
	>>> (MY_CONST == MY_UNNAMED_CONST).to_string({})
	'(My Const:42 == 42 -> True)'
	>>> (MY_CONST == 42).to_string({})
	'(My Const:42 == 42 -> True)'

	"""

	name: str | None
	value: T

	def eval(self, ctx: Any) -> "Const[T]":  # noqa: ARG002
		return self

	def to_string(self, ctx: Any | None = None) -> str:  # noqa: ARG002
		return f"{self.name}:{self.value}" if self.name else str(self.value)

	def partial(self, ctx: Any) -> "Const[T]":  # noqa: ARG002
		return self


@dataclass(frozen=True, eq=False, slots=True)
class Var(BinaryOperationOverloads[T, S], BooleanBinaryOperationOverloads[T, S]):
	"""A variable that evaluates to the named attribute of the context.

	>>> from typing import NamedTuple
	...
	>>> class Context(NamedTuple):
	... 	my_var: int
	...
	>>> my_var = Var[int, Context]("my_var")
	...
	>>> my_var.to_string()
	'my_var'

	>>> my_var.eval(Context(my_var=42))
	Const(name='my_var', value=42)

	>>> my_var.to_string(Context(my_var=43))
	'my_var:43'

	>>> my_var.unwrap(Context(my_var=44))
	44
	"""

	name: str

	def eval(self, ctx: S) -> Const[T]:
		try:
			return Const(self.name, getattr(ctx, self.name))
		except AttributeError as e:
			available_attrs = (attr for attr in dir(ctx) if not attr.startswith("_"))
			suggestions = get_close_matches(self.name, available_attrs, n=3, cutoff=0.6)

			suggestion_text = ""
			if suggestions:
				suggestion_text = f" (did you mean {', '.join(repr(s) for s in suggestions)}?)"

			raise EvalError(f"Variable '{self.name}' not found in context{suggestion_text}") from e

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return self.name
		else:
			return f"{self.name}:{self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "Expr[T, Any, T]":
		if hasattr(ctx, self.name):
			return Const(self.name, getattr(ctx, self.name))
		return self


@dataclass(frozen=True, eq=False, slots=True)
class UnaryOpEval(Eval[bool, S]):
	left: Expr[Any, S, Any]


class UnaryOpToString(ToString[S], UnaryOpEval[S]):
	op: ClassVar[str] = "not "
	template: ClassVar[str] = "({op}{left})"
	template_eval: ClassVar[str] = "({op}{left} -> {out})"

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		if ctx is None:
			return self.template.format(op=self.op, left=left)
		else:
			return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


class UnaryToStringMixin(ToString[S], Eval[T, S], Generic[T, S]):
	left: Expr[T, S, T]
	op: ClassVar[str]
	template: ClassVar[str]
	template_eval: ClassVar[str]

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		if ctx is None:
			return self.template.format(op=self.op, left=left)
		return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


class Not(UnaryOpToString[S], UnaryOperationOverloads[S], BooleanBinaryOperationOverloads[bool, S]):
	op: ClassVar[str] = "not "

	def eval(self, ctx: S) -> Const[bool]:
		return Const(None, not self.left.eval(ctx).value)

	def partial(self, ctx: Any) -> "Expr[bool, Any, bool]":
		return Not(self.left.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class Neg(
	UnaryToStringMixin[T, S],
	BinaryOperationOverloads[T, S],
	BooleanBinaryOperationOverloads[T, S],
):
	left: Expr[T, S, T]
	op: ClassVar[str] = "-"
	template: ClassVar[str] = "({op}{left})"
	template_eval: ClassVar[str] = "({op}{left} -> {out})"

	def eval(self, ctx: S) -> Const[T]:
		return Const(None, -self.left.eval(ctx).value)  # type: ignore[operator]

	def partial(self, ctx: Any) -> "Neg[T, Any]":
		return Neg(self.left.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class Abs(
	UnaryToStringMixin[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	"""Absolute value expression.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> abs_x = Abs(x)
	>>> abs_x.to_string()
	'(abs x)'
	>>> abs_x.unwrap(Ctx(x=-5))
	5
	>>> abs_x.unwrap(Ctx(x=5))
	5
	>>> abs_x.to_string(Ctx(x=-5))
	'(abs x:-5 -> 5)'
	"""

	left: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	op: ClassVar[str] = "abs"
	template: ClassVar[str] = "({op} {left})"
	template_eval: ClassVar[str] = "({op} {left} -> {out})"

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, abs(self.left.eval(ctx).value))

	def partial(self, ctx: Any) -> "Abs[TSupportsArithmetic, Any]":
		return Abs(self.left.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class ClampExpr(
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[TSupportsComparison, S],
):
	"""Clamp expression - use Clamp(lo, hi)(expr) to create.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> clamped = Clamp(0, 10)(x)
	>>> clamped.to_string()
	'(clamp 0 10 x)'
	>>> clamped.unwrap(Ctx(x=-5))
	0
	>>> clamped.unwrap(Ctx(x=5))
	5
	>>> clamped.unwrap(Ctx(x=15))
	10
	>>> clamped.to_string(Ctx(x=15))
	'(clamp 0 10 x:15 -> 10)'
	"""

	lo: TSupportsComparison
	hi: TSupportsComparison
	value: Expr[TSupportsComparison, S, TSupportsComparison]
	op: ClassVar[str] = "clamp"

	def eval(self, ctx: S) -> Const[TSupportsComparison]:
		v = self.value.eval(ctx).value
		return Const(None, max(self.lo, min(self.hi, v)))

	def unwrap(self, ctx: S) -> TSupportsComparison:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		val_str: Final = self.value.to_string(ctx)
		if ctx is None:
			return f"({self.op} {self.lo} {self.hi} {val_str})"
		return f"({self.op} {self.lo} {self.hi} {val_str} -> {self.eval(ctx).value})"

	def partial(self, ctx: Any) -> "ClampExpr[TSupportsComparison, Any]":
		return ClampExpr(self.lo, self.hi, self.value.partial(ctx))


class Clamp(Generic[TSupportsComparison]):
	"""Curried clamp: Clamp(lo, hi)(expr) clamps expr to range [lo, hi].

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> clamped = Clamp(0, 10)(x)
	>>> clamped.to_string()
	'(clamp 0 10 x)'
	>>> clamped.unwrap(Ctx(x=-5))
	0
	>>> clamped.unwrap(Ctx(x=5))
	5
	>>> clamped.unwrap(Ctx(x=15))
	10
	>>> # Reusable clamper
	>>> normalize = Clamp(0.0, 1.0)
	>>> normalize(x).to_string()
	'(clamp 0.0 1.0 x)'
	"""

	__slots__ = ("lo", "hi")

	def __init__(self, lo: TSupportsComparison, hi: TSupportsComparison) -> None:
		self.lo = lo
		self.hi = hi

	def __call__(
		self, value: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> ClampExpr[TSupportsComparison, S]:
		return ClampExpr(self.lo, self.hi, value)

	def __repr__(self) -> str:
		return f"Clamp({self.lo}, {self.hi})"


class BinaryOpProtocol(Expr[T, S, R], Protocol[T, S, R]):
	left: Expr[T, S, T]
	right: Expr[T, S, T]


@dataclass(frozen=True, eq=False, slots=True)
class BinaryOp(ToString[S], BinaryOpProtocol[T, S, R], Generic[T, S, R]):
	op: ClassVar[str] = " ? "
	template: ClassVar[str] = "({left}{op}{right})"
	template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

	left: Expr[T, S, T]
	right: Expr[T, S, T]

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		right: Final = self.right.to_string(ctx)
		if ctx is None:
			return self.template.format(left=left, op=self.op, right=right)
		else:
			return self.template_eval.format(
				left=left, op=self.op, right=right, out=self.eval(ctx).value
			)

	def partial(self, ctx: Any) -> "Expr[T, Any, R]":
		return type(self)(self.left.partial(ctx), self.right.partial(ctx))


class SymmetricalBinaryOpProtocol(BinaryOpProtocol[T, S, T], Protocol[T, S]):
	op_func: Callable[[Expr[T, S, T], Expr[T, S, T]], Expr[T, S, T]]
	identity_element: T


@dataclass(frozen=True, eq=False, slots=True)
class Foldable(SymmetricalBinaryOpProtocol[T, S], BinaryOp[T, S, T]): ...


class And(
	Foldable[TSupportsLogic, S],
	BooleanBinaryOperationOverloads[TSupportsLogic, S],
):
	op: ClassVar[str] = " & "
	op_func: Final[ClassVar] = operator.and_  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = True  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsLogic]:
		return Const(None, self.left.eval(ctx).value and self.right.eval(ctx).value)


class Or(
	Foldable[TSupportsLogic, S],
	BooleanBinaryOperationOverloads[TSupportsLogic, S],
):
	op: ClassVar[str] = " | "
	op_func: Final[ClassVar] = operator.or_  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = False  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsLogic]:
		return Const(None, self.left.eval(ctx).value or self.right.eval(ctx).value)


class Eq(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsEquality, S, bool],
	BinaryOperationOverloads[TSupportsEquality, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " == "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value == self.right.eval(ctx).value)


class Ne(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsEquality, S, bool],
	BinaryOperationOverloads[TSupportsEquality, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " != "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value != self.right.eval(ctx).value)


class Lt(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsComparison, S, bool],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " < "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value < self.right.eval(ctx).value)


class Le(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsComparison, S, bool],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " <= "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value <= self.right.eval(ctx).value)


class Gt(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsComparison, S, bool],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " > "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value > self.right.eval(ctx).value)


class Ge(  # pyright: ignore[reportGeneralTypeIssues]
	BinaryOp[TSupportsComparison, S, bool],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " >= "

	def eval(self, ctx: S) -> Const[bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
		return Const(None, self.left.eval(ctx).value >= self.right.eval(ctx).value)


class Min(
	Foldable[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
):
	op: ClassVar[str] = "min"
	template: ClassVar[str] = "({op} {left} {right})"
	template_eval: ClassVar[str] = "({op} {left} {right} -> {out})"
	op_func: Final[ClassVar] = min  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = float("inf")  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsComparison]:
		return Const(None, min(self.left.eval(ctx).value, self.right.eval(ctx).value))


class Max(
	Foldable[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
):
	op: ClassVar[str] = "max"
	template: ClassVar[str] = "({op} {left} {right})"
	template_eval: ClassVar[str] = "({op} {left} {right} -> {out})"
	op_func: Final[ClassVar] = max  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = float("-inf")  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsComparison]:
		return Const(None, max(self.left.eval(ctx).value, self.right.eval(ctx).value))


class Add(
	Foldable[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " + "
	op_func: Final[ClassVar] = operator.add  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = 0  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value + self.right.eval(ctx).value)


class Sub(
	Foldable[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " - "
	op_func: Final[ClassVar] = operator.sub  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = 0  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value - self.right.eval(ctx).value)


class Mul(
	Foldable[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " * "
	op_func: Final[ClassVar] = operator.mul  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = 1  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value * self.right.eval(ctx).value)


class Div(
	Foldable[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " / "
	op_func: Final[ClassVar] = operator.truediv  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = 1  # type: ignore[misc,assignment]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value / self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class Pow(
	Foldable[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = "^"
	op_func: Final[ClassVar] = operator.pow  # type: ignore[misc,assignment]
	identity_element: Final[ClassVar] = 1  # type: ignore[misc,assignment]

	left: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	right: Expr[TSupportsArithmetic, S, TSupportsArithmetic]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value ** self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class Mod(
	BinaryOp[TSupportsArithmetic, S, TSupportsArithmetic],
	BinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " % "

	left: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	right: Expr[TSupportsArithmetic, S, TSupportsArithmetic]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value % self.right.eval(ctx).value)


class ConstToleranceProtocol(Protocol):
	@property
	def max_abs_error(self) -> _SupportsArithmetic: ...

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.max_abs_error}"


class ConstTolerance(
	Const[_SupportsArithmetic], ConstToleranceProtocol, Generic[TSupportsArithmetic]
):
	"""Base class for constants with tolerance for approximate comparisons.

	Tolerance constants automatically create Approximately expressions when
	compared with == to other values or expressions.

	>>> five_ish = PlusMinus("5ish", 5.0, plus_minus=0.1)
	>>> check = five_ish == 5.05
	>>> # Doctest assert_type: This creates an Approximately[float, Any]
	>>> isinstance(check, Approximately)
	True
	>>> check.unwrap(None)
	True
	"""

	def eval(self, ctx: Any) -> Self:
		return self

	def to_string(self, ctx: Any | None = None) -> str:
		if ctx is None:
			return (
				f"{self.name}:{self.value}{self.tolerance_string}"
				if self.name
				else f"{self.value}{self.tolerance_string}"
			)
		else:
			return (
				f"{self.name}:{self.eval(ctx).value}{self.tolerance_string}"
				if self.name
				else f"{self.eval(ctx).value}{self.tolerance_string}"
			)

	@overload  # type: ignore[override]
	def __eq__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "Approximately[TSupportsArithmetic, S]": ...

	@overload
	def __eq__(self, other: TSupportsArithmetic) -> "Approximately[TSupportsArithmetic, Any]": ...

	def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic] | TSupportsArithmetic
	) -> "Approximately[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Approximately(other, self)
		else:
			return Approximately(Const(None, other), self)


@dataclass(frozen=True, eq=False, slots=True)
class PlusMinus(ConstTolerance[TSupportsArithmetic]):
	"""A constant with plus/minus tolerance for approximate comparisons.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: float
	>>> x = Var[float, Ctx]("x")
	>>> target = PlusMinus("Target", 5.0, 0.1)
	>>> target.value
	5.0
	>>> target.max_abs_error
	0.1
	>>> target.to_string()
	'Target:5.0 ± 0.1'
	>>> # Comparison coerces an Approximately
	>>> expr = x == target
	>>> expr.to_string()
	'(x ≈ Target:5.0 ± 0.1)'
	>>> expr.unwrap(Ctx(x=5.05))
	True
	"""

	name: str | None
	value: TSupportsArithmetic
	plus_minus: TSupportsArithmetic

	@property
	def max_abs_error(self) -> TSupportsArithmetic:
		return self.plus_minus


@dataclass(frozen=True, eq=False, slots=True)
class Percent(ConstTolerance[TSupportsArithmetic]):
	"""A constant with percentage tolerance for approximate comparisons.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: float
	>>> x = Var[float, Ctx]("x")
	>>> target = Percent("Target", 100.0, 5.0)
	>>> target.value
	100.0
	>>> target.max_abs_error
	5.0
	>>> target.to_string()
	'Target:100.0 ± 5.0%'
	>>> # Comparison coerces an Approximately
	>>> expr = x == target
	>>> expr.to_string()
	'(x ≈ Target:100.0 ± 5.0%)'
	>>> expr.unwrap(Ctx(x=102.0))
	True
	"""

	name: str | None
	value: TSupportsArithmetic
	percent: float

	@property
	def max_abs_error(self) -> TSupportsArithmetic:
		return self.value * self.percent / 100.0  # type: ignore[no-any-return]

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.percent}%"


@dataclass(frozen=True, eq=False, slots=True)
class Approximately(
	BinaryOp[TSupportsArithmetic, S, bool],
	BooleanBinaryOperationOverloads[bool, S],
):
	"""Approximate equality comparison with tolerance.

	Created automatically when comparing values to a ConstTolerance using ==.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	value: float
	>>> value = Var[float, Ctx]("value")
	>>> tolerance = PlusMinus("Target", 10.0, 0.5)
	>>> approx = Approximately(value, tolerance)
	>>> approx.to_string()
	'(value ≈ Target:10.0 ± 0.5)'
	>>> approx.unwrap(Ctx(value=10.2))
	True
	>>> approx.unwrap(Ctx(value=10.6))
	False
	>>> # Comparison coerces an Approximately
	>>> auto_approx = value == tolerance
	>>> auto_approx.to_string()
	'(value ≈ Target:10.0 ± 0.5)'
	"""

	op: ClassVar[str] = " ≈ "

	left: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	right: ConstTolerance[TSupportsArithmetic]  # type: ignore[assignment]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(
			None, abs(self.left.eval(ctx).value - self.right.value) <= self.right.max_abs_error
		)

	def partial(self, ctx: Any) -> "Expr[TSupportsArithmetic, Any, bool]":
		return Approximately(self.left.partial(ctx), self.right)


@dataclass(frozen=True, eq=False, slots=True)
class Predicate(BooleanBinaryOperationOverloads[bool, S]):
	"""A named predicate that evaluates to `True` or `False`.

	>>> my_predicate = Predicate("My Predicate", Const("two", 2) == 2)
	>>> my_predicate.to_string({})
	'My Predicate: True (two:2 == 2 -> True)'

	>>> from typing import NamedTuple
	...
	>>> class Sides(NamedTuple):
	... 	a: int
	... 	b: int
	...
	>>> a = Var[int, Sides]("a")
	>>> b = Var[int, Sides]("b")
	>>> C = Const("c", 5)
	>>> pythagorean_theorem = a**2 + b**2 == C**2
	>>> pythagorean_theorem.to_string()
	'(((a^2) + (b^2)) == (c:5^2))'
	>>> pythagorean_theorem.to_string(Sides(a=3, b=4))
	'(((a:3^2 -> 9) + (b:4^2 -> 16) -> 25) == (c:5^2 -> 25) -> True)'
	>>> is_right = Predicate(
	... 	"Pythagorean theorem holds",
	... 	pythagorean_theorem
	... )
	...
	>>> is_right.to_string()
	'Pythagorean theorem holds: (((a^2) + (b^2)) == (c:5^2))'
	>>> is_right.to_string(Sides(a=3, b=4))
	'Pythagorean theorem holds: True (((a:3^2 -> 9) + (b:4^2 -> 16) -> 25) == (c:5^2 -> 25) -> True)'
	>>> is_right.unwrap(Sides(a=3, b=4))
	True
	>>> is_right.to_string(Sides(a=1, b=2))
	'Pythagorean theorem holds: False (((a:1^2 -> 1) + (b:2^2 -> 4) -> 5) == (c:5^2 -> 25) -> False)'
	>>> is_right.unwrap(Sides(a=1, b=2))
	False
	"""

	name: str | None
	expr: Expr[Any, S, bool]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(self.name, self.expr.eval(ctx).value)

	def __call__(self, ctx: S) -> Const[bool]:
		return self.eval(ctx)

	def unwrap(self, ctx: S) -> bool:
		return self.eval(ctx).value

	def bind(self, ctx: S) -> "BoundExpr[bool, S, bool]":
		return BoundExpr(self, ctx)

	def to_string(self, ctx: S | None = None) -> str:
		result: Final = (
			self.expr.to_string(ctx)
			if ctx is None
			else f"{self.unwrap(ctx)} {self.expr.to_string(ctx)}"
		)
		return f"{self.name}: {result}" if self.name else result

	def partial(self, ctx: Any) -> "Predicate[Any]":
		return Predicate(self.name, self.expr.partial(ctx))

	def to_func(self) -> "Func[bool, S]":
		return Func(_extract_vars((), self), self)

	def map(self, container: "Expr[SizedIterable[Any], Any, Any]") -> "MapExpr[Any, bool, Any]":
		return MapExpr(self.to_func(), container)


def format_iterable_var(expr: Expr[SizedIterable[Any], S, Any], ctx: S | None) -> str:
	"""Format an iterable variable with compact container display.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], Ctx]("nums")
	>>> format_iterable_var(nums, None)
	'nums'
	>>> format_iterable_var(nums, Ctx(nums=[1, 2]))
	'nums:2[1,2]'
	>>> format_iterable_var(nums, Ctx(nums=[1, 2, 3, 4, 5]))
	'nums:5[1,..5]'
	"""
	if ctx is None:
		return expr.to_string(ctx)

	value: Final = expr.unwrap(ctx)

	if isinstance(value, (str, bytes)):
		return expr.to_string(ctx)

	length: Final = len(value)
	name: Final = getattr(expr, "name", None)
	prefix: Final = f"{name}:" if name else ""

	def _serialize_elem(elem: Any) -> str:
		return elem.to_string(ctx) if hasattr(elem, "to_string") else str(elem)

	# Handle different container types
	if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
		# Indexable sequences (list, tuple)
		if length <= 2:
			return f"{prefix}{length}[{','.join(_serialize_elem(elem) for elem in value)}]"
		else:
			return f"{prefix}{length}[{_serialize_elem(value[0])},..{_serialize_elem(value[-1])}]"
	else:
		# Sets, other iterables without indexing
		elements = list(value)
		if length <= 2:
			return f"{prefix}{length}[{','.join(_serialize_elem(elem) for elem in elements)}]"
		else:
			return f"{prefix}{length}[{_serialize_elem(elements[0])},..{_serialize_elem(elements[-1])}]"


@dataclass(frozen=True, eq=False, slots=True)
class Contains(
	BinaryOperationOverloads[bool, S],
	BooleanBinaryOperationOverloads[bool, S],
	Generic[T, S],
):
	"""Check if a value is contained in a collection.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	... 	target: int
	>>> values = Var[SizedIterable[int], Ctx]("values")
	>>> target = Var[int, Ctx]("target")
	>>> contains_expr = Contains(target, values)
	>>> contains_expr.to_string()
	'(target in values)'
	>>> contains_expr.unwrap(Ctx(values=[1, 2, 3], target=2))
	True
	>>> contains_expr.unwrap(Ctx(values=[1, 2, 3], target=5))
	False
	>>> contains_expr.to_string(Ctx(values=[1, 2, 3], target=2))
	'(target:2 in values:3[1,..3] -> True)'
	"""

	op: ClassVar[str] = " in "
	template: ClassVar[str] = "({left}{op}{right})"
	template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

	element: Expr[T, S, T]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(None, self.element.unwrap(ctx) in self.container.unwrap(ctx))

	def unwrap(self, ctx: S) -> bool:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.element.to_string(ctx)
		right: Final = format_iterable_var(self.container, ctx)
		if ctx is None:
			return self.template.format(left=left, op=self.op, right=right)
		else:
			return self.template_eval.format(
				left=left, op=self.op, right=right, out=self.eval(ctx).value
			)

	def partial(self, ctx: Any) -> "Expr[bool, Any, bool]":
		return Contains(self.element.partial(ctx), self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class MapExpr(
	BinaryOperationOverloads[SizedIterable[U], S],
	Generic[T, U, S],
):
	"""Apply a function to each element in a container."""

	op: ClassVar[str] = "map"
	template: ClassVar[str] = "({op} {func} {container})"
	template_eval: ClassVar[str] = "({op} {func} {container} -> {out})"

	func: "Func[U, Any]"
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> "Const[SizedIterable[U]]":
		container_values = self.container.unwrap(ctx)
		result: list[U] = []
		for item in container_values:
			if self.func.args and hasattr(self.func.args[0], "name"):
				arg_name = self.func.args[0].name  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
				temp_ctx = type("TempCtx", (), {arg_name: item})()
				result.append(self.func.expr.unwrap(temp_ctx))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
		return Const(None, result)

	def unwrap(self, ctx: S) -> SizedIterable[U]:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		func_str = self.func.to_string()
		if ctx is None:
			container_str = self.container.to_string()
			return self.template.format(op=self.op, func=func_str, container=container_str)
		else:
			container_str = format_iterable_var(self.container, ctx)
			result_value = self.eval(ctx).value
			result_expr = Const(None, result_value)
			out_str = format_iterable_var(result_expr, ctx)
			return self.template_eval.format(
				op=self.op, func=func_str, container=container_str, out=out_str
			)

	def partial(self, ctx: Any) -> "Expr[SizedIterable[U], Any, SizedIterable[U]]":
		partial_func = Func(self.func.args, self.func.expr.partial(ctx))
		return MapExpr(partial_func, self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class FoldLExpr(
	BinaryOperationOverloads[R, S],
	Generic[T, S, R],
):
	op_cls: type[Foldable[Any, Any]]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]
	initial: R | None = None

	@overload
	def __init__(
		self: "FoldLExpr[Expr[Any, Any, R], S, R]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Expr[Any, Any, R]], S, SizedIterable[Expr[Any, Any, R]]],
		initial: None = None,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[Expr[Any, Any, R], S, R]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Expr[Any, Any, R]], S, SizedIterable[Expr[Any, Any, R]]],
		initial: R = ...,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[T, S, T]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[T], S, SizedIterable[T]],
		initial: None = None,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[T, S, T]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[T], S, SizedIterable[T]],
		initial: T = ...,
	) -> None: ...

	def __init__(
		self,
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Any], S, SizedIterable[Any]],
		initial: Any | None = None,
	) -> None:
		object.__setattr__(self, "op_cls", op_cls)
		object.__setattr__(self, "container", container)
		object.__setattr__(self, "initial", initial)

	def eval(self, ctx: S) -> Const[R]:
		result_value: R = self.initial if self.initial is not None else self.op_cls.identity_element  # pyright: ignore[reportGeneralTypeIssues]

		for item in self.container.unwrap(ctx):
			item_value: R = item.unwrap(ctx) if isinstance(item, Expr) else item  # type: ignore[assignment]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAssignmentType]
			result_value = self.op_cls.op_func(result_value, item_value)  # type: ignore[arg-type, assignment]  # pyright: ignore[reportArgumentType]

		return Const(None, result_value)

	def unwrap(self, ctx: S) -> R:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		op_str: Final = self.op_cls.op.strip()
		container_name: Final = (
			self.container.to_string()
			if ctx is None
			else self.container.to_string() + f":{len(self.container.unwrap(ctx))}"
		)
		if ctx is None:
			return f"(foldl {op_str} {container_name})"
		result = self.eval(ctx).value
		if self.op_cls in (Min, Max):
			return f"(foldl {op_str} {container_name} -> {result})"
		items = list(self.container.unwrap(ctx))
		initial_str = f"{str(self.initial)}{self.op_cls.op}" if self.initial is not None else ""
		items_str = self._format_items(items, op_str, ctx)
		return f"(foldl {op_str} {container_name} -> ({initial_str}{items_str}) -> {result})"

	@staticmethod
	def _format_items(items: list[Any], op: str, ctx: S) -> str:
		def serialize(item: Any) -> str:
			if isinstance(item, Expr):
				return item.to_string(ctx)  # pyright: ignore[reportUnknownMemberType]
			return str(item)

		return f" {op} ".join(serialize(i) for i in items)

	def partial(self, ctx: Any) -> "Expr[R, Any, R]":
		return FoldLExpr(self.op_cls, self.container.partial(ctx), self.initial)  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False, slots=True)
class BoolContainerFoldBase(
	BinaryOperationOverloads[bool, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str]
	op_cls: ClassVar[type[Foldable[Any, Any]]]
	template: ClassVar[str] = "({op} {left})"
	template_eval: ClassVar[str] = "({op} {left} -> {out})"

	container: Expr[SizedIterable[Any], S, SizedIterable[Any]]
	_foldl: FoldLExpr[bool, S, bool] = field(init=False)

	def __post_init__(self) -> None:
		object.__setattr__(self, "_foldl", FoldLExpr(self.op_cls, self.container))

	def eval(self, ctx: S) -> Const[bool]:
		return self._foldl.eval(ctx)

	def unwrap(self, ctx: S) -> bool:
		return self._foldl.unwrap(ctx)

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return self.template.format(op=self.op, left=format_iterable_var(self.container, ctx))
		left = (
			format_iterable_var(self.container, ctx)
			if isinstance(self.container, (Var, Const))
			else self.container.to_string(ctx)
		)
		return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class AnyExpr(BoolContainerFoldBase[S]):
	"""True if any element in the container is truthy.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[bool]
	>>> values = Var[list[bool], Ctx]("values")
	>>> any_expr = AnyExpr(values)
	>>> any_expr.to_string()
	'(any values)'
	>>> any_expr.unwrap(Ctx(values=[False, True, False]))
	True
	>>> any_expr.unwrap(Ctx(values=[False, False, False]))
	False
	>>> any_expr.to_string(Ctx(values=[False, True, False]))
	'(any values:3[False,..False] -> True)'

	With complex expressions like MapExpr, shows the full evaluation trace:

	>>> class NumCtx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], NumCtx]("nums")
	>>> n = Var[int, NumCtx]("n")
	>>> gt_five = (n > 5).map(nums)
	>>> any_gt_five = AnyExpr(gt_five)
	>>> any_gt_five.to_string()
	'(any (map n -> (n > 5) nums))'
	>>> any_gt_five.to_string(NumCtx(nums=[3, 7, 2]))
	'(any (map n -> (n > 5) nums:3[3,..2] -> 3[False,..False]) -> True)'
	>>> any_gt_five.to_string(NumCtx(nums=[1, 2, 3]))
	'(any (map n -> (n > 5) nums:3[1,..3] -> 3[False,..False]) -> False)'
	"""

	op: ClassVar[str] = "any"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Or

	def partial(self, ctx: Any) -> "AnyExpr[Any]":
		return AnyExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class AllExpr(BoolContainerFoldBase[S]):
	"""True if all elements in the container are truthy.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[bool]
	>>> values = Var[list[bool], Ctx]("values")
	>>> all_expr = AllExpr(values)
	>>> all_expr.to_string()
	'(all values)'
	>>> all_expr.unwrap(Ctx(values=[True, True, True]))
	True
	>>> all_expr.unwrap(Ctx(values=[True, False, True]))
	False
	>>> all_expr.to_string(Ctx(values=[True, False, True]))
	'(all values:3[True,..True] -> False)'

	With complex expressions like MapExpr, shows the full evaluation trace:

	>>> class NumCtx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], NumCtx]("nums")
	>>> n = Var[int, NumCtx]("n")
	>>> lt_ten = (n < 10).map(nums)
	>>> all_lt_ten = AllExpr(lt_ten)
	>>> all_lt_ten.to_string()
	'(all (map n -> (n < 10) nums))'
	>>> all_lt_ten.to_string(NumCtx(nums=[3, 7, 2]))
	'(all (map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> True)'
	>>> all_lt_ten.to_string(NumCtx(nums=[3, 15, 2]))
	'(all (map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> False)'
	"""

	op: ClassVar[str] = "all"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = And

	def partial(self, ctx: Any) -> "AllExpr[Any]":
		return AllExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class ComparableContainerFoldBase(
	BinaryOperationOverloads[TSupportsComparison, S],
	Generic[TSupportsComparison, S],
):
	op: ClassVar[str]
	op_cls: ClassVar[type[Foldable[Any, Any]]]
	template: ClassVar[str] = "({op} {left})"
	template_eval: ClassVar[str] = "({op} {left} -> {out})"

	container: Expr[SizedIterable[TSupportsComparison], S, SizedIterable[TSupportsComparison]]
	_foldl: FoldLExpr[TSupportsComparison, S, TSupportsComparison] = field(init=False)

	def __post_init__(self) -> None:
		object.__setattr__(self, "_foldl", FoldLExpr(self.op_cls, self.container))

	def eval(self, ctx: S) -> Const[TSupportsComparison]:
		return self._foldl.eval(ctx)

	def unwrap(self, ctx: S) -> TSupportsComparison:
		return self._foldl.unwrap(ctx)

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return self.template.format(op=self.op, left=format_iterable_var(self.container, ctx))
		left = (
			format_iterable_var(self.container, ctx)
			if isinstance(self.container, (Var, Const))
			else self.container.to_string(ctx)
		)
		return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class MinExpr(ComparableContainerFoldBase[TSupportsComparison, S]):
	"""Minimum element in a container.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[list[int], Ctx]("values")
	>>> min_expr = MinExpr(values)
	>>> min_expr.to_string()
	'(min values)'
	>>> min_expr.unwrap(Ctx(values=[3, 1, 4, 1, 5]))
	1
	>>> min_expr.to_string(Ctx(values=[3, 1, 4]))
	'(min values:3[3,..4] -> 1)'
	"""

	op: ClassVar[str] = "min"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Min

	def partial(self, ctx: Any) -> "MinExpr[TSupportsComparison, Any]":
		return MinExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class FilterExpr(
	BinaryOperationOverloads[SizedIterable[T], S],
	Generic[T, S],
):
	"""Filter container elements by predicate.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[SizedIterable[int], Ctx]("values")
	>>> x = Var[int, Ctx]("x")
	>>> is_positive = (x > 0).to_func()
	>>> filter_expr = FilterExpr(is_positive, values)
	>>> filter_expr.to_string()
	'(filter x -> (x > 0) values)'
	>>> filter_expr.unwrap(Ctx(values=[-1, 2, -3, 4, 5]))
	(2, 4, 5)
	>>> filter_expr.to_string(Ctx(values=[-1, 2, -3, 4, 5]))
	'(filter x -> (x > 0) values:5[-1,..5] -> 3[2,..5])'
	"""

	op: ClassVar[str] = "filter"
	template: ClassVar[str] = "({op} {func} {container})"
	template_eval: ClassVar[str] = "({op} {func} {container} -> {out})"

	predicate: Func[bool, Any]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> Const[SizedIterable[T]]:
		predicate_results = MapExpr(self.predicate, self.container).unwrap(ctx)
		return Const(
			None,
			tuple(
				value
				for value, keep in zip(self.container.unwrap(ctx), predicate_results, strict=True)
				if keep
			),
		)

	def unwrap(self, ctx: S) -> SizedIterable[T]:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		func_str = self.predicate.to_string()
		if ctx is None:
			container_str = self.container.to_string()
			return self.template.format(op=self.op, func=func_str, container=container_str)
		else:
			container_str = format_iterable_var(self.container, ctx)
			result_value = self.eval(ctx).value
			result_expr = Const(None, result_value)
			out_str = format_iterable_var(result_expr, ctx)
			return self.template_eval.format(
				op=self.op, func=func_str, container=container_str, out=out_str
			)

	def partial(self, ctx: Any) -> "Expr[SizedIterable[T], Any, SizedIterable[T]]":
		partial_func = Func(self.predicate.args, self.predicate.expr.partial(ctx))
		return FilterExpr(partial_func, self.container.partial(ctx))


type FloatVar[S] = Var[float, S]
type IntVar[S] = Var[int, S]
type BoolVar[S] = Var[bool, S]
type StrVar[S] = Var[str, S]
type ListVar[T, S] = Var[SizedIterable[T], S]


@overload
def context_vars[T1](
	f1: tuple[str, type[T1]], /
) -> tuple[Callable[[T1], tuple[T1]], Var[T1, tuple[T1]]]: ...
@overload
def context_vars[T1, T2](
	f1: tuple[str, type[T1]], f2: tuple[str, type[T2]], /
) -> tuple[Callable[[T1, T2], tuple[T1, T2]], Var[T1, tuple[T1, T2]], Var[T2, tuple[T1, T2]]]: ...
@overload
def context_vars[T1, T2, T3](
	f1: tuple[str, type[T1]], f2: tuple[str, type[T2]], f3: tuple[str, type[T3]], /
) -> tuple[
	Callable[[T1, T2, T3], tuple[T1, T2, T3]],
	Var[T1, tuple[T1, T2, T3]],
	Var[T2, tuple[T1, T2, T3]],
	Var[T3, tuple[T1, T2, T3]],
]: ...
@overload
def context_vars[T1, T2, T3, T4](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4], tuple[T1, T2, T3, T4]],
	Var[T1, tuple[T1, T2, T3, T4]],
	Var[T2, tuple[T1, T2, T3, T4]],
	Var[T3, tuple[T1, T2, T3, T4]],
	Var[T4, tuple[T1, T2, T3, T4]],
]: ...
@overload
def context_vars[T1, T2, T3, T4, T5](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	f5: tuple[str, type[T5]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4, T5], tuple[T1, T2, T3, T4, T5]],
	Var[T1, tuple[T1, T2, T3, T4, T5]],
	Var[T2, tuple[T1, T2, T3, T4, T5]],
	Var[T3, tuple[T1, T2, T3, T4, T5]],
	Var[T4, tuple[T1, T2, T3, T4, T5]],
	Var[T5, tuple[T1, T2, T3, T4, T5]],
]: ...
@overload
def context_vars[T1, T2, T3, T4, T5, T6](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	f5: tuple[str, type[T5]],
	f6: tuple[str, type[T6]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4, T5, T6], tuple[T1, T2, T3, T4, T5, T6]],
	Var[T1, tuple[T1, T2, T3, T4, T5, T6]],
	Var[T2, tuple[T1, T2, T3, T4, T5, T6]],
	Var[T3, tuple[T1, T2, T3, T4, T5, T6]],
	Var[T4, tuple[T1, T2, T3, T4, T5, T6]],
	Var[T5, tuple[T1, T2, T3, T4, T5, T6]],
	Var[T6, tuple[T1, T2, T3, T4, T5, T6]],
]: ...


def context_vars(*fields: tuple[str, type[Any]]) -> Any:
	"""Create a context class and matching Var instances with preserved types.

	Takes (name, type) tuples and returns a tuple of (ContextClass, var1, var2, ...).
	Both Var types and the context type are preserved by the type checker for up to 6 fields.
	The context is typed as a Callable that returns a tuple of the field types.

	>>> Ctx, x, y = context_vars(("x", int), ("y", float))
	>>> expr = x + y
	>>> expr.to_string()
	'(x + y)'
	>>> expr.unwrap(Ctx(x=1, y=2.5))
	3.5
	>>> expr.to_string(Ctx(x=1, y=2.5))
	'(x:1 + y:2.5 -> 3.5)'

	>>> Measurements, voltage, current = context_vars(
	... 	("voltage", float),
	... 	("current", float),
	... )
	>>> power = voltage * current
	>>> power.unwrap(Measurements(voltage=5.0, current=2.0))
	10.0
	"""
	from typing import NamedTuple as NT

	context_class = NT("Ctx", list(fields))  # type: ignore[misc]
	return (context_class,) + tuple(Var(name) for name, _ in fields)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


@dataclass(frozen=True, eq=False, slots=True)
class MaxExpr(ComparableContainerFoldBase[TSupportsComparison, S]):
	"""Maximum element in a container.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[list[int], Ctx]("values")
	>>> max_expr = MaxExpr(values)
	>>> max_expr.to_string()
	'(max values)'
	>>> max_expr.unwrap(Ctx(values=[3, 1, 4, 1, 5]))
	5
	>>> max_expr.to_string(Ctx(values=[3, 1, 4]))
	'(max values:3[3,..4] -> 4)'
	"""

	op: ClassVar[str] = "max"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Max

	def partial(self, ctx: Any) -> "MaxExpr[TSupportsComparison, Any]":
		return MaxExpr(self.container.partial(ctx))


from mahonia.match import Match, MatchExpr  # noqa: E402

__all__ = [
	"Match",
	"MatchExpr",
]
