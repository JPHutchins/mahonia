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
mahonia.types.EvalError: Variable 'wrong_name' not found in context

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
from dataclasses import dataclass
from typing import (
	TYPE_CHECKING,
	Any,
	Callable,
	ClassVar,
	Final,
	Generic,
	Protocol,
	overload,
	runtime_checkable,
)

from mahonia.types import ContextProtocol as ContextProtocol
from mahonia.types import EvalError as EvalError
from mahonia.types import MergeContextProtocol as MergeContextProtocol
from mahonia.types import R as R
from mahonia.types import R_Eval as R_Eval
from mahonia.types import S as S
from mahonia.types import S_contra as S_contra
from mahonia.types import SizedIterable as SizedIterable
from mahonia.types import Ss as Ss
from mahonia.types import SupportsArithmetic as SupportsArithmetic
from mahonia.types import SupportsComparison as SupportsComparison
from mahonia.types import SupportsEquality as SupportsEquality
from mahonia.types import SupportsLogic as SupportsLogic
from mahonia.types import T as T
from mahonia.types import T_co as T_co
from mahonia.types import TSupportsArithmetic as TSupportsArithmetic
from mahonia.types import TSupportsComparison as TSupportsComparison
from mahonia.types import TSupportsEquality as TSupportsEquality
from mahonia.types import TSupportsLogic as TSupportsLogic
from mahonia.types import U as U

if TYPE_CHECKING:
	from mahonia.match import Match as Match
	from mahonia.match import MatchExpr as MatchExpr


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
		return Func(extract_vars((), self), self)

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


@dataclass(frozen=True, eq=False, slots=True)
class Func(Generic[T, S]):  # type: ignore[misc]
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


from mahonia.bound import BoundExpr as BoundExpr  # noqa: E402
from mahonia.containers import AllExpr as AllExpr  # noqa: E402
from mahonia.containers import AnyExpr as AnyExpr  # noqa: E402
from mahonia.containers import BoolContainerFoldBase as BoolContainerFoldBase  # noqa: E402
from mahonia.containers import (  # noqa: E402
	ComparableContainerFoldBase as ComparableContainerFoldBase,
)
from mahonia.containers import Contains as Contains  # noqa: E402
from mahonia.containers import FilterExpr as FilterExpr  # noqa: E402
from mahonia.containers import FoldLExpr as FoldLExpr  # noqa: E402
from mahonia.containers import MapExpr as MapExpr  # noqa: E402
from mahonia.containers import MaxExpr as MaxExpr  # noqa: E402
from mahonia.containers import MinExpr as MinExpr  # noqa: E402
from mahonia.context import context_vars as context_vars  # noqa: E402
from mahonia.context import merge as merge  # noqa: E402
from mahonia.extract_vars import extract_vars as extract_vars  # noqa: E402
from mahonia.formatting import format_iterable_var as format_iterable_var  # noqa: E402
from mahonia.match import Match as Match  # noqa: E402
from mahonia.match import MatchExpr as MatchExpr  # noqa: E402
from mahonia.predicate import Predicate as Predicate  # noqa: E402
from mahonia.tolerance import Approximately as Approximately  # noqa: E402
from mahonia.tolerance import ConstTolerance as ConstTolerance  # noqa: E402
from mahonia.tolerance import ConstToleranceProtocol as ConstToleranceProtocol  # noqa: E402
from mahonia.tolerance import Percent as Percent  # noqa: E402
from mahonia.tolerance import PlusMinus as PlusMinus  # noqa: E402
from mahonia.unary import Abs as Abs  # noqa: E402
from mahonia.unary import Clamp as Clamp  # noqa: E402
from mahonia.unary import ClampExpr as ClampExpr  # noqa: E402
from mahonia.var import Var as Var  # noqa: E402

type FloatVar[S] = Var[float, S]
type IntVar[S] = Var[int, S]
type BoolVar[S] = Var[bool, S]
type StrVar[S] = Var[str, S]
type ListVar[T, S] = Var[SizedIterable[T], S]
