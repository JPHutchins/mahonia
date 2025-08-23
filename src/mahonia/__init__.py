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

The most important rule to remember is that the leftmost leaf of your expression
must be a Mahonia `Expr`. This is usually a `Var` or `Const`, but can be any
type that implements the `Expr` protocol, e.g., any Mahonia expression.

For example, this will work, because the literal `1`
is coerced to an unnamed `Const`:
>>> X = Const("X", 41)
>>> X + 1
Add(left=Const(name='X', value=41), right=Const(name=None, value=1))

But this will not:
>>> 1 + X
Traceback (most recent call last):
	...
TypeError: unsupported operand type(s) for +: 'int' and 'Const'

This is a result of Python's binary operator rules, which will use the type
of the leftmost operand to determine the operation. In this case, `1` is an
`int`, so Python will try to use the `int`'s `__add__` method, instead of
Mahonia's `__add__` method. See `BinaryOperationOverloads` for more details.

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

from dataclasses import dataclass
from difflib import get_close_matches
from typing import (
	Any,
	ClassVar,
	Final,
	Generic,
	NamedTuple,
	Protocol,
	Self,
	TypeVar,
	overload,
	runtime_checkable,
)

T = TypeVar("T")
"""The type of the expression's value."""


class ContextProtocol(Protocol):
	def __getattribute__(self, name: str, /) -> Any: ...


S = TypeVar("S", bound=ContextProtocol)
"""The type of the expression's context."""

S_contra = TypeVar("S_contra", bound=ContextProtocol, contravariant=True)
"""The contravariant type of the expression's context."""


class _SupportsArithmetic(Protocol):
	def __add__(self, other: Any, /) -> Any: ...
	def __sub__(self, other: Any, /) -> Any: ...
	def __mul__(self, other: Any, /) -> Any: ...
	def __truediv__(self, other: Any, /) -> Any: ...
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


@runtime_checkable
class Eval(Protocol[T, S_contra]):
	"""Protocol for objects that can evaluate themselves with a context.

	This protocol defines the interface for expression evaluation. All Mahonia
	expressions implement this protocol.

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

	def eval(self, ctx: S_contra) -> "Const[T]": ...


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


class BoundExpr(NamedTuple, Generic[T, S]):
	"""An immutable expression bound to a specific context.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> x = Var[int, Ctx]("x")
	>>> bound = (x > 5).bind(Ctx(x=10))
	>>> bound.unwrap()
	True
	>>> str(bound)
	'(x:10 > 5 -> True)'
	"""

	expr: "Expr[T, S]"
	ctx: S

	def unwrap(self) -> T:
		return self.expr.unwrap(self.ctx)

	def __str__(self) -> str:
		return self.expr.to_string(self.ctx)


@runtime_checkable
class Expr(Protocol[T, S]):
	"""Base class for all Mahonia expressions.

	Provides core evaluation, serialization, and binding functionality.
	Users typically work with concrete expression types like Var, Const, Add, etc.

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

	def eval(self, ctx: S) -> "Const[T]": ...

	def to_string(self, ctx: S | None = None) -> str: ...

	def __call__(self, ctx: S) -> "Const[T]":
		return self.eval(ctx)

	def unwrap(self, ctx: S) -> T:
		return self.eval(ctx).value

	def bind(self, ctx: S) -> BoundExpr[T, S]:
		return BoundExpr(self, ctx)


@runtime_checkable
class BoolExpr(Expr[TSupportsLogic, S], Protocol[TSupportsLogic, S]):
	"""Base class for boolean expressions that support logical operations.

	Extends Expr with support for logical operators like & (and), | (or), and ~ (not).

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

	def eval(self, ctx: S) -> "Const[TSupportsLogic]": ...  # type: ignore[override]

	def __call__(self, ctx: S) -> "Const[TSupportsLogic]":
		return self.eval(ctx)


class UnaryOperationOverloads(Expr[bool, S]):
	def __invert__(self) -> "Not[S]":
		return Not(self)


class BooleanBinaryOperationOverloads(BoolExpr[TSupportsLogic, S]):
	def __and__(self, other: BoolExpr[TSupportsLogic, S]) -> "And[TSupportsLogic, S]":
		return And(self, other)

	def __or__(self, other: BoolExpr[TSupportsLogic, S]) -> "Or[TSupportsLogic, S]":
		return Or(self, other)

	def __invert__(self) -> "Not[S]":
		return Not(self)


class BinaryOperationOverloads(Expr[T, S]):
	@overload  # type: ignore[override]
	def __eq__(
		self, other: "ConstTolerance[TSupportsArithmetic]"
	) -> "Approximately[TSupportsArithmetic, S]": ...

	@overload  # type: ignore[override]
	def __eq__(self, other: Expr[TSupportsEquality, S]) -> "Eq[TSupportsEquality, S]": ...

	@overload  # type: ignore[override]
	def __eq__(self, other: TSupportsEquality) -> "Eq[TSupportsEquality, S]": ...

	def __eq__(  # type: ignore[misc]
		self,
		other: Expr[TSupportsEquality, S]
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
	def __ne__(self, other: Expr[T, S]) -> "Ne[T, S]": ...

	@overload  # type: ignore[override]
	def __ne__(self, other: T) -> "Ne[T, S]": ...

	def __ne__(self, other: Expr[T, S] | T) -> "Ne[T, S]":  # type: ignore[override]
		if isinstance(other, Expr):
			return Ne(self, other)  # type: ignore[arg-type]
		else:
			return Ne(self, Const[T](None, other))

	@overload
	def __lt__(self, other: TSupportsComparison) -> "Lt[TSupportsComparison, S]": ...

	@overload
	def __lt__(self, other: Expr[TSupportsComparison, S]) -> "Lt[TSupportsComparison, S]": ...

	def __lt__(
		self, other: Expr[TSupportsComparison, S] | TSupportsComparison
	) -> "Lt[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Lt(self, other)  # type: ignore[arg-type]
		else:
			return Lt(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __le__(self, other: TSupportsComparison) -> "Le[TSupportsComparison, S]": ...

	@overload
	def __le__(self, other: Expr[TSupportsComparison, S]) -> "Le[TSupportsComparison, S]": ...

	def __le__(
		self, other: Expr[TSupportsComparison, S] | TSupportsComparison
	) -> "Le[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Le(self, other)  # type: ignore[arg-type]
		else:
			return Le(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __gt__(self, other: TSupportsComparison) -> "Gt[TSupportsComparison,S]": ...

	@overload
	def __gt__(self, other: Expr[TSupportsComparison, S]) -> "Gt[TSupportsComparison, S]": ...

	def __gt__(
		self, other: Expr[TSupportsComparison, S] | TSupportsComparison
	) -> "Gt[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Gt(self, other)  # type: ignore[arg-type]
		else:
			return Gt(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __ge__(self, other: TSupportsComparison) -> "Ge[TSupportsComparison, S]": ...

	@overload
	def __ge__(self, other: Expr[TSupportsComparison, S]) -> "Ge[TSupportsComparison, S]": ...

	def __ge__(
		self, other: Expr[TSupportsComparison, S] | TSupportsComparison
	) -> "Ge[TSupportsComparison, S]":
		if isinstance(other, Expr):
			return Ge(self, other)  # type: ignore[arg-type]
		else:
			return Ge(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __add__(self, other: TSupportsArithmetic) -> "Add[TSupportsArithmetic, S]": ...

	@overload
	def __add__(self, other: Expr[TSupportsArithmetic, S]) -> "Add[TSupportsArithmetic, S]": ...

	def __add__(
		self, other: Expr[TSupportsArithmetic, S] | TSupportsArithmetic
	) -> "Add[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Add(self, other)  # type: ignore[arg-type]
		else:
			return Add(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __sub__(self, other: TSupportsArithmetic) -> "Sub[TSupportsArithmetic, S]": ...

	@overload
	def __sub__(self, other: Expr[TSupportsArithmetic, S]) -> "Sub[TSupportsArithmetic, S]": ...

	def __sub__(
		self, other: Expr[TSupportsArithmetic, S] | TSupportsArithmetic
	) -> "Sub[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Sub(self, other)  # type: ignore[arg-type]
		else:
			return Sub(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __mul__(self, other: TSupportsArithmetic) -> "Mul[TSupportsArithmetic, S]": ...

	@overload
	def __mul__(self, other: Expr[TSupportsArithmetic, S]) -> "Mul[TSupportsArithmetic, S]": ...

	def __mul__(
		self, other: Expr[TSupportsArithmetic, S] | TSupportsArithmetic
	) -> "Mul[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Mul(self, other)  # type: ignore[arg-type]
		else:
			return Mul(self, Const(None, other))  # type: ignore[arg-type]

	@overload
	def __truediv__(self, other: float) -> "Div[float, S]": ...

	@overload
	def __truediv__(self, other: Expr[float, S]) -> "Div[float, S]": ...

	def __truediv__(self, other: Expr[float, S] | float) -> "Div[float, S]":
		if isinstance(other, Expr):
			return Div(self, other)  # type: ignore[arg-type]
		else:
			return Div(self, Const[float](None, other))  # type: ignore[arg-type]

	@overload
	def __pow__(self, power: TSupportsArithmetic) -> "Pow[TSupportsArithmetic, S]": ...

	@overload
	def __pow__(self, power: Expr[TSupportsArithmetic, S]) -> "Pow[TSupportsArithmetic, S]": ...

	def __pow__(
		self, power: Expr[TSupportsArithmetic, S] | TSupportsArithmetic
	) -> "Pow[TSupportsArithmetic, S]":
		if isinstance(power, Expr):
			return Pow(self, power)  # type: ignore[arg-type]
		else:
			return Pow(self, Const(None, power))  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False, slots=True)
class Const(BinaryOperationOverloads[T, Any], BooleanBinaryOperationOverloads[T, Any]):  # type: ignore[type-var]
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

	def eval(self, ctx: Any) -> "Const[T]":
		return self

	def to_string(self, ctx: Any | None = None) -> str:
		return f"{self.name}:{self.value}" if self.name else str(self.value)


@dataclass(frozen=True, eq=False, slots=True)
class Var(BinaryOperationOverloads[T, S], BooleanBinaryOperationOverloads[T, S]):  # type: ignore[type-var]
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


@dataclass(frozen=True, eq=False, slots=True)
class UnaryOpEval(Eval[bool, S]):
	left: Expr[Any, S]


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


class Not(UnaryOpToString[S], UnaryOperationOverloads[S], BooleanBinaryOperationOverloads[bool, S]):
	op: ClassVar[str] = "not "

	def eval(self, ctx: S) -> Const[bool]:
		return Const(None, not self.left.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class BinaryOpEval(Expr[T, S], Generic[T, S]):
	left: Expr[T, S]
	right: Expr[T, S]


class BinaryOpToString(ToString[S], BinaryOpEval[T, S], Generic[T, S]):
	op: ClassVar[str] = " ? "
	template: ClassVar[str] = "({left}{op}{right})"
	template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		right: Final = self.right.to_string(ctx)
		if ctx is None:
			return self.template.format(left=left, op=self.op, right=right)
		else:
			return self.template_eval.format(
				left=left, op=self.op, right=right, out=self.eval(ctx).value
			)


class And(BinaryOpToString[TSupportsLogic, S], BooleanBinaryOperationOverloads[TSupportsLogic, S]):
	op: ClassVar[str] = " & "

	def eval(self, ctx: S) -> Const[TSupportsLogic]:
		return Const(None, self.left.eval(ctx).value and self.right.eval(ctx).value)


class Or(BinaryOpToString[TSupportsLogic, S], BooleanBinaryOperationOverloads[TSupportsLogic, S]):
	op: ClassVar[str] = " | "

	def eval(self, ctx: S) -> Const[TSupportsLogic]:
		return Const(None, self.left.eval(ctx).value or self.right.eval(ctx).value)


class Eq(
	BinaryOpToString[TSupportsEquality, S],
	BinaryOperationOverloads[TSupportsEquality, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " == "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value == self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Ne(
	BinaryOpToString[TSupportsEquality, S],
	BinaryOperationOverloads[TSupportsEquality, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " != "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value != self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Lt(
	BinaryOpToString[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " < "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value < self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Le(
	BinaryOpToString[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " <= "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value <= self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Gt(
	BinaryOpToString[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " > "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value > self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Ge(
	BinaryOpToString[TSupportsComparison, S],
	BinaryOperationOverloads[TSupportsComparison, S],
	BooleanBinaryOperationOverloads[Any, S],
):
	op: ClassVar[str] = " >= "

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(None, self.left.eval(ctx).value >= self.right.eval(ctx).value)

	def unwrap(self, ctx: S) -> bool:  # type: ignore[override]
		return self.eval(ctx).value


class Add(
	BinaryOpToString[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	op: ClassVar[str] = " + "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value + self.right.eval(ctx).value)


class Sub(
	BinaryOpToString[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	op: ClassVar[str] = " - "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value - self.right.eval(ctx).value)


class Mul(
	BinaryOpToString[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	op: ClassVar[str] = " * "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value * self.right.eval(ctx).value)


class Div(
	BinaryOpToString[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	op: ClassVar[str] = " / "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value / self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class Pow(
	BinaryOpToString[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	op: ClassVar[str] = "^"

	left: Expr[TSupportsArithmetic, S]
	right: Expr[TSupportsArithmetic, S]

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		return Const(None, self.left.eval(ctx).value ** self.right.eval(ctx).value)


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
		self, other: Expr[TSupportsArithmetic, S]
	) -> "Approximately[TSupportsArithmetic, S]": ...

	@overload  # type: ignore[override]
	def __eq__(self, other: TSupportsArithmetic) -> "Approximately[TSupportsArithmetic, Any]": ...

	def __eq__(  # type: ignore[misc]
		self, other: Expr[TSupportsArithmetic, S] | TSupportsArithmetic
	) -> "Approximately[TSupportsArithmetic, S]":
		if isinstance(other, Expr):
			return Approximately(other, self)  # type: ignore
		else:
			return Approximately(Const(None, other), self)  # type: ignore


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
		return self.value * self.percent / 100.0

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.percent}%"


@dataclass(frozen=True, eq=False, slots=True)
class Approximately(
	BinaryOpToString[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[Any, S],
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

	left: Expr[TSupportsArithmetic, S]
	right: ConstTolerance[TSupportsArithmetic]  # type: ignore[assignment]

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(
			None, abs(self.left.eval(ctx).value - self.right.value) <= self.right.max_abs_error
		)


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
	expr: BoolExpr[bool, S]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(self.name, self.expr.eval(ctx).value)

	def to_string(self, ctx: S | None = None) -> str:
		result: Final = (
			self.expr.to_string(ctx)
			if ctx is None
			else f"{self.unwrap(ctx)} {self.expr.to_string(ctx)}"
		)
		return f"{self.name}: {result}" if self.name else result
