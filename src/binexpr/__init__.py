# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
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

T_co = TypeVar("T_co", covariant=True)
"""The covariant type of the expression's value."""

S = TypeVar("S")
"""The type of the expression's context."""

S_contra = TypeVar("S_contra", contravariant=True)
"""The contravariant type of the expression's context."""


class _SupportsArithmetic(Protocol):
	def __add__(self, other: Any, /) -> Any: ...
	def __sub__(self, other: Any, /) -> Any: ...
	def __mul__(self, other: Any, /) -> Any: ...
	def __truediv__(self, other: Any, /) -> Any: ...
	def __abs__(self, /) -> Any: ...


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


@runtime_checkable
class Eval(Protocol[T, S_contra]):
	def eval(self, ctx: S_contra) -> "Const[T]": ...


@runtime_checkable
class ToString(Protocol[S_contra]):
	def to_string(self, ctx: S_contra | None = None) -> str: ...


class BoundExpr(NamedTuple, Generic[T, S]):
	expr: "Expr[T, S]"
	ctx: S

	def unwrap(self) -> T:
		return self.expr.unwrap(self.ctx)

	def __str__(self) -> str:
		return self.expr.to_string(self.ctx)


class Expr(Generic[T, S]):
	def eval(self, ctx: S) -> "Const[T]":
		raise NotImplementedError()

	def to_string(self, ctx: S | None = None) -> str:
		raise NotImplementedError()

	def __call__(self, ctx: S) -> "Const[T]":
		return self.eval(ctx)

	def unwrap(self, ctx: S) -> T:
		return self.eval(ctx).value

	def bind(self, ctx: S) -> BoundExpr[T, S]:
		return BoundExpr(self, ctx)


class BoolExpr(Expr[TSupportsLogic, S]):
	def eval(self, ctx: S) -> "Const[TSupportsLogic]":
		raise NotImplementedError()

	def to_string(self, ctx: S | None = None) -> str:
		return super().to_string(ctx)

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
	def __eq__(self, other: Expr[TSupportsEquality, S]) -> "Eq[TSupportsEquality, S]": ...

	@overload  # type: ignore[override]
	def __eq__(self, other: TSupportsEquality) -> "Eq[TSupportsEquality, S]": ...

	def __eq__(  # type: ignore[misc]
		self, other: Expr[TSupportsEquality, S] | TSupportsEquality
	) -> "Eq[TSupportsEquality, S]":
		if isinstance(other, Expr):
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


@dataclass(frozen=True, eq=False)
class Const(BinaryOperationOverloads[T, Any], BooleanBinaryOperationOverloads[T, Any]):  # type: ignore[type-var]
	name: str | None
	value: T

	def eval(self, ctx: Any) -> "Const[T]":
		return self

	def to_string(self, ctx: Any | None = None) -> str:
		return f"{self.name}:{self.value}" if self.name else str(self.value)


@dataclass(frozen=True, eq=False)
class Var(BinaryOperationOverloads[T, S], BooleanBinaryOperationOverloads[T, S]):  # type: ignore[type-var]
	name: str

	def eval(self, ctx: S) -> Const[T]:
		return Const(self.name, getattr(ctx, self.name))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return self.name
		else:
			return f"{self.name}:{self.eval(ctx).value}"


@dataclass(frozen=True, eq=False)
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


@dataclass(frozen=True, eq=False)
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


class ConstToleranceProtocol(Protocol):
	@property
	def max_abs_error(self) -> _SupportsArithmetic: ...

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.max_abs_error}"


class ConstTolerence(Const[_SupportsArithmetic], ConstToleranceProtocol):
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


@dataclass(frozen=True, eq=False)
class PlusMinus(ConstTolerence, Generic[TSupportsArithmetic]):
	name: str | None
	value: TSupportsArithmetic
	plus_minus: TSupportsArithmetic

	@property
	def max_abs_error(self) -> TSupportsArithmetic:
		return self.plus_minus


@dataclass(frozen=True, eq=False)
class Percent(ConstTolerence, Generic[TSupportsArithmetic]):
	name: str | None
	value: TSupportsArithmetic
	percent: float

	@property
	def max_abs_error(self) -> TSupportsArithmetic:
		return self.value * self.percent / 100.0

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.percent}%"


@dataclass(frozen=True, eq=False)
class Approximately(
	BinaryOpToString[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " ≈ "

	left: Expr[TSupportsArithmetic, S]
	right: ConstTolerence  # type: ignore[assignment]

	def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
		return Const(
			None, abs(self.left.eval(ctx).value - self.right.value) <= self.right.max_abs_error
		)


@dataclass(frozen=True, eq=False)
class Predicate(BooleanBinaryOperationOverloads[bool, S]):
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
