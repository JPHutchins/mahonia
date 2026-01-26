# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Generic

from mahonia import (
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	S,
	TSupportsArithmetic,
	TSupportsComparison,
	UnaryToStringMixin,
)


@dataclass(frozen=True, eq=False, slots=True)
class Abs(
	UnaryToStringMixin[TSupportsArithmetic, S],
	BinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	"""Absolute value expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
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
	>>> from mahonia import Var
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
	>>> from mahonia import Var
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
