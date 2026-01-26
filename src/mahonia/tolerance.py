# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, Self, overload

from mahonia import (
	BinaryOp,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	S,
	SupportsArithmetic,
	TSupportsArithmetic,
)

if TYPE_CHECKING:
	pass


class ConstToleranceProtocol(Protocol):
	@property
	def max_abs_error(self) -> SupportsArithmetic: ...

	@property
	def tolerance_string(self) -> str:
		return f" ± {self.max_abs_error}"


class ConstTolerance(
	Const[SupportsArithmetic], ConstToleranceProtocol, Generic[TSupportsArithmetic]
):
	"""Base class for constants with tolerance for approximate comparisons.

	Tolerance constants automatically create Approximately expressions when
	compared with == to other values or expressions.

	>>> from mahonia import PlusMinus, Approximately
	>>> five_ish = PlusMinus("5ish", 5.0, plus_minus=0.1)
	>>> check = five_ish == 5.05
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
	>>> from mahonia import Var, PlusMinus
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
	>>> from mahonia import Var, Percent
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
	>>> from mahonia import Var, PlusMinus, Approximately
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
