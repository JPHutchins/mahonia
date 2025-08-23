# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Statistical operations for Mahonia expressions.

This module provides statistical functions that operate on iterables within contexts,
useful for manufacturing quality control and batch analysis.
"""

import statistics
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import (
	TYPE_CHECKING,
	Any,
	ClassVar,
	Final,
	Generic,
	Protocol,
	TypeVar,
)

from mahonia import BinaryOperationOverloads, Const, Eval, Expr, S, ToString, UnaryOpToString

if TYPE_CHECKING:
	from statistics import _NumberT
else:
	from statistics import Decimal, Fraction

	_NumberT = TypeVar("_NumberT", bound=float | Decimal | Fraction)  # type: ignore[misc]

T_co = TypeVar("T_co", covariant=True)


class SizedIterable(Sized, Iterable[T_co], Protocol[T_co]):
	def __getitem__(self, index: int, /) -> T_co: ...


@dataclass(frozen=True, eq=False, slots=True)
class UnaryStatisticalOpEval(Eval["_NumberT", S], Generic[_NumberT, S]):
	"""Base evaluation class for unary statistical operations."""

	left: Expr[SizedIterable["_NumberT"], S]


class UnaryStatisticalOpToString(
	ToString[S], UnaryStatisticalOpEval["_NumberT", S], Generic[_NumberT, S]
):
	"""String formatting mixin for unary statistical operations."""

	op: ClassVar[str] = "stat"
	template: ClassVar[str] = "{op}({left})"
	template_eval: ClassVar[str] = "{op}({left} -> {out})"

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = _format_iterable_var(self.left, ctx)
		if ctx is None:
			return self.template.format(op=self.op, left=left)
		else:
			return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


class UnaryStatisticalOp(
	UnaryStatisticalOpToString["_NumberT", S],
	BinaryOperationOverloads["_NumberT", S],
	Generic[_NumberT, S],
):
	"""Base class for unary statistical operations that take a container of numbers and return a single number."""

	op: ClassVar[str] = "stat"


@dataclass(frozen=True, eq=False, slots=True)
class Mean(
	UnaryStatisticalOpToString["_NumberT", S],
	BinaryOperationOverloads["_NumberT", S],
	Generic[_NumberT, S],
):
	"""Arithmetic mean of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> mean_expr = Mean(values)  # User can provide any iterable
	>>> mean_expr.to_string()
	'mean(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> mean_expr.unwrap(ctx)
	3.0
	>>> mean_expr.to_string(ctx)
	'mean(values:5[1.0,..5.0] -> 3.0)'
	"""

	op: ClassVar[str] = "mean"
	left: Expr[SizedIterable["_NumberT"], S]

	def eval(self, ctx: S) -> Const["_NumberT"]:
		return Const(None, statistics.mean(self.left.unwrap(ctx)))


@dataclass(frozen=True, eq=False, slots=True)
class StdDev(
	UnaryStatisticalOpToString["_NumberT", S],
	BinaryOperationOverloads["_NumberT", S],
	Generic[_NumberT, S],
):
	"""Standard deviation of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> std_expr = StdDev(values)
	>>> std_expr.to_string()
	'stddev(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> round(std_expr.unwrap(ctx), 3)
	1.581
	"""

	op: ClassVar[str] = "stddev"
	left: Expr[SizedIterable["_NumberT"], S]

	def eval(self, ctx: S) -> Const["_NumberT"]:
		return Const(None, statistics.stdev(self.left.unwrap(ctx)))


@dataclass(frozen=True, eq=False, slots=True)
class Median(
	UnaryStatisticalOpToString["_NumberT", S],
	BinaryOperationOverloads["_NumberT", S],
	Generic[_NumberT, S],
):
	"""Median of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> median_expr = Median(values)
	>>> median_expr.to_string()
	'median(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> median_expr.unwrap(ctx)
	3.0
	"""

	op: ClassVar[str] = "median"
	left: Expr[SizedIterable["_NumberT"], S]

	def eval(self, ctx: S) -> Const["_NumberT"]:
		return Const(None, statistics.median(self.left.unwrap(ctx)))


@dataclass(frozen=True, eq=False, slots=True)
class Percentile(BinaryOperationOverloads[float, S]):
	"""Percentile of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> p95_expr = Percentile(95, values)
	>>> p95_expr.to_string()
	'percentile:95(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> p95_expr.unwrap(ctx)
	4.8
	"""

	op: ClassVar[str] = "percentile"

	percentile: float
	left: Expr[SizedIterable[float], S]

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.left.unwrap(ctx)
		# Convert to tuple to make a copy and prevent mutation, then sort
		values = sorted(tuple(iterable))
		n = len(values)

		if self.percentile == 100:
			return Const(None, float(values[-1]))
		elif self.percentile == 0:
			return Const(None, float(values[0]))

		# Use linear interpolation method
		index = (self.percentile / 100.0) * (n - 1)
		lower_index = int(index)
		upper_index = min(lower_index + 1, n - 1)

		if lower_index == upper_index:
			return Const(None, float(values[lower_index]))
		else:
			weight = index - lower_index
			return Const(
				None, float(values[lower_index] * (1 - weight) + values[upper_index] * weight)
			)

	def to_string(self, ctx: S | None = None) -> str:
		left_str = _format_iterable_var(self.left, ctx)
		if ctx is None:
			return f"{self.op}:{self.percentile}({left_str})"
		else:
			return f"{self.op}:{self.percentile}({left_str} -> {self.unwrap(ctx)})"


@dataclass(frozen=True, eq=False, slots=True)
class Range(UnaryStatisticalOpToString[float, S], BinaryOperationOverloads[float, S]):
	"""Range (max - min) of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> range_expr = Range(values)
	>>> range_expr.to_string()
	'range(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> range_expr.unwrap(ctx)
	4.0
	"""

	op: ClassVar[str] = "range"

	left: Expr[SizedIterable[float], S]

	def eval(self, ctx: S) -> Const[float]:
		left = self.left.unwrap(ctx)
		return Const(None, float(max(left) - min(left)))


@dataclass(frozen=True, eq=False, slots=True)
class Count(UnaryOpToString[S], BinaryOperationOverloads[int, S], Generic[S]):
	"""Count of elements in an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> count_expr = Count(values)
	>>> count_expr.to_string()
	'count(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> count_expr.unwrap(ctx)
	5
	"""

	op: ClassVar[str] = "count"
	template: ClassVar[str] = "{op}({left})"
	template_eval: ClassVar[str] = "{op}({left}) -> {out}"

	left: Expr[SizedIterable[Any], S]

	def eval(self, ctx: S) -> Const[int]:  # type: ignore[override]
		return Const(None, len(self.left.unwrap(ctx)))

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = _format_iterable_var(self.left, ctx)  # type: ignore[arg-type]
		if ctx is None:
			return self.template.format(op=self.op, left=left)
		else:
			return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


def _format_iterable_var(expr: Expr[SizedIterable[Any], S], ctx: S | None) -> str:
	"""Format an iterable variable with custom container display logic."""
	if ctx is None:
		return expr.to_string(ctx)

	value: Final = expr.unwrap(ctx)

	if isinstance(value, (str, bytes)):
		return expr.to_string(ctx)

	length: Final = len(value)
	name: Final = getattr(expr, "name", None)
	prefix: Final = f"{name}:" if name else ""

	if length <= 2:
		return f"{prefix}{length}[{','.join(str(elem) for elem in value)}]"
	else:
		return f"{prefix}{length}[{value[0]},..{value[-1]}]"
