# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Statistical operations for Mahonia expressions.

This module provides statistical functions that operate on iterables within contexts,
useful for manufacturing quality control and batch analysis.
"""

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, TypeAlias, TypeVar

from mahonia import BinaryOperationOverloads, Const, Expr, S, TSupportsArithmetic

_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)


class SupportsDunderLT(Protocol[_T_contra]):
	def __lt__(self, other: _T_contra, /) -> bool: ...


class SupportsDunderGT(Protocol[_T_contra]):
	def __gt__(self, other: _T_contra, /) -> bool: ...


SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]
SupportsRichComparisonT = TypeVar("SupportsRichComparisonT", bound=SupportsRichComparison)  # noqa: Y001


class SupportsSub(Protocol[_T_contra, _T_co]):
	def __sub__(self, x: _T_contra, /) -> _T_co: ...


class _Sized(Protocol):
	def __len__(self) -> int: ...


TSized = TypeVar("TSized", bound=_Sized)


@dataclass(frozen=True, eq=False, slots=True)
class UnaryStatisticalOp(Expr[TSupportsArithmetic, S]):
	"""Base class for statistical operations on single expressions."""

	expr: Expr[Iterable[TSupportsArithmetic], S]
	op: ClassVar[str] = "stat"

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.op}({self.expr.to_string()})"
		else:
			return f"{self.op}({self.expr.to_string(ctx)} -> {self.eval(ctx).value})"


class Mean(
	UnaryStatisticalOp[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
):
	"""Arithmetic mean of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> mean_expr = Mean(values)
	>>> mean_expr.to_string()
	'mean(values)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> mean_expr.unwrap(ctx)
	3.0
	>>> mean_expr.to_string(ctx)
	'mean(values:[1.0, 2.0, 3.0, 4.0, 5.0] -> 3.0)'
	"""

	op: ClassVar[str] = "mean"

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.mean(iterable))


class StdDev(
	UnaryStatisticalOp[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
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

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.stdev(iterable))


class Median(
	UnaryStatisticalOp[TSupportsArithmetic, S], BinaryOperationOverloads[TSupportsArithmetic, S]
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

	def eval(self, ctx: S) -> Const[TSupportsArithmetic]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.median(iterable))


@dataclass(frozen=True, eq=False, slots=True)
class Percentile(BinaryOperationOverloads[TSupportsArithmetic, S]):
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
	expr: Expr[Iterable[TSupportsArithmetic], S]

	def eval(self, ctx: S) -> Const[float]:  # type: ignore[override]
		values = sorted(self.expr.unwrap(ctx))
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
		if ctx is None:
			return f"{self.op}:{self.percentile}({self.expr.to_string()})"
		else:
			return f"{self.op}:{self.percentile}({self.expr.to_string(ctx)} -> {self.unwrap(ctx)})"


@dataclass(frozen=True, eq=False, slots=True)
class Range(
	BinaryOperationOverloads[SupportsSub[SupportsRichComparisonT, SupportsRichComparisonT], S],
):
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

	expr: Expr[Iterable[SupportsSub[SupportsRichComparisonT, SupportsRichComparisonT]], S]

	def eval(self, ctx: S) -> Const[SupportsRichComparisonT]:
		iterable = self.expr.unwrap(ctx)
		return Const(None, max(iterable) - min(iterable))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.op}({self.expr.to_string()})"
		else:
			return f"{self.op}({self.expr.to_string(ctx)} -> {self.eval(ctx).value})"


@dataclass(frozen=True, eq=False, slots=True)
class Count(BinaryOperationOverloads[TSupportsArithmetic, S]):
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

	expr: Expr[_Sized, S]

	def eval(self, ctx: S) -> Const[int]:  # type: ignore[override]
		return Const(None, len(self.expr.unwrap(ctx)))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"count({self.expr.to_string()})"
		else:
			return f"count({self.expr.to_string(ctx)} -> {self.unwrap(ctx)})"
