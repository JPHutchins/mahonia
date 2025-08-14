# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Statistical operations for Mahonia expressions.

This module provides statistical functions that operate on iterables within contexts,
useful for manufacturing quality control and batch analysis.
"""

import statistics
from dataclasses import dataclass
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar

from mahonia import BinaryOperationOverloads, Const, Expr, S


class _SizedIterable(Protocol):
	"""Protocol for iterables that support len()."""

	def __iter__(self): ...
	def __len__(self) -> int: ...


class _IterableFloat(Protocol):
	"""Protocol for iterables containing floats."""

	def __iter__(self) -> Iterator[float]: ...


TSizedIterable = TypeVar("TSizedIterable", bound=_SizedIterable)
TIterableFloat = TypeVar("TIterableFloat", bound=_IterableFloat)


@dataclass(frozen=True, eq=False, slots=True)
class UnaryStatisticalOp(Expr[float, S], Generic[TIterableFloat, S]):
	"""Base class for statistical operations on single expressions."""

	expr: Expr[TIterableFloat, S]
	op: ClassVar[str] = "stat"

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.op}({self.expr.to_string()})"
		else:
			result = self.eval(ctx).value
			return f"{self.op}({self.expr.to_string(ctx)} -> {result})"


class Mean(
	UnaryStatisticalOp[TIterableFloat, S],
	BinaryOperationOverloads[float, S],
	Generic[TIterableFloat, S],
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

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.mean(iterable))


class StdDev(
	UnaryStatisticalOp[TIterableFloat, S],
	BinaryOperationOverloads[float, S],
	Generic[TIterableFloat, S],
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

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.stdev(iterable))


class Median(
	UnaryStatisticalOp[TIterableFloat, S],
	BinaryOperationOverloads[float, S],
	Generic[TIterableFloat, S],
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

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.expr.eval(ctx).value
		return Const(None, statistics.median(iterable))


@dataclass(frozen=True, eq=False, slots=True)
class Percentile(BinaryOperationOverloads[float, S], Generic[TIterableFloat, S]):
	"""Percentile of an iterable expression.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Data(NamedTuple):
	... 	values: list[float]
	>>> values = Var[list[float], Data]("values")
	>>> p95_expr = Percentile(values, 95)
	>>> p95_expr.to_string()
	'percentile(values, 95)'
	>>> ctx = Data(values=[1.0, 2.0, 3.0, 4.0, 5.0])
	>>> p95_expr.unwrap(ctx)
	4.8
	"""

	expr: Expr[TIterableFloat, S]
	percentile: float

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.expr.eval(ctx).value
		values = sorted(iterable)
		n = len(values)

		if self.percentile == 100:
			return Const(None, values[-1])
		elif self.percentile == 0:
			return Const(None, values[0])

		# Use linear interpolation method
		index = (self.percentile / 100.0) * (n - 1)
		lower_index = int(index)
		upper_index = min(lower_index + 1, n - 1)

		if lower_index == upper_index:
			result = values[lower_index]
		else:
			weight = index - lower_index
			result = values[lower_index] * (1 - weight) + values[upper_index] * weight

		return Const(None, result)

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"percentile({self.expr.to_string()}, {self.percentile})"
		else:
			result = self.eval(ctx).value
			return f"percentile({self.expr.to_string(ctx)}, {self.percentile} -> {result})"


@dataclass(frozen=True, eq=False, slots=True)
class Range(BinaryOperationOverloads[float, S], Generic[TIterableFloat, S]):
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

	expr: Expr[TIterableFloat, S]

	def eval(self, ctx: S) -> Const[float]:
		iterable = self.expr.eval(ctx).value
		return Const(None, max(iterable) - min(iterable))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"range({self.expr.to_string()})"
		else:
			result = self.eval(ctx).value
			return f"range({self.expr.to_string(ctx)} -> {result})"


@dataclass(frozen=True, eq=False, slots=True)
class Count(BinaryOperationOverloads[int, S], Generic[TSizedIterable, S]):
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

	expr: Expr[TSizedIterable, S]

	def eval(self, ctx: S) -> Const[int]:
		iterable = self.expr.eval(ctx).value
		return Const(None, len(iterable))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"count({self.expr.to_string()})"
		else:
			result = self.eval(ctx).value
			return f"count({self.expr.to_string(ctx)} -> {result})"
