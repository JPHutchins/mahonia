# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from mahonia import (
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	Func,
	R,
	S,
	SizedIterable,
	T,
)

if TYPE_CHECKING:
	from mahonia import MapExpr


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
	>>> from mahonia import Var
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
		from mahonia import MapExpr

		return MapExpr(self.to_func(), container)
