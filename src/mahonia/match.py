# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, overload

from mahonia import (
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	S,
	T,
)


@dataclass(frozen=True, eq=False, slots=True)
class MatchExpr(
	BinaryOperationOverloads[T, S],
	BooleanBinaryOperationOverloads[T, S],
):
	"""Multi-branch conditional expression: match first true condition.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> from mahonia import Var, Const
	>>> x = Var[int, Ctx]("x")
	>>> match_expr = MatchExpr(
	... 	(x > 10, Const("large", "large")),
	... 	(x > 5, Const("medium", "medium")),
	... 	default=Const("small", "small"),
	... )
	>>> match_expr.to_string()
	'(match (x > 10 -> large:large) (x > 5 -> medium:medium) else small:small)'
	>>> match_expr.unwrap(Ctx(x=15))
	'large'
	>>> match_expr.unwrap(Ctx(x=7))
	'medium'
	>>> match_expr.unwrap(Ctx(x=3))
	'small'
	>>> match_expr.to_string(Ctx(x=7))
	'(match (x:7 > 10 -> False -> large:large) (x:7 > 5 -> True -> medium:medium) else small:small -> medium)'
	"""

	branches: tuple[tuple[Expr[Any, S, bool], Expr[Any, S, Any]], ...]
	default: Expr[Any, S, Any]

	def __init__(
		self,
		*branches: tuple[Expr[Any, S, bool], Expr[Any, S, Any]],
		default: Expr[Any, S, Any],
	) -> None:
		object.__setattr__(self, "branches", branches)
		object.__setattr__(self, "default", default)

	def eval(self, ctx: S) -> Const[T]:
		for condition, value in self.branches:
			if condition.eval(ctx).value:
				return value.eval(ctx)
		return self.default.eval(ctx)

	def unwrap(self, ctx: S) -> T:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		def strip_outer_parens(s: str) -> str:
			return s[1:-1] if s.startswith("(") and s.endswith(")") else s

		if ctx is None:
			branches_str = " ".join(
				f"({strip_outer_parens(cond.to_string())} -> {val.to_string()})"
				for cond, val in self.branches
			)
			return f"(match {branches_str} else {self.default.to_string()})"
		branches_str = " ".join(
			f"({strip_outer_parens(cond.to_string(ctx))} -> {val.to_string()})"
			for cond, val in self.branches
		)
		return f"(match {branches_str} else {self.default.to_string()} -> {self.eval(ctx).value})"

	def partial(self, ctx: Any) -> "Expr[T, Any, T]":
		return MatchExpr(
			*((cond.partial(ctx), val.partial(ctx)) for cond, val in self.branches),
			default=self.default.partial(ctx),
		)


@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	branch5: tuple[Expr[Any, S, bool], Const[R5]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	branch5: tuple[Expr[Any, S, bool], Const[R5]],
	branch6: tuple[Expr[Any, S, bool], Const[R6]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	branch5: tuple[Expr[Any, S, bool], Const[R5]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], Const[R1]],
	branch2: tuple[Expr[Any, S, bool], Const[R2]],
	branch3: tuple[Expr[Any, S, bool], Const[R3]],
	branch4: tuple[Expr[Any, S, bool], Const[R4]],
	branch5: tuple[Expr[Any, S, bool], Const[R5]],
	branch6: tuple[Expr[Any, S, bool], Const[R6]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	branch6: tuple[Expr[Any, S, bool], Expr[Any, S, R6]],
	/,
	*,
	default: Const[D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	branch6: tuple[Expr[Any, S, bool], Expr[Any, S, R6]],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], Expr[Any, S, R1]],
	branch2: tuple[Expr[Any, S, bool], Expr[Any, S, R2]],
	branch3: tuple[Expr[Any, S, bool], Expr[Any, S, R3]],
	branch4: tuple[Expr[Any, S, bool], Expr[Any, S, R4]],
	branch5: tuple[Expr[Any, S, bool], Expr[Any, S, R5]],
	branch6: tuple[Expr[Any, S, bool], Expr[Any, S, R6]],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	branch5: tuple[Expr[Any, S, bool], R5],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	branch5: tuple[Expr[Any, S, bool], R5],
	branch6: tuple[Expr[Any, S, bool], R6],
	/,
	*,
	default: Expr[Any, S, D],
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
@overload
def Match[R1, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | D, S]: ...
@overload
def Match[R1, R2, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | D, S]: ...
@overload
def Match[R1, R2, R3, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	branch5: tuple[Expr[Any, S, bool], R5],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | D, S]: ...
@overload
def Match[R1, R2, R3, R4, R5, R6, D, S](
	branch1: tuple[Expr[Any, S, bool], R1],
	branch2: tuple[Expr[Any, S, bool], R2],
	branch3: tuple[Expr[Any, S, bool], R3],
	branch4: tuple[Expr[Any, S, bool], R4],
	branch5: tuple[Expr[Any, S, bool], R5],
	branch6: tuple[Expr[Any, S, bool], R6],
	/,
	*,
	default: D,
) -> MatchExpr[R1 | R2 | R3 | R4 | R5 | R6 | D, S]: ...
def Match[S](
	*branches: tuple[Expr[Any, S, bool], Expr[Any, S, Any] | Any],
	default: Expr[Any, S, Any] | Any,
) -> MatchExpr[Any, S]:
	"""Create a multi-branch conditional MatchExpr with proper type inference.

	Branch values and default can be either Expr objects or plain values.
	Plain values are automatically coerced to Const.

	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	... 	x: int
	>>> from mahonia import Var, Const
	>>> x = Var[int, Ctx]("x")
	>>> match_expr = Match(
	... 	(x > 10, Const("large", "large")),
	... 	(x > 5, Const("medium", "medium")),
	... 	default=Const("small", "small"),
	... )
	>>> match_expr.to_string()
	'(match (x > 10 -> large:large) (x > 5 -> medium:medium) else small:small)'
	>>> match_expr.unwrap(Ctx(x=15))
	'large'
	>>> match_coerced = Match((x > 5, "high"), default="low")
	>>> match_coerced.unwrap(Ctx(x=10))
	'high'
	>>> match_coerced.unwrap(Ctx(x=3))
	'low'
	"""
	coerced_branches: tuple[tuple[Expr[Any, S, bool], Expr[Any, S, Any]], ...] = tuple(  # pyright: ignore[reportUnknownVariableType]
		(cond, val if isinstance(val, Expr) else Const(None, val))
		for cond, val in branches  # pyright: ignore[reportUnknownArgumentType]
	)
	coerced_default: Expr[Any, S, Any] = (  # pyright: ignore[reportUnknownVariableType]
		default if isinstance(default, Expr) else Const(None, default)
	)
	return MatchExpr(*coerced_branches, default=coerced_default)
