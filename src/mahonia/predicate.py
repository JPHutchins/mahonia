# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from mahonia import (
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	S,
	SizedIterable,
)

if TYPE_CHECKING:
	from mahonia import BoundExpr, Func, MapExpr


@dataclass(frozen=True, eq=False, slots=True)
class Predicate(BooleanBinaryOperationOverloads[bool, S]):
	"""A named predicate that evaluates to `True` or `False`.

	>>> from mahonia import Const, Predicate
	>>> my_predicate = Predicate("My Predicate", Const("two", 2) == 2)
	>>> my_predicate.to_string({})
	'My Predicate: True (two:2 == 2 -> True)'

	>>> from typing import NamedTuple
	>>> from mahonia import Var
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
	expr: Expr[Any, S, bool]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(self.name, self.expr.eval(ctx).value)

	def __call__(self, ctx: S) -> Const[bool]:
		return self.eval(ctx)

	def unwrap(self, ctx: S) -> bool:
		return self.eval(ctx).value

	def bind(self, ctx: S) -> "BoundExpr[bool, S, bool]":
		from mahonia import BoundExpr

		return BoundExpr(self, ctx)

	def to_string(self, ctx: S | None = None) -> str:
		result: Final = (
			self.expr.to_string(ctx)
			if ctx is None
			else f"{self.unwrap(ctx)} {self.expr.to_string(ctx)}"
		)
		return f"{self.name}: {result}" if self.name else result

	def partial(self, ctx: Any) -> "Predicate[Any]":
		return Predicate(self.name, self.expr.partial(ctx))

	def to_func(self) -> "Func[bool, S]":
		from mahonia import Func
		from mahonia.extract_vars import extract_vars

		return Func(extract_vars((), self), self)

	def map(self, container: "Expr[SizedIterable[Any], Any, Any]") -> "MapExpr[Any, bool, Any]":
		from mahonia import MapExpr

		return MapExpr(self.to_func(), container)
