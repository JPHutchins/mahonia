# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from mahonia import (
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	EvalError,
	Expr,
	S,
	T,
)


@dataclass(frozen=True, eq=False, slots=True)
class Var(BinaryOperationOverloads[T, S], BooleanBinaryOperationOverloads[T, S]):
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

	def partial(self, ctx: Any) -> "Expr[T, Any, T]":
		if hasattr(ctx, self.name):
			return Const(self.name, getattr(ctx, self.name))
		return self
