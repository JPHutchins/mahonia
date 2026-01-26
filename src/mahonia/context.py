# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, overload

from mahonia.types import MergeContextProtocol

if TYPE_CHECKING:
	from mahonia import Var


def merge[*Ss](*contexts: *Ss) -> MergeContextProtocol[*Ss]:
	"""Merge values from multiple context instances into a single namespace.

	>>> from typing import NamedTuple
	>>> class A(NamedTuple):
	... 	a: int
	>>> class B(NamedTuple):
	... 	b: int
	>>> ctx = merge(A(a=1), B(b=2))
	>>> ctx.a
	1
	>>> ctx.b
	2
	"""

	values: dict[str, Any] = {}
	for context in contexts:
		# https://typing.python.org/en/latest/spec/generics.html#variance-type-constraints-and-type-bounds-not-yet-supported
		annotations: dict[str, Any] = getattr(type(context), "__annotations__")
		for field_name in annotations:
			if field_name in values:
				raise TypeError(f"Duplicate field '{field_name}' found when merging contexts.")
			values[field_name] = getattr(context, field_name)
	return SimpleNamespace(**values)


@overload
def context_vars[T1](
	f1: tuple[str, type[T1]], /
) -> tuple[Callable[[T1], tuple[T1]], "Var[T1, tuple[T1]]"]: ...
@overload
def context_vars[T1, T2](
	f1: tuple[str, type[T1]], f2: tuple[str, type[T2]], /
) -> tuple[
	Callable[[T1, T2], tuple[T1, T2]], "Var[T1, tuple[T1, T2]]", "Var[T2, tuple[T1, T2]]"
]: ...
@overload
def context_vars[T1, T2, T3](
	f1: tuple[str, type[T1]], f2: tuple[str, type[T2]], f3: tuple[str, type[T3]], /
) -> tuple[
	Callable[[T1, T2, T3], tuple[T1, T2, T3]],
	"Var[T1, tuple[T1, T2, T3]]",
	"Var[T2, tuple[T1, T2, T3]]",
	"Var[T3, tuple[T1, T2, T3]]",
]: ...
@overload
def context_vars[T1, T2, T3, T4](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4], tuple[T1, T2, T3, T4]],
	"Var[T1, tuple[T1, T2, T3, T4]]",
	"Var[T2, tuple[T1, T2, T3, T4]]",
	"Var[T3, tuple[T1, T2, T3, T4]]",
	"Var[T4, tuple[T1, T2, T3, T4]]",
]: ...
@overload
def context_vars[T1, T2, T3, T4, T5](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	f5: tuple[str, type[T5]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4, T5], tuple[T1, T2, T3, T4, T5]],
	"Var[T1, tuple[T1, T2, T3, T4, T5]]",
	"Var[T2, tuple[T1, T2, T3, T4, T5]]",
	"Var[T3, tuple[T1, T2, T3, T4, T5]]",
	"Var[T4, tuple[T1, T2, T3, T4, T5]]",
	"Var[T5, tuple[T1, T2, T3, T4, T5]]",
]: ...
@overload
def context_vars[T1, T2, T3, T4, T5, T6](
	f1: tuple[str, type[T1]],
	f2: tuple[str, type[T2]],
	f3: tuple[str, type[T3]],
	f4: tuple[str, type[T4]],
	f5: tuple[str, type[T5]],
	f6: tuple[str, type[T6]],
	/,
) -> tuple[
	Callable[[T1, T2, T3, T4, T5, T6], tuple[T1, T2, T3, T4, T5, T6]],
	"Var[T1, tuple[T1, T2, T3, T4, T5, T6]]",
	"Var[T2, tuple[T1, T2, T3, T4, T5, T6]]",
	"Var[T3, tuple[T1, T2, T3, T4, T5, T6]]",
	"Var[T4, tuple[T1, T2, T3, T4, T5, T6]]",
	"Var[T5, tuple[T1, T2, T3, T4, T5, T6]]",
	"Var[T6, tuple[T1, T2, T3, T4, T5, T6]]",
]: ...


def context_vars(*fields: tuple[str, type[Any]]) -> Any:
	"""Create a context class and matching Var instances with preserved types.

	Takes (name, type) tuples and returns a tuple of (ContextClass, var1, var2, ...).
	Both Var types and the context type are preserved by the type checker for up to 6 fields.
	The context is typed as a Callable that returns a tuple of the field types.

	>>> Ctx, x, y = context_vars(("x", int), ("y", float))
	>>> expr = x + y
	>>> expr.to_string()
	'(x + y)'
	>>> expr.unwrap(Ctx(x=1, y=2.5))
	3.5
	>>> expr.to_string(Ctx(x=1, y=2.5))
	'(x:1 + y:2.5 -> 3.5)'

	>>> Measurements, voltage, current = context_vars(
	... 	("voltage", float),
	... 	("current", float),
	... )
	>>> power = voltage * current
	>>> power.unwrap(Measurements(voltage=5.0, current=2.0))
	10.0
	"""
	from typing import NamedTuple as NT

	from mahonia import Var

	context_class = NT("Ctx", list(fields))  # type: ignore[misc]
	return (context_class,) + tuple(Var(name) for name, _ in fields)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
