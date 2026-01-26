# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
	from mahonia import Expr, S, SizedIterable


def format_iterable_var(expr: "Expr[SizedIterable[Any], S, Any]", ctx: "S | None") -> str:
	"""Format an iterable variable with compact container display.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], Ctx]("nums")
	>>> format_iterable_var(nums, None)
	'nums'
	>>> format_iterable_var(nums, Ctx(nums=[1, 2]))
	'nums:2[1,2]'
	>>> format_iterable_var(nums, Ctx(nums=[1, 2, 3, 4, 5]))
	'nums:5[1,..5]'
	"""
	if ctx is None:
		return expr.to_string(ctx)

	value: Final = expr.unwrap(ctx)

	if isinstance(value, (str, bytes)):
		return expr.to_string(ctx)

	length: Final = len(value)
	name: Final = getattr(expr, "name", None)
	prefix: Final = f"{name}:" if name else ""

	def _serialize_elem(elem: Any) -> str:
		return elem.to_string(ctx) if hasattr(elem, "to_string") else str(elem)

	# Handle different container types
	if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
		# Indexable sequences (list, tuple)
		if length <= 2:
			return f"{prefix}{length}[{','.join(_serialize_elem(elem) for elem in value)}]"
		else:
			return f"{prefix}{length}[{_serialize_elem(value[0])},..{_serialize_elem(value[-1])}]"
	else:
		# Sets, other iterables without indexing
		elements = list(value)
		if length <= 2:
			return f"{prefix}{length}[{','.join(_serialize_elem(elem) for elem in elements)}]"
		else:
			return f"{prefix}{length}[{_serialize_elem(elements[0])},..{_serialize_elem(elements[-1])}]"
