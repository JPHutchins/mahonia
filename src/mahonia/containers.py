# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Generic, overload

from mahonia import (
	And,
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	Foldable,
	Func,
	Max,
	Min,
	Or,
	R,
	S,
	SizedIterable,
	T,
	TSupportsComparison,
	U,
)
from mahonia.formatting import format_iterable_var


@dataclass(frozen=True, eq=False, slots=True)
class Contains(
	BinaryOperationOverloads[bool, S],
	BooleanBinaryOperationOverloads[bool, S],
	Generic[T, S],
):
	"""Check if a value is contained in a collection.

	>>> from typing import NamedTuple
	>>> from mahonia import Var, SizedIterable
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	... 	target: int
	>>> values = Var[SizedIterable[int], Ctx]("values")
	>>> target = Var[int, Ctx]("target")
	>>> contains_expr = Contains(target, values)
	>>> contains_expr.to_string()
	'(target in values)'
	>>> contains_expr.unwrap(Ctx(values=[1, 2, 3], target=2))
	True
	>>> contains_expr.unwrap(Ctx(values=[1, 2, 3], target=5))
	False
	>>> contains_expr.to_string(Ctx(values=[1, 2, 3], target=2))
	'(target:2 in values:3[1,..3] -> True)'
	"""

	op: ClassVar[str] = " in "
	template: ClassVar[str] = "({left}{op}{right})"
	template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

	element: Expr[T, S, T]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> Const[bool]:
		return Const(None, self.element.unwrap(ctx) in self.container.unwrap(ctx))

	def unwrap(self, ctx: S) -> bool:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.element.to_string(ctx)
		right: Final = format_iterable_var(self.container, ctx)
		if ctx is None:
			return self.template.format(left=left, op=self.op, right=right)
		else:
			return self.template_eval.format(
				left=left, op=self.op, right=right, out=self.eval(ctx).value
			)

	def partial(self, ctx: Any) -> "Expr[bool, Any, bool]":
		return Contains(self.element.partial(ctx), self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class MapExpr(
	BinaryOperationOverloads[SizedIterable[U], S],
	Generic[T, U, S],
):
	"""Apply a function to each element in a container."""

	op: ClassVar[str] = "map"
	template: ClassVar[str] = "({op} {func} {container})"
	template_eval: ClassVar[str] = "({op} {func} {container} -> {out})"

	func: "Func[U, Any]"
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> "Const[SizedIterable[U]]":
		container_values = self.container.unwrap(ctx)
		result: list[U] = []
		for item in container_values:
			if self.func.args and hasattr(self.func.args[0], "name"):
				arg_name = self.func.args[0].name  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
				temp_ctx = type("TempCtx", (), {arg_name: item})()
				result.append(self.func.expr.unwrap(temp_ctx))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
		return Const(None, result)

	def unwrap(self, ctx: S) -> SizedIterable[U]:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		func_str = self.func.to_string()
		if ctx is None:
			container_str = self.container.to_string()
			return self.template.format(op=self.op, func=func_str, container=container_str)
		else:
			container_str = format_iterable_var(self.container, ctx)
			result_value = self.eval(ctx).value
			result_expr = Const(None, result_value)
			out_str = format_iterable_var(result_expr, ctx)
			return self.template_eval.format(
				op=self.op, func=func_str, container=container_str, out=out_str
			)

	def partial(self, ctx: Any) -> "Expr[SizedIterable[U], Any, SizedIterable[U]]":
		partial_func = Func(self.func.args, self.func.expr.partial(ctx))
		return MapExpr(partial_func, self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class FoldLExpr(
	BinaryOperationOverloads[R, S],
	Generic[T, S, R],
):
	op_cls: type[Foldable[Any, Any]]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]
	initial: R | None = None

	@overload
	def __init__(
		self: "FoldLExpr[Expr[Any, Any, R], S, R]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Expr[Any, Any, R]], S, SizedIterable[Expr[Any, Any, R]]],
		initial: None = None,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[Expr[Any, Any, R], S, R]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Expr[Any, Any, R]], S, SizedIterable[Expr[Any, Any, R]]],
		initial: R = ...,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[T, S, T]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[T], S, SizedIterable[T]],
		initial: None = None,
	) -> None: ...

	@overload
	def __init__(
		self: "FoldLExpr[T, S, T]",  # pyright: ignore[reportInvalidTypeVarUse]
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[T], S, SizedIterable[T]],
		initial: T = ...,
	) -> None: ...

	def __init__(
		self,
		op_cls: type[Foldable[Any, Any]],
		container: Expr[SizedIterable[Any], S, SizedIterable[Any]],
		initial: Any | None = None,
	) -> None:
		object.__setattr__(self, "op_cls", op_cls)
		object.__setattr__(self, "container", container)
		object.__setattr__(self, "initial", initial)

	def eval(self, ctx: S) -> Const[R]:
		result_value: R = self.initial if self.initial is not None else self.op_cls.identity_element  # pyright: ignore[reportGeneralTypeIssues]

		for item in self.container.unwrap(ctx):
			item_value: R = item.unwrap(ctx) if isinstance(item, Expr) else item  # type: ignore[assignment]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAssignmentType]
			result_value = self.op_cls.op_func(result_value, item_value)  # type: ignore[arg-type, assignment]  # pyright: ignore[reportArgumentType]

		return Const(None, result_value)

	def unwrap(self, ctx: S) -> R:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		op_str: Final = self.op_cls.op.strip()
		container_name: Final = (
			self.container.to_string()
			if ctx is None
			else self.container.to_string() + f":{len(self.container.unwrap(ctx))}"
		)
		if ctx is None:
			return f"(foldl {op_str} {container_name})"
		result = self.eval(ctx).value
		if self.op_cls in (Min, Max):
			return f"(foldl {op_str} {container_name} -> {result})"
		items = list(self.container.unwrap(ctx))
		initial_str = f"{str(self.initial)}{self.op_cls.op}" if self.initial is not None else ""
		items_str = self._format_items(items, op_str, ctx)
		return f"(foldl {op_str} {container_name} -> ({initial_str}{items_str}) -> {result})"

	@staticmethod
	def _format_items(items: list[Any], op: str, ctx: S) -> str:
		def serialize(item: Any) -> str:
			if isinstance(item, Expr):
				return item.to_string(ctx)  # pyright: ignore[reportUnknownMemberType]
			return str(item)

		return f" {op} ".join(serialize(i) for i in items)

	def partial(self, ctx: Any) -> "Expr[R, Any, R]":
		return FoldLExpr(self.op_cls, self.container.partial(ctx), self.initial)  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False, slots=True)
class BoolContainerFoldBase(
	BinaryOperationOverloads[bool, S],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str]
	op_cls: ClassVar[type[Foldable[Any, Any]]]
	template: ClassVar[str] = "({op} {left})"
	template_eval: ClassVar[str] = "({op} {left} -> {out})"

	container: Expr[SizedIterable[Any], S, SizedIterable[Any]]
	_foldl: FoldLExpr[bool, S, bool] = field(init=False)

	def __post_init__(self) -> None:
		object.__setattr__(self, "_foldl", FoldLExpr(self.op_cls, self.container))

	def eval(self, ctx: S) -> Const[bool]:
		return self._foldl.eval(ctx)

	def unwrap(self, ctx: S) -> bool:
		return self._foldl.unwrap(ctx)

	def to_string(self, ctx: S | None = None) -> str:
		from mahonia import Var

		if ctx is None:
			return self.template.format(op=self.op, left=format_iterable_var(self.container, ctx))
		left = (
			format_iterable_var(self.container, ctx)
			if isinstance(self.container, (Var, Const))
			else self.container.to_string(ctx)
		)
		return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class AnyExpr(BoolContainerFoldBase[S]):
	"""True if any element in the container is truthy.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	... 	values: list[bool]
	>>> values = Var[list[bool], Ctx]("values")
	>>> any_expr = AnyExpr(values)
	>>> any_expr.to_string()
	'(any values)'
	>>> any_expr.unwrap(Ctx(values=[False, True, False]))
	True
	>>> any_expr.unwrap(Ctx(values=[False, False, False]))
	False
	>>> any_expr.to_string(Ctx(values=[False, True, False]))
	'(any values:3[False,..False] -> True)'

	With complex expressions like MapExpr, shows the full evaluation trace:

	>>> class NumCtx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], NumCtx]("nums")
	>>> n = Var[int, NumCtx]("n")
	>>> gt_five = (n > 5).map(nums)
	>>> any_gt_five = AnyExpr(gt_five)
	>>> any_gt_five.to_string()
	'(any (map n -> (n > 5) nums))'
	>>> any_gt_five.to_string(NumCtx(nums=[3, 7, 2]))
	'(any (map n -> (n > 5) nums:3[3,..2] -> 3[False,..False]) -> True)'
	>>> any_gt_five.to_string(NumCtx(nums=[1, 2, 3]))
	'(any (map n -> (n > 5) nums:3[1,..3] -> 3[False,..False]) -> False)'
	"""

	op: ClassVar[str] = "any"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Or

	def partial(self, ctx: Any) -> "AnyExpr[Any]":
		return AnyExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class AllExpr(BoolContainerFoldBase[S]):
	"""True if all elements in the container are truthy.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	... 	values: list[bool]
	>>> values = Var[list[bool], Ctx]("values")
	>>> all_expr = AllExpr(values)
	>>> all_expr.to_string()
	'(all values)'
	>>> all_expr.unwrap(Ctx(values=[True, True, True]))
	True
	>>> all_expr.unwrap(Ctx(values=[True, False, True]))
	False
	>>> all_expr.to_string(Ctx(values=[True, False, True]))
	'(all values:3[True,..True] -> False)'

	With complex expressions like MapExpr, shows the full evaluation trace:

	>>> class NumCtx(NamedTuple):
	... 	nums: list[int]
	>>> nums = Var[list[int], NumCtx]("nums")
	>>> n = Var[int, NumCtx]("n")
	>>> lt_ten = (n < 10).map(nums)
	>>> all_lt_ten = AllExpr(lt_ten)
	>>> all_lt_ten.to_string()
	'(all (map n -> (n < 10) nums))'
	>>> all_lt_ten.to_string(NumCtx(nums=[3, 7, 2]))
	'(all (map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> True)'
	>>> all_lt_ten.to_string(NumCtx(nums=[3, 15, 2]))
	'(all (map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> False)'
	"""

	op: ClassVar[str] = "all"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = And

	def partial(self, ctx: Any) -> "AllExpr[Any]":
		return AllExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class ComparableContainerFoldBase(
	BinaryOperationOverloads[TSupportsComparison, S],
	Generic[TSupportsComparison, S],
):
	op: ClassVar[str]
	op_cls: ClassVar[type[Foldable[Any, Any]]]
	template: ClassVar[str] = "({op} {left})"
	template_eval: ClassVar[str] = "({op} {left} -> {out})"

	container: Expr[SizedIterable[TSupportsComparison], S, SizedIterable[TSupportsComparison]]
	_foldl: FoldLExpr[TSupportsComparison, S, TSupportsComparison] = field(init=False)

	def __post_init__(self) -> None:
		object.__setattr__(self, "_foldl", FoldLExpr(self.op_cls, self.container))

	def eval(self, ctx: S) -> Const[TSupportsComparison]:
		return self._foldl.eval(ctx)

	def unwrap(self, ctx: S) -> TSupportsComparison:
		return self._foldl.unwrap(ctx)

	def to_string(self, ctx: S | None = None) -> str:
		from mahonia import Var

		if ctx is None:
			return self.template.format(op=self.op, left=format_iterable_var(self.container, ctx))
		left = (
			format_iterable_var(self.container, ctx)
			if isinstance(self.container, (Var, Const))
			else self.container.to_string(ctx)
		)
		return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


@dataclass(frozen=True, eq=False, slots=True)
class MinExpr(ComparableContainerFoldBase[TSupportsComparison, S]):
	"""Minimum element in a container.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[list[int], Ctx]("values")
	>>> min_expr = MinExpr(values)
	>>> min_expr.to_string()
	'(min values)'
	>>> min_expr.unwrap(Ctx(values=[3, 1, 4, 1, 5]))
	1
	>>> min_expr.to_string(Ctx(values=[3, 1, 4]))
	'(min values:3[3,..4] -> 1)'
	"""

	op: ClassVar[str] = "min"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Min

	def partial(self, ctx: Any) -> "MinExpr[TSupportsComparison, Any]":
		return MinExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class MaxExpr(ComparableContainerFoldBase[TSupportsComparison, S]):
	"""Maximum element in a container.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[list[int], Ctx]("values")
	>>> max_expr = MaxExpr(values)
	>>> max_expr.to_string()
	'(max values)'
	>>> max_expr.unwrap(Ctx(values=[3, 1, 4, 1, 5]))
	5
	>>> max_expr.to_string(Ctx(values=[3, 1, 4]))
	'(max values:3[3,..4] -> 4)'
	"""

	op: ClassVar[str] = "max"
	op_cls: ClassVar[type[Foldable[Any, Any]]] = Max

	def partial(self, ctx: Any) -> "MaxExpr[TSupportsComparison, Any]":
		return MaxExpr(self.container.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class FilterExpr(
	BinaryOperationOverloads[SizedIterable[T], S],
	Generic[T, S],
):
	"""Filter container elements by predicate.

	>>> from typing import NamedTuple
	>>> from mahonia import Var, SizedIterable
	>>> class Ctx(NamedTuple):
	... 	values: list[int]
	>>> values = Var[SizedIterable[int], Ctx]("values")
	>>> x = Var[int, Ctx]("x")
	>>> is_positive = (x > 0).to_func()
	>>> filter_expr = FilterExpr(is_positive, values)
	>>> filter_expr.to_string()
	'(filter x -> (x > 0) values)'
	>>> filter_expr.unwrap(Ctx(values=[-1, 2, -3, 4, 5]))
	(2, 4, 5)
	>>> filter_expr.to_string(Ctx(values=[-1, 2, -3, 4, 5]))
	'(filter x -> (x > 0) values:5[-1,..5] -> 3[2,..5])'
	"""

	op: ClassVar[str] = "filter"
	template: ClassVar[str] = "({op} {func} {container})"
	template_eval: ClassVar[str] = "({op} {func} {container} -> {out})"

	predicate: Func[bool, Any]
	container: Expr[SizedIterable[T], S, SizedIterable[T]]

	def eval(self, ctx: S) -> Const[SizedIterable[T]]:
		predicate_results = MapExpr(self.predicate, self.container).unwrap(ctx)
		return Const(
			None,
			tuple(
				value
				for value, keep in zip(self.container.unwrap(ctx), predicate_results, strict=True)
				if keep
			),
		)

	def unwrap(self, ctx: S) -> SizedIterable[T]:
		return self.eval(ctx).value

	def to_string(self, ctx: S | None = None) -> str:
		func_str = self.predicate.to_string()
		if ctx is None:
			container_str = self.container.to_string()
			return self.template.format(op=self.op, func=func_str, container=container_str)
		else:
			container_str = format_iterable_var(self.container, ctx)
			result_value = self.eval(ctx).value
			result_expr = Const(None, result_value)
			out_str = format_iterable_var(result_expr, ctx)
			return self.template_eval.format(
				op=self.op, func=func_str, container=container_str, out=out_str
			)

	def partial(self, ctx: Any) -> "Expr[SizedIterable[T], Any, SizedIterable[T]]":
		partial_func = Func(self.predicate.args, self.predicate.expr.partial(ctx))
		return FilterExpr(partial_func, self.container.partial(ctx))
