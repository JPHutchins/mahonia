# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test container operations."""

from typing import Any, NamedTuple, assert_type

import pytest

from mahonia import (
	Add,
	AllExpr,
	And,
	AnyExpr,
	Const,
	Contains,
	Expr,
	FoldLExpr,
	Max,
	MaxExpr,
	Min,
	MinExpr,
	Mul,
	Or,
	SizedIterable,
	Var,
)


class ContainerData(NamedTuple):
	values: list[int]
	targets: list[str]
	flags: list[bool]


def test_contains_basic() -> None:
	"""Test basic Contains operation."""
	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, False, True])

	values = Var[SizedIterable[int], ContainerData]("values")
	target_val = Const("target", 3)

	contains_expr = Contains(target_val, values)
	assert contains_expr.to_string() == "(target:3 in values)"
	assert contains_expr.unwrap(ctx) is True
	assert contains_expr.to_string(ctx) == "(target:3 in values:5[1,..5] -> True)"

	# Test with value not in container
	missing_val = Const("missing", 10)
	missing_expr = Contains(missing_val, values)
	assert missing_expr.unwrap(ctx) is False
	assert missing_expr.to_string(ctx) == "(missing:10 in values:5[1,..5] -> False)"


def test_contains_with_vars() -> None:
	"""Test Contains with variable elements."""

	# Create a context that has a target_value field
	class SearchCtx(NamedTuple):
		values: list[int]
		target_value: int

	search_ctx = SearchCtx(values=[1, 2, 3, 4, 5], target_value=3)
	search_values = Var[SizedIterable[int], SearchCtx]("values")
	target_var = Var[int, SearchCtx]("target_value")

	contains_expr = Contains(target_var, search_values)
	assert contains_expr.to_string() == "(target_value in values)"
	assert contains_expr.unwrap(search_ctx) is True


def test_any_expr() -> None:
	"""Test AnyExpr operation."""
	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, False, True])

	flags = Var[SizedIterable[bool], ContainerData]("flags")
	any_expr = AnyExpr(flags)

	assert any_expr.to_string() == "(any flags)"
	assert any_expr.unwrap(ctx) is True
	assert any_expr.to_string(ctx) == "(any flags:3[True,..True] -> True)"

	# Test with all False
	all_false_ctx = ContainerData(values=[1, 2, 3], targets=["a", "b"], flags=[False, False, False])
	assert any_expr.unwrap(all_false_ctx) is False
	assert any_expr.to_string(all_false_ctx) == "(any flags:3[False,..False] -> False)"


def test_all_expr() -> None:
	"""Test AllExpr operation."""
	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, True, True])

	flags = Var[SizedIterable[bool], ContainerData]("flags")
	all_expr = AllExpr(flags)

	assert all_expr.to_string() == "(all flags)"
	assert all_expr.unwrap(ctx) is True
	assert all_expr.to_string(ctx) == "(all flags:3[True,..True] -> True)"

	# Test with one False
	mixed_ctx = ContainerData(values=[1, 2, 3], targets=["a", "b"], flags=[True, False, True])
	assert all_expr.unwrap(mixed_ctx) is False
	assert all_expr.to_string(mixed_ctx) == "(all flags:3[True,..True] -> False)"


def test_empty_containers() -> None:
	"""Test container operations with empty containers."""
	empty_ctx = ContainerData(values=[], targets=[], flags=[])

	# Contains should work with empty containers
	values = Var[SizedIterable[int], ContainerData]("values")
	target = Const("target", 1)
	contains_expr = Contains(target, values)
	assert contains_expr.unwrap(empty_ctx) is False

	# Any of empty container should be False
	flags = Var[SizedIterable[bool], ContainerData]("flags")
	any_expr = AnyExpr(flags)
	assert any_expr.unwrap(empty_ctx) is False

	# All of empty container should be True
	all_expr = AllExpr(flags)
	assert all_expr.unwrap(empty_ctx) is True


def test_container_arithmetic() -> None:
	"""Test that container operations support arithmetic combinations."""
	ctx = ContainerData(values=[1, 2, 3], targets=["a", "b", "c"], flags=[True, False, True])

	values = Var[SizedIterable[int], ContainerData]("values")
	flags = Var[SizedIterable[bool], ContainerData]("flags")

	target = Const("target", 2)
	contains_expr = Contains(target, values)
	any_expr = AnyExpr(flags)

	# Test logical combination
	combined = contains_expr & any_expr
	assert combined.unwrap(ctx) is True

	# Test with negation
	not_contains = ~contains_expr
	assert not_contains.unwrap(ctx) is False


def test_container_types() -> None:
	"""Test that container operations have correct types."""
	values = Var[SizedIterable[int], ContainerData]("values")
	flags = Var[SizedIterable[bool], ContainerData]("flags")
	target = Const("target", 1)

	# Test Contains type
	contains_expr = Contains(target, values)
	assert_type(contains_expr, Contains[int, Any])

	# Test AnyExpr type
	any_expr = AnyExpr(flags)
	assert_type(any_expr, AnyExpr[ContainerData])

	# Test AllExpr type
	all_expr = AllExpr(flags)
	assert_type(all_expr, AllExpr[ContainerData])


def test_sized_iterable_protocol() -> None:
	"""Test that SizedIterable protocol works correctly."""
	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, False, True])

	# Verify that list satisfies SizedIterable
	values = Var[SizedIterable[int], ContainerData]("values")
	target = Const("target", 3)

	contains_expr = Contains(target, values)
	assert contains_expr.unwrap(ctx) is True


def test_container_integration_with_predicates() -> None:
	"""Test container operations integrated with predicates."""
	from mahonia import Predicate

	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, False, True])

	values = Var[SizedIterable[int], ContainerData]("values")
	flags = Var[SizedIterable[bool], ContainerData]("flags")
	target = Const("target", 3)

	# Create a predicate that combines container operations
	contains_target = Contains(target, values)
	has_any_flags = AnyExpr(flags)

	validation = Predicate("Contains target and has flags", contains_target & has_any_flags)

	assert validation.unwrap(ctx) is True
	assert "Contains target and has flags: True" in validation.to_string(ctx)


def test_explicit_contains_construction() -> None:
	"""Test explicit Contains construction since Python's 'in' operator doesn't work with lazy evaluation."""

	class CoerceCtx(NamedTuple):
		target: int
		values: list[int]

	coerce_ctx = CoerceCtx(target=3, values=[1, 2, 3, 4, 5])
	target = Var[int, CoerceCtx]("target")
	# Users must type containers as SizedIterable[T] for Contains to work
	container = Var[SizedIterable[int], CoerceCtx]("values")

	# Test explicit Contains construction
	contains_expr = Contains(target, container)
	assert contains_expr.unwrap(coerce_ctx) is True
	assert contains_expr.to_string() == "(target in values)"
	assert contains_expr.to_string(coerce_ctx) == "(target:3 in values:5[1,..5] -> True)"


def test_explicit_contains_types() -> None:
	"""Test that explicit Contains construction has proper types."""

	values = Var[SizedIterable[int], ContainerData]("values")
	target = Const("target", 1)

	# Test with variable containers
	contains_expr = Contains(target, values)
	assert_type(contains_expr, Contains[int, Any])  # Context type is Any due to Const

	# Test with literal element - wrap in Const
	contains_expr2 = Contains(Const("literal", 5), values)
	assert_type(contains_expr2, Contains[int, Any])  # Context type is Any due to Const


def test_iterable_serialization_with_expr_elements() -> None:
	"""Test that iterables containing Expr objects are serialized correctly."""
	from mahonia import format_iterable_var

	class ExprCtx(NamedTuple):
		exprs: list[Const[int]]

	# Create a list of Const expressions
	const_list = [Const("a", 1), Const("b", 2), Const("c", 3)]
	ctx = ExprCtx(exprs=const_list)

	exprs_var = Var[SizedIterable[Const[int]], ExprCtx]("exprs")

	# Test with 3 elements (should show first..last)
	result = format_iterable_var(exprs_var, ctx)
	assert result == "exprs:3[a:1,..c:3]"

	# Test with 2 elements (should show all)
	two_ctx = ExprCtx(exprs=[Const("x", 10), Const("y", 20)])
	result = format_iterable_var(exprs_var, two_ctx)
	assert result == "exprs:2[x:10,y:20]"

	# Test with 1 element
	one_ctx = ExprCtx(exprs=[Const("z", 99)])
	result = format_iterable_var(exprs_var, one_ctx)
	assert result == "exprs:1[z:99]"


def test_set_serialization() -> None:
	"""Test serialization of set iterables."""
	from mahonia import format_iterable_var

	class SetCtx(NamedTuple):
		number_set: set[int]
		string_set: set[str]

	ctx = SetCtx(number_set={1, 2, 3, 4, 5}, string_set={"a", "b"})

	# Sets are non-indexable, should be converted to list first
	numbers = Var[SizedIterable[int], SetCtx]("number_set")
	result = format_iterable_var(numbers, ctx)
	# Result should show length and format (order may vary for sets, but length is stable)
	assert result.startswith("number_set:5[")
	assert result.endswith("]")
	assert ".." in result  # Should use first..last format for >2 elements

	# Test small set
	strings = Var[SizedIterable[str], SetCtx]("string_set")
	result = format_iterable_var(strings, ctx)
	assert result.startswith("string_set:2[")
	# For 2 elements, should show all (though order may vary)
	assert "," in result or len(result.split("[")[1].split("]")[0].split(",")) == 2


def test_tuple_vs_list_serialization() -> None:
	"""Test that tuples and lists serialize the same way."""
	from mahonia import format_iterable_var

	class TupleListCtx(NamedTuple):
		tuple_data: tuple[int, ...]
		list_data: list[int]

	ctx = TupleListCtx(tuple_data=(1, 2, 3, 4, 5), list_data=[1, 2, 3, 4, 5])

	tuple_var = Var[SizedIterable[int], TupleListCtx]("tuple_data")
	list_var = Var[SizedIterable[int], TupleListCtx]("list_data")

	tuple_result = format_iterable_var(tuple_var, ctx)
	list_result = format_iterable_var(list_var, ctx)

	# Both should format identically (both are indexable)
	assert tuple_result == "tuple_data:5[1,..5]"
	assert list_result == "list_data:5[1,..5]"

	# Test with small tuples
	small_ctx = TupleListCtx(tuple_data=(10, 20), list_data=[10, 20])
	tuple_result = format_iterable_var(tuple_var, small_ctx)
	list_result = format_iterable_var(list_var, small_ctx)

	assert tuple_result == "tuple_data:2[10,20]"
	assert list_result == "list_data:2[10,20]"


def test_mapexpr_in_anyexpr_serialization() -> None:
	"""Test that MapExpr in AnyExpr shows full evaluation trace."""

	class NumCtx(NamedTuple):
		nums: list[int]

	n = Var[int, NumCtx]("n")
	nums = Var[SizedIterable[int], NumCtx]("nums")

	# Create a mapped expression that returns booleans
	lt_ten = (n < 10).map(nums)
	any_lt_ten = AnyExpr(lt_ten)

	# Without context
	assert any_lt_ten.to_string() == "(any (map n -> (n < 10) nums))"

	# With context - should show full MapExpr evaluation
	ctx = NumCtx(nums=[3, 7, 2])
	result = any_lt_ten.to_string(ctx)
	assert result == "(any (map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> True)"

	# With some values >= 10
	ctx2 = NumCtx(nums=[15, 20, 25])
	result = any_lt_ten.to_string(ctx2)
	assert result == "(any (map n -> (n < 10) nums:3[15,..25] -> 3[False,..False]) -> False)"

	# Mixed values
	ctx3 = NumCtx(nums=[5, 15, 3])
	result = any_lt_ten.to_string(ctx3)
	assert result == "(any (map n -> (n < 10) nums:3[5,..3] -> 3[True,..True]) -> True)"


def test_mapexpr_in_allexpr_serialization() -> None:
	"""Test that MapExpr in AllExpr shows full evaluation trace."""

	class NumCtx(NamedTuple):
		nums: list[int]

	n = Var[int, NumCtx]("n")
	nums = Var[SizedIterable[int], NumCtx]("nums")

	# Create a mapped expression that returns booleans
	gt_zero = (n > 0).map(nums)
	all_gt_zero = AllExpr(gt_zero)

	# Without context
	assert all_gt_zero.to_string() == "(all (map n -> (n > 0) nums))"

	# With context - all positive
	ctx = NumCtx(nums=[3, 7, 2])
	result = all_gt_zero.to_string(ctx)
	assert result == "(all (map n -> (n > 0) nums:3[3,..2] -> 3[True,..True]) -> True)"

	# With some non-positive
	ctx2 = NumCtx(nums=[3, 0, 2])
	result = all_gt_zero.to_string(ctx2)
	assert result == "(all (map n -> (n > 0) nums:3[3,..2] -> 3[True,..True]) -> False)"

	# All non-positive
	ctx3 = NumCtx(nums=[-1, -5, 0])
	result = all_gt_zero.to_string(ctx3)
	assert result == "(all (map n -> (n > 0) nums:3[-1,..0] -> 3[False,..False]) -> False)"


def test_var_vs_mapexpr_in_anyexpr() -> None:
	"""Test the distinction between Var/Const and MapExpr in AnyExpr serialization."""

	class FlagCtx(NamedTuple):
		flags: list[bool]
		nums: list[int]

	flags = Var[SizedIterable[bool], FlagCtx]("flags")
	nums = Var[SizedIterable[int], FlagCtx]("nums")
	n = Var[int, FlagCtx]("n")

	# Simple Var - uses compact format
	simple_any = AnyExpr(flags)
	ctx = FlagCtx(flags=[True, False, True], nums=[1, 2, 3])
	assert simple_any.to_string(ctx) == "(any flags:3[True,..True] -> True)"

	# MapExpr - shows full evaluation trace
	mapped = (n > 1).map(nums)
	complex_any = AnyExpr(mapped)
	assert (
		complex_any.to_string(ctx)
		== "(any (map n -> (n > 1) nums:3[1,..3] -> 3[False,..True]) -> True)"
	)


def test_var_vs_mapexpr_in_allexpr() -> None:
	"""Test the distinction between Var/Const and MapExpr in AllExpr serialization."""

	class FlagCtx(NamedTuple):
		flags: list[bool]
		nums: list[int]

	flags = Var[SizedIterable[bool], FlagCtx]("flags")
	nums = Var[SizedIterable[int], FlagCtx]("nums")
	n = Var[int, FlagCtx]("n")

	# Simple Var - uses compact format
	simple_all = AllExpr(flags)
	ctx = FlagCtx(flags=[True, True, True], nums=[5, 10, 15])
	assert simple_all.to_string(ctx) == "(all flags:3[True,..True] -> True)"

	# MapExpr - shows full evaluation trace
	mapped = (n >= 5).map(nums)
	complex_all = AllExpr(mapped)
	assert (
		complex_all.to_string(ctx)
		== "(all (map n -> (n >= 5) nums:3[5,..15] -> 3[True,..True]) -> True)"
	)


def test_empty_iterable_serialization() -> None:
	"""Test serialization of empty iterables."""
	from mahonia import format_iterable_var

	class EmptyCtx(NamedTuple):
		empty_list: list[int]
		empty_set: set[str]
		empty_tuple: tuple[int, ...]

	ctx = EmptyCtx(empty_list=[], empty_set=set(), empty_tuple=())

	empty_list = Var[SizedIterable[int], EmptyCtx]("empty_list")
	assert format_iterable_var(empty_list, ctx) == "empty_list:0[]"

	empty_set = Var[SizedIterable[str], EmptyCtx]("empty_set")
	assert format_iterable_var(empty_set, ctx) == "empty_set:0[]"

	empty_tuple = Var[SizedIterable[int], EmptyCtx]("empty_tuple")
	assert format_iterable_var(empty_tuple, ctx) == "empty_tuple:0[]"


def test_large_iterable_serialization() -> None:
	"""Test that large iterables show first..last format."""
	from mahonia import format_iterable_var

	class LargeCtx(NamedTuple):
		large_list: list[int]

	# Test with exactly 3 elements (threshold for first..last)
	ctx3 = LargeCtx(large_list=[10, 20, 30])
	large_var = Var[SizedIterable[int], LargeCtx]("large_list")
	assert format_iterable_var(large_var, ctx3) == "large_list:3[10,..30]"

	# Test with many elements
	ctx_many = LargeCtx(large_list=list(range(100)))
	assert format_iterable_var(large_var, ctx_many) == "large_list:100[0,..99]"


def test_foldl_add_int() -> None:
	"""Test FoldLExpr with Add operation."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")

	foldl_expr = FoldLExpr(Add, values)

	assert foldl_expr.to_string() == "(foldl + values)"

	ctx = ContainerCtx(values=[1, 2, 3, 4])
	assert foldl_expr.unwrap(ctx) == 10
	assert foldl_expr.to_string(ctx) == "(foldl + values:4 -> (1 + 2 + 3 + 4) -> 10)"

	assert FoldLExpr(Add, values, 10).unwrap(ctx) == 20
	assert FoldLExpr(Add, values, 10).to_string(ctx) == (
		"(foldl + values:4 -> (10 + 1 + 2 + 3 + 4) -> 20)"
	)


def test_foldl_add_expr() -> None:
	"""Test FoldLExpr with Add operation on Expr elements."""

	class XYCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, XYCtx]("x")
	y = Var[int, XYCtx]("y")

	expr_1 = x + y
	expr_2 = x * x
	expr_3 = x * y
	expr_4 = expr_1 * expr_2 - expr_3

	class FoldCtx(NamedTuple):
		x: int
		y: int
		values: list[Expr[int, XYCtx, int]]

	values = Var[SizedIterable[Expr[int, XYCtx, int]], FoldCtx]("values")

	foldl_expr = FoldLExpr(Add, values)

	assert foldl_expr.to_string() == "(foldl + values)"
	ctx = FoldCtx(x=5, y=8, values=[expr_1, expr_2, expr_3, expr_4])
	assert foldl_expr.unwrap(ctx) == 363
	assert foldl_expr.to_string(ctx) == (
		"(foldl + values:4 -> ((x:5 + y:8 -> 13) + (x:5 * x:5 -> 25) + (x:5 * y:8 -> 40) "
		"+ (((x:5 + y:8 -> 13) * (x:5 * x:5 -> 25) -> 325) - (x:5 * y:8 -> 40) -> 285)) -> 363)"
	)


def test_foldl_partial() -> None:
	"""Test FoldLExpr with partial application."""

	class PartialCtx(NamedTuple):
		values: list[int]
		multiplier: int

	values = Var[SizedIterable[int], PartialCtx]("values")

	foldl_expr = FoldLExpr(Add, values)

	class ValuesOnlyCtx(NamedTuple):
		values: list[int]

	partial_expr = foldl_expr.partial(ValuesOnlyCtx(values=[1, 2, 3, 4]))

	assert partial_expr.to_string() == "(foldl + values:[1, 2, 3, 4])"

	class EmptyCtx(NamedTuple):
		pass

	assert partial_expr.unwrap(EmptyCtx()) == 10


def test_foldl_partial_preserves_structure() -> None:
	"""Test that FoldLExpr partial application preserves nested expression structure.

	This tests that when we partial a FoldLExpr, we can later provide elements
	that are themselves expressions requiring a context for final evaluation.
	"""

	class XYCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, XYCtx]("x")
	y = Var[int, XYCtx]("y")

	expr_1 = x + y
	expr_2 = x * y

	class FoldCtx(NamedTuple):
		x: int
		y: int
		values: list[Expr[int, XYCtx, int]]

	values = Var[SizedIterable[Expr[int, XYCtx, int]], FoldCtx]("values")
	foldl_expr = FoldLExpr(Add, values)

	class ValuesOnlyCtx(NamedTuple):
		values: list[Expr[int, XYCtx, int]]

	partial_ctx = ValuesOnlyCtx(values=[expr_1, expr_2])
	partial_expr = foldl_expr.partial(partial_ctx)

	class XYOnlyCtx(NamedTuple):
		x: int
		y: int

	result = partial_expr.unwrap(XYOnlyCtx(x=5, y=3))
	assert result == (5 + 3) + (5 * 3)  # 8 + 15 = 23


def test_foldl_with_bound_expr() -> None:
	"""Test FoldLExpr with BoundExpr elements (expr-compatible BoundExpr)."""
	from mahonia import BoundExpr

	class XYCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, XYCtx]("x")
	y = Var[int, XYCtx]("y")

	ctx1 = XYCtx(x=2, y=3)
	ctx2 = XYCtx(x=4, y=5)
	ctx3 = XYCtx(x=6, y=7)

	bound_1 = (x + y).bind(ctx1)
	bound_2 = (x * y).bind(ctx2)
	bound_3 = (x - y).bind(ctx3)

	class BoundFoldCtx(NamedTuple):
		bounds: list[BoundExpr[int, XYCtx, int]]

	bounds = Var[SizedIterable[BoundExpr[int, XYCtx, int]], BoundFoldCtx]("bounds")
	foldl_expr = FoldLExpr(Add, bounds)

	ctx = BoundFoldCtx(bounds=[bound_1, bound_2, bound_3])
	assert foldl_expr.unwrap(ctx) == (2 + 3) + (4 * 5) + (6 - 7)  # 5 + 20 + (-1) = 24


def test_foldl_bound_expr_composition() -> None:
	"""Test FoldLExpr result can be composed with BoundExpr."""
	from mahonia import BoundExpr

	class SumCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], SumCtx]("values")
	foldl_sum = FoldLExpr(Add, values)

	ctx = SumCtx(values=[1, 2, 3, 4])
	bound_sum: BoundExpr[int, SumCtx, int] = foldl_sum.bind(ctx)

	assert bound_sum.unwrap() == 10

	class MultiplyCtx(NamedTuple):
		factor: int

	factor = Var[int, MultiplyCtx]("factor")
	combined = bound_sum * factor

	mult_ctx = MultiplyCtx(factor=3)
	assert combined.unwrap(mult_ctx) == 30
	assert combined.to_string(mult_ctx) == (
		"((foldl + values:4 -> (1 + 2 + 3 + 4) -> 10) * factor:3 -> 30)"
	)


def test_foldl_partial_then_bind() -> None:
	"""Test partial application followed by binding produces correct closed term."""
	from mahonia import BoundExpr

	class FullCtx(NamedTuple):
		values: list[int]
		offset: int

	values = Var[SizedIterable[int], FullCtx]("values")
	offset = Var[int, FullCtx]("offset")

	foldl_sum = FoldLExpr(Add, values)
	expr = foldl_sum + offset

	class ValuesCtx(NamedTuple):
		values: list[int]

	partial_expr = expr.partial(ValuesCtx(values=[10, 20, 30]))

	class OffsetCtx(NamedTuple):
		offset: int

	bound: BoundExpr[int, OffsetCtx, int] = partial_expr.bind(OffsetCtx(offset=5))

	assert bound.unwrap() == 65  # 10 + 20 + 30 + 5
	assert bound.to_string() == (
		"((foldl + values:[10, 20, 30]:3 -> (10 + 20 + 30) -> 60) + offset:5 -> 65)"
	)


def test_foldl_mul() -> None:
	"""Test FoldLExpr with Mul operation."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")
	foldl_expr = FoldLExpr(Mul, values)

	assert foldl_expr.to_string() == "(foldl * values)"

	ctx = ContainerCtx(values=[2, 3, 4])
	assert foldl_expr.unwrap(ctx) == 24
	assert foldl_expr.to_string(ctx) == "(foldl * values:3 -> (2 * 3 * 4) -> 24)"


def test_foldl_and() -> None:
	"""Test FoldLExpr with And operation."""

	class ContainerCtx(NamedTuple):
		flags: list[bool]

	flags = Var[SizedIterable[bool], ContainerCtx]("flags")
	foldl_expr = FoldLExpr(And, flags)

	assert foldl_expr.to_string() == "(foldl & flags)"

	ctx_all_true = ContainerCtx(flags=[True, True, True])
	assert foldl_expr.unwrap(ctx_all_true) is True
	assert foldl_expr.to_string(ctx_all_true) == "(foldl & flags:3 -> (True & True & True) -> True)"

	ctx_one_false = ContainerCtx(flags=[True, False, True])
	assert foldl_expr.unwrap(ctx_one_false) is False
	assert (
		foldl_expr.to_string(ctx_one_false) == "(foldl & flags:3 -> (True & False & True) -> False)"
	)


def test_foldl_or() -> None:
	"""Test FoldLExpr with Or operation."""

	class ContainerCtx(NamedTuple):
		flags: list[bool]

	flags = Var[SizedIterable[bool], ContainerCtx]("flags")
	foldl_expr = FoldLExpr(Or, flags)

	assert foldl_expr.to_string() == "(foldl | flags)"

	ctx_all_false = ContainerCtx(flags=[False, False, False])
	assert foldl_expr.unwrap(ctx_all_false) is False

	ctx_one_true = ContainerCtx(flags=[False, True, False])
	assert foldl_expr.unwrap(ctx_one_true) is True


def test_foldl_empty_container() -> None:
	"""Test that FoldLExpr returns identity element on empty container."""

	class ContainerCtx(NamedTuple):
		values: list[int]
		flags: list[bool]

	values = Var[SizedIterable[int], ContainerCtx]("values")
	flags = Var[SizedIterable[bool], ContainerCtx]("flags")

	ctx = ContainerCtx(values=[], flags=[])

	assert FoldLExpr(Add, values).unwrap(ctx) == 0
	assert FoldLExpr(Mul, values).unwrap(ctx) == 1
	assert FoldLExpr(And, flags).unwrap(ctx) is True
	assert FoldLExpr(Or, flags).unwrap(ctx) is False


def test_foldl_single_element() -> None:
	"""Test FoldLExpr with single element returns identity op element."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")

	ctx = ContainerCtx(values=[42])
	assert FoldLExpr(Add, values).unwrap(ctx) == 42
	assert FoldLExpr(Mul, values).unwrap(ctx) == 42


def test_foldl_bound_predicates() -> None:
	"""Test folding bound predicates with And - the motivating use case.

	This demonstrates validating multiple conditions on a measurement context,
	where predicates are bound to the measurement, then reduced lazily.
	"""
	from mahonia import Approximately, BoundExpr, PlusMinus, Predicate

	class Measurement(NamedTuple):
		voltage: float
		current: float
		temperature: float
		power: float

	voltage = Var[float, Measurement]("voltage")
	current = Var[float, Measurement]("current")
	temperature = Var[float, Measurement]("temperature")
	power = Var[float, Measurement]("power")

	voltage_ok = Predicate("Voltage OK", Approximately(voltage, PlusMinus("target", 5.0, 0.1)))
	current_ok = Predicate("Current OK", Approximately(current, PlusMinus("target", 2.0, 0.05)))
	temp_ok = Predicate("Temp OK", (temperature > 20.0) & (temperature < 80.0))
	power_ok = Predicate("Power OK", power < 15.0)

	class ValidationCtx(NamedTuple):
		tests: list[BoundExpr[bool, Measurement, bool]]

	tests = Var[SizedIterable[BoundExpr[bool, Measurement, bool]], ValidationCtx]("tests")
	all_pass = FoldLExpr(And, tests)

	assert all_pass.to_string() == "(foldl & tests)"

	passing_measurement = Measurement(voltage=5.02, current=1.98, temperature=25.0, power=10.0)
	passing_ctx = ValidationCtx(
		tests=[
			pred.bind(passing_measurement) for pred in [voltage_ok, current_ok, temp_ok, power_ok]
		],
	)

	assert all_pass.unwrap(passing_ctx) is True
	assert all_pass.to_string(passing_ctx) == (
		"(foldl & tests:4 -> ("
		"Voltage OK: True (voltage:5.02 ≈ target:5.0 ± 0.1 -> True) & "
		"Current OK: True (current:1.98 ≈ target:2.0 ± 0.05 -> True) & "
		"Temp OK: True ((temperature:25.0 > 20.0 -> True) & (temperature:25.0 < 80.0 -> True) -> True) & "
		"Power OK: True (power:10.0 < 15.0 -> True)) -> True)"
	)

	failing_measurement = Measurement(voltage=5.5, current=1.98, temperature=25.0, power=10.0)
	failing_ctx = ValidationCtx(
		tests=[
			pred.bind(failing_measurement) for pred in [voltage_ok, current_ok, temp_ok, power_ok]
		],
	)

	assert all_pass.unwrap(failing_ctx) is False
	assert all_pass.to_string(failing_ctx) == (
		"(foldl & tests:4 -> ("
		"Voltage OK: False (voltage:5.5 ≈ target:5.0 ± 0.1 -> False) & "
		"Current OK: True (current:1.98 ≈ target:2.0 ± 0.05 -> True) & "
		"Temp OK: True ((temperature:25.0 > 20.0 -> True) & (temperature:25.0 < 80.0 -> True) -> True) & "
		"Power OK: True (power:10.0 < 15.0 -> True)) -> False)"
	)


def test_min_expr() -> None:
	"""Test MinExpr operation."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")
	min_expr = MinExpr(values)

	assert min_expr.to_string() == "(min values)"

	ctx = ContainerCtx(values=[3, 1, 4, 1, 5, 9])
	assert min_expr.unwrap(ctx) == 1
	assert min_expr.to_string(ctx) == "(min values:6[3,..9] -> 1)"


def test_max_expr() -> None:
	"""Test MaxExpr operation."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")
	max_expr = MaxExpr(values)

	assert max_expr.to_string() == "(max values)"

	ctx = ContainerCtx(values=[3, 1, 4, 1, 5, 9])
	assert max_expr.unwrap(ctx) == 9
	assert max_expr.to_string(ctx) == "(max values:6[3,..9] -> 9)"


def test_foldl_min() -> None:
	"""Test FoldLExpr with Min operation."""

	class ContainerCtx(NamedTuple):
		values: list[float]

	values = Var[SizedIterable[float], ContainerCtx]("values")
	foldl_expr = FoldLExpr(Min, values)

	assert foldl_expr.to_string() == "(foldl min values)"

	ctx = ContainerCtx(values=[3.5, 1.2, 4.8])
	assert foldl_expr.unwrap(ctx) == 1.2


def test_foldl_max() -> None:
	"""Test FoldLExpr with Max operation."""

	class ContainerCtx(NamedTuple):
		values: list[float]

	values = Var[SizedIterable[float], ContainerCtx]("values")
	foldl_expr = FoldLExpr(Max, values)

	assert foldl_expr.to_string() == "(foldl max values)"

	ctx = ContainerCtx(values=[3.5, 1.2, 4.8])
	assert foldl_expr.unwrap(ctx) == 4.8


def test_minexpr_maxexpr_single_element() -> None:
	"""Test Min/Max with single element."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")

	ctx = ContainerCtx(values=[42])
	assert MinExpr(values).unwrap(ctx) == 42
	assert MaxExpr(values).unwrap(ctx) == 42
	assert FoldLExpr(Min, values).unwrap(ctx) == 42
	assert FoldLExpr(Max, values).unwrap(ctx) == 42


def test_min_max_with_floats() -> None:
	"""Test Min/Max with float values."""

	class ContainerCtx(NamedTuple):
		temps: list[float]

	temps = Var[SizedIterable[float], ContainerCtx]("temps")

	ctx = ContainerCtx(temps=[23.5, 19.2, 25.8, 21.0])
	assert MinExpr(temps).unwrap(ctx) == 19.2
	assert MaxExpr(temps).unwrap(ctx) == 25.8


def test_min_max_types() -> None:
	"""Test that MinExpr and MaxExpr have correct types."""
	values = Var[SizedIterable[int], ContainerData]("values")

	min_expr = MinExpr(values)
	assert_type(min_expr, MinExpr[int, ContainerData])

	max_expr = MaxExpr(values)
	assert_type(max_expr, MaxExpr[int, ContainerData])


def test_min_max_composition() -> None:
	"""Test Min/Max in expression composition."""

	class ContainerCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], ContainerCtx]("values")
	threshold = Const("threshold", 5)

	min_above_threshold = MinExpr(values) > threshold
	max_below_threshold = MaxExpr(values) < threshold

	ctx_high = ContainerCtx(values=[6, 7, 8])
	assert min_above_threshold.unwrap(ctx_high) is True

	ctx_low = ContainerCtx(values=[1, 2, 3])
	assert max_below_threshold.unwrap(ctx_low) is True


class FoldLCtx(NamedTuple):
	ints: list[int]
	floats: list[float]
	bools: list[bool]


@pytest.mark.mypy_testing
def test_foldl_add_type_preservation() -> None:
	"""Verify FoldLExpr with Add preserves int type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")

	fold_add = FoldLExpr(Add, ints)
	assert_type(fold_add, FoldLExpr[int, FoldLCtx])

	ctx = FoldLCtx(ints=[1, 2, 3], floats=[], bools=[])
	result = fold_add.eval(ctx)
	assert_type(result, Const[int])
	assert_type(fold_add.unwrap(ctx), int)


@pytest.mark.mypy_testing
def test_foldl_mul_type_preservation() -> None:
	"""Verify FoldLExpr with Mul preserves int type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")

	fold_mul = FoldLExpr(Mul, ints)
	assert_type(fold_mul, FoldLExpr[int, FoldLCtx])

	ctx = FoldLCtx(ints=[2, 3, 4], floats=[], bools=[])
	result = fold_mul.eval(ctx)
	assert_type(result, Const[int])


@pytest.mark.mypy_testing
def test_foldl_and_type_preservation() -> None:
	"""Verify FoldLExpr with And preserves bool type."""
	bools = Var[SizedIterable[bool], FoldLCtx]("bools")

	fold_and = FoldLExpr(And, bools)
	assert_type(fold_and, FoldLExpr[bool, FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[], bools=[True, True, False])
	result = fold_and.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(fold_and.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_foldl_or_type_preservation() -> None:
	"""Verify FoldLExpr with Or preserves bool type."""
	bools = Var[SizedIterable[bool], FoldLCtx]("bools")

	fold_or = FoldLExpr(Or, bools)
	assert_type(fold_or, FoldLExpr[bool, FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[], bools=[False, False, True])
	result = fold_or.eval(ctx)
	assert_type(result, Const[bool])


@pytest.mark.mypy_testing
def test_foldl_min_type_preservation() -> None:
	"""Verify FoldLExpr with Min preserves float type."""
	floats = Var[SizedIterable[float], FoldLCtx]("floats")

	fold_min = FoldLExpr(Min, floats)
	assert_type(fold_min, FoldLExpr[float, FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[3.5, 1.2, 4.8], bools=[])
	result = fold_min.eval(ctx)
	assert_type(result, Const[float])
	assert_type(fold_min.unwrap(ctx), float)


@pytest.mark.mypy_testing
def test_foldl_max_type_preservation() -> None:
	"""Verify FoldLExpr with Max preserves float type."""
	floats = Var[SizedIterable[float], FoldLCtx]("floats")

	fold_max = FoldLExpr(Max, floats)
	assert_type(fold_max, FoldLExpr[float, FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[3.5, 1.2, 4.8], bools=[])
	result = fold_max.eval(ctx)
	assert_type(result, Const[float])


@pytest.mark.mypy_testing
def test_foldl_with_initial_value_type() -> None:
	"""Verify FoldLExpr with initial value preserves type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")

	fold_with_init = FoldLExpr(Add, ints, 100)
	assert_type(fold_with_init, FoldLExpr[int, FoldLCtx])

	ctx = FoldLCtx(ints=[1, 2, 3], floats=[], bools=[])
	result = fold_with_init.eval(ctx)
	assert_type(result, Const[int])


@pytest.mark.mypy_testing
def test_minexpr_type_preservation() -> None:
	"""Verify MinExpr preserves element type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")

	min_expr = MinExpr(ints)
	assert_type(min_expr, MinExpr[int, FoldLCtx])

	ctx = FoldLCtx(ints=[5, 2, 8], floats=[], bools=[])
	result = min_expr.eval(ctx)
	assert_type(result, Const[int])
	assert_type(min_expr.unwrap(ctx), int)


@pytest.mark.mypy_testing
def test_maxexpr_type_preservation() -> None:
	"""Verify MaxExpr preserves element type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")

	max_expr = MaxExpr(ints)
	assert_type(max_expr, MaxExpr[int, FoldLCtx])

	ctx = FoldLCtx(ints=[5, 2, 8], floats=[], bools=[])
	result = max_expr.eval(ctx)
	assert_type(result, Const[int])
	assert_type(max_expr.unwrap(ctx), int)


@pytest.mark.mypy_testing
def test_anyexpr_type_preservation() -> None:
	"""Verify AnyExpr returns bool type."""
	bools = Var[SizedIterable[bool], FoldLCtx]("bools")

	any_expr = AnyExpr(bools)
	assert_type(any_expr, AnyExpr[FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[], bools=[False, True, False])
	result = any_expr.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(any_expr.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_allexpr_type_preservation() -> None:
	"""Verify AllExpr returns bool type."""
	bools = Var[SizedIterable[bool], FoldLCtx]("bools")

	all_expr = AllExpr(bools)
	assert_type(all_expr, AllExpr[FoldLCtx])

	ctx = FoldLCtx(ints=[], floats=[], bools=[True, True, True])
	result = all_expr.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(all_expr.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_contains_type_preservation() -> None:
	"""Verify Contains returns bool type."""
	ints = Var[SizedIterable[int], FoldLCtx]("ints")
	target = Const("target", 5)

	contains_expr = Contains(target, ints)
	assert_type(contains_expr, Contains[int, Any])

	ctx = FoldLCtx(ints=[1, 5, 10], floats=[], bools=[])
	result = contains_expr.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(contains_expr.unwrap(ctx), bool)


class MapComposeCtx(NamedTuple):
	nums: list[int]


class ElemCtx(NamedTuple):
	n: int


@pytest.mark.mypy_testing
def test_anyexpr_with_mapexpr_composition() -> None:
	"""Verify AnyExpr composed with MapExpr preserves bool type."""
	from mahonia import MapExpr

	n = Var[int, ElemCtx]("n")
	nums = Var[SizedIterable[int], MapComposeCtx]("nums")

	mapped = (n > 5).map(nums)
	assert_type(mapped, MapExpr[Any, bool, Any])

	any_mapped = AnyExpr(mapped)
	assert_type(any_mapped, AnyExpr[Any])

	ctx = MapComposeCtx(nums=[1, 10, 3])
	result = any_mapped.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(any_mapped.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_allexpr_with_mapexpr_composition() -> None:
	"""Verify AllExpr composed with MapExpr preserves bool type."""
	from mahonia import MapExpr

	n = Var[int, ElemCtx]("n")
	nums = Var[SizedIterable[int], MapComposeCtx]("nums")

	mapped = (n > 0).map(nums)
	assert_type(mapped, MapExpr[Any, bool, Any])

	all_mapped = AllExpr(mapped)
	assert_type(all_mapped, AllExpr[Any])

	ctx = MapComposeCtx(nums=[1, 2, 3])
	result = all_mapped.eval(ctx)
	assert_type(result, Const[bool])
	assert_type(all_mapped.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_foldl_with_mapexpr_composition() -> None:
	"""Verify FoldLExpr composed with MapExpr preserves element type."""
	from mahonia import MapExpr

	n = Var[int, ElemCtx]("n")
	nums = Var[SizedIterable[int], MapComposeCtx]("nums")

	mapped = (n * 2).map(nums)
	assert_type(mapped, MapExpr[Any, int, Any])

	fold_mapped = FoldLExpr(Add, mapped)
	assert_type(fold_mapped, FoldLExpr[int, Any])

	ctx = MapComposeCtx(nums=[1, 2, 3])
	result = fold_mapped.eval(ctx)
	assert_type(result, Const[int])
	assert_type(fold_mapped.unwrap(ctx), int)
