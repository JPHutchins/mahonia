# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test container operations."""

from typing import Any, NamedTuple, assert_type

from mahonia import AllExpr, AnyExpr, Const, Contains, SizedIterable, Var


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

	assert any_expr.to_string() == "any(flags)"
	assert any_expr.unwrap(ctx) is True
	assert any_expr.to_string(ctx) == "any(flags:3[True,..True] -> True)"

	# Test with all False
	all_false_ctx = ContainerData(values=[1, 2, 3], targets=["a", "b"], flags=[False, False, False])
	assert any_expr.unwrap(all_false_ctx) is False
	assert any_expr.to_string(all_false_ctx) == "any(flags:3[False,..False] -> False)"


def test_all_expr() -> None:
	"""Test AllExpr operation."""
	ctx = ContainerData(values=[1, 2, 3, 4, 5], targets=["a", "b", "c"], flags=[True, True, True])

	flags = Var[SizedIterable[bool], ContainerData]("flags")
	all_expr = AllExpr(flags)

	assert all_expr.to_string() == "all(flags)"
	assert all_expr.unwrap(ctx) is True
	assert all_expr.to_string(ctx) == "all(flags:3[True,..True] -> True)"

	# Test with one False
	mixed_ctx = ContainerData(values=[1, 2, 3], targets=["a", "b"], flags=[True, False, True])
	assert all_expr.unwrap(mixed_ctx) is False
	assert all_expr.to_string(mixed_ctx) == "all(flags:3[True,..True] -> False)"


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
	assert any_lt_ten.to_string() == "any((map n -> (n < 10) nums))"

	# With context - should show full MapExpr evaluation
	ctx = NumCtx(nums=[3, 7, 2])
	result = any_lt_ten.to_string(ctx)
	assert result == "any((map n -> (n < 10) nums:3[3,..2] -> 3[True,..True]) -> True)"

	# With some values >= 10
	ctx2 = NumCtx(nums=[15, 20, 25])
	result = any_lt_ten.to_string(ctx2)
	assert result == "any((map n -> (n < 10) nums:3[15,..25] -> 3[False,..False]) -> False)"

	# Mixed values
	ctx3 = NumCtx(nums=[5, 15, 3])
	result = any_lt_ten.to_string(ctx3)
	assert result == "any((map n -> (n < 10) nums:3[5,..3] -> 3[True,..True]) -> True)"


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
	assert all_gt_zero.to_string() == "all((map n -> (n > 0) nums))"

	# With context - all positive
	ctx = NumCtx(nums=[3, 7, 2])
	result = all_gt_zero.to_string(ctx)
	assert result == "all((map n -> (n > 0) nums:3[3,..2] -> 3[True,..True]) -> True)"

	# With some non-positive
	ctx2 = NumCtx(nums=[3, 0, 2])
	result = all_gt_zero.to_string(ctx2)
	assert result == "all((map n -> (n > 0) nums:3[3,..2] -> 3[True,..True]) -> False)"

	# All non-positive
	ctx3 = NumCtx(nums=[-1, -5, 0])
	result = all_gt_zero.to_string(ctx3)
	assert result == "all((map n -> (n > 0) nums:3[-1,..0] -> 3[False,..False]) -> False)"


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
	assert simple_any.to_string(ctx) == "any(flags:3[True,..True] -> True)"

	# MapExpr - shows full evaluation trace
	mapped = (n > 1).map(nums)
	complex_any = AnyExpr(mapped)
	assert (
		complex_any.to_string(ctx)
		== "any((map n -> (n > 1) nums:3[1,..3] -> 3[False,..True]) -> True)"
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
	assert simple_all.to_string(ctx) == "all(flags:3[True,..True] -> True)"

	# MapExpr - shows full evaluation trace
	mapped = (n >= 5).map(nums)
	complex_all = AllExpr(mapped)
	assert (
		complex_all.to_string(ctx)
		== "all((map n -> (n >= 5) nums:3[5,..15] -> 3[True,..True]) -> True)"
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
