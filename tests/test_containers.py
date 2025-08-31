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
