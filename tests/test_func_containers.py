# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test functional programming with containers."""

from typing import Any, NamedTuple, assert_type

import pytest

from mahonia import (
	Const,
	Expr,
	Func,
	MapExpr,
	SizedIterable,
	Var,
)


class ContainerCtx(NamedTuple):
	numbers: list[int]
	values: list[float]


class ElementCtx(NamedTuple):
	x: int


class FloatElementCtx(NamedTuple):
	f: float


def test_map_expr_construction() -> None:
	"""Test MapExpr direct construction."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	func = (x * 2).to_func()
	map_expr = MapExpr(func, numbers)

	assert map_expr.to_string() == "(map x -> (x * 2) numbers)"


def test_map_expr_evaluation() -> None:
	"""Test MapExpr evaluation with context."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	map_expr = MapExpr((x * 2).to_func(), numbers)
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	result = map_expr.unwrap(ctx)
	assert result == [2, 4, 6]


def test_fluent_map_syntax() -> None:
	"""Test fluent .map() method on expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 3).map(numbers)
	assert mapped.to_string() == "(map x -> (x * 3) numbers)"

	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])
	assert mapped.unwrap(ctx) == [3, 6, 9]


def test_fluent_chaining() -> None:
	"""Test fluent chaining with unwrap."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3, 4], values=[])

	result = (x * x).map(numbers).unwrap(ctx)
	assert result == [1, 4, 9, 16]


def test_complex_expressions() -> None:
	"""Test mapping complex expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	complex_expr = (x + 1) * (x + 1)
	mapped = complex_expr.map(numbers)

	ctx = ContainerCtx(numbers=[0, 1, 2], values=[])
	result = mapped.unwrap(ctx)
	assert result == [1, 4, 9]  # (0+1)²=1, (1+1)²=4, (2+1)²=6


def test_arithmetic_operations() -> None:
	"""Test various arithmetic operations in mapping."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	assert (x + 10).map(numbers).unwrap(ctx) == [11, 12, 13]
	assert (x - 1).map(numbers).unwrap(ctx) == [0, 1, 2]
	assert (x * 4).map(numbers).unwrap(ctx) == [4, 8, 12]
	assert (x + x).map(numbers).unwrap(ctx) == [2, 4, 6]


def test_empty_container() -> None:
	"""Test mapping over empty containers."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	ctx = ContainerCtx(numbers=[], values=[])
	result = (x * 2).map(numbers).unwrap(ctx)
	assert result == []


def test_single_element() -> None:
	"""Test mapping over single-element containers."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	ctx = ContainerCtx(numbers=[5], values=[])
	result = (x * 2).map(numbers).unwrap(ctx)
	assert result == [10]


def test_to_func_conversion() -> None:
	"""Test converting mapped expressions to functions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 2).map(numbers)
	func = mapped.to_func()

	assert "numbers" in func.to_string()
	assert "x" in func.to_string()


def test_string_representations_without_context() -> None:
	"""Test string representations without context."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Simple expressions
	assert (x + 5).map(numbers).to_string() == "(map x -> (x + 5) numbers)"
	assert (x * 2).map(numbers).to_string() == "(map x -> (x * 2) numbers)"
	assert (x - 1).map(numbers).to_string() == "(map x -> (x - 1) numbers)"

	# Complex expressions
	assert (x * x).map(numbers).to_string() == "(map x -> (x * x) numbers)"
	assert ((x + 1) * 2).map(numbers).to_string() == "(map x -> ((x + 1) * 2) numbers)"
	assert (x + (x * 2)).map(numbers).to_string() == "(map x -> (x + (x * 2)) numbers)"


def test_string_representations_with_context() -> None:
	"""Test string representations with context."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	# Simple expressions with context
	result = (x + 5).map(numbers).to_string(ctx)
	assert result == "(map x -> (x + 5) numbers:3[1,..3] -> 3[6,..8])"

	result = (x * 2).map(numbers).to_string(ctx)
	assert result == "(map x -> (x * 2) numbers:3[1,..3] -> 3[2,..6])"


def test_string_representations_edge_cases() -> None:
	"""Test string representations for edge cases."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Empty container
	empty_ctx = ContainerCtx(numbers=[], values=[])
	result = (x * 2).map(numbers).to_string(empty_ctx)
	assert result == "(map x -> (x * 2) numbers:0[] -> 0[])"

	# Single element
	single_ctx = ContainerCtx(numbers=[42], values=[])
	result = (x + 1).map(numbers).to_string(single_ctx)
	assert result == "(map x -> (x + 1) numbers:1[42] -> 1[43])"


def test_string_representations_boolean_expressions() -> None:
	"""Test string representations for boolean expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Without context
	gt_expr = (x > 5).map(numbers)
	assert gt_expr.to_string() == "(map x -> (x > 5) numbers)"

	eq_expr = (x == 3).map(numbers)
	assert eq_expr.to_string() == "(map x -> (x == 3) numbers)"

	# With context
	ctx = ContainerCtx(numbers=[1, 3, 7], values=[])

	result = gt_expr.to_string(ctx)
	assert result == "(map x -> (x > 5) numbers:3[1,..7] -> 3[False,..True])"

	result = eq_expr.to_string(ctx)
	assert result == "(map x -> (x == 3) numbers:3[1,..7] -> 3[False,..False])"


def test_string_representations_logical_operations() -> None:
	"""Test string representations for logical operations."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Complex logical expression
	logic_expr = ((x > 1) & (x < 5)).map(numbers)
	expected = "(map x -> ((x > 1) & (x < 5)) numbers)"
	assert logic_expr.to_string() == expected

	# With context
	ctx = ContainerCtx(numbers=[0, 2, 4, 6], values=[])
	result = logic_expr.to_string(ctx)
	assert result == "(map x -> ((x > 1) & (x < 5)) numbers:4[0,..6] -> 4[False,..False])"


def test_string_representations_with_constants() -> None:
	"""Test string representations with constants."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	from mahonia import Const

	offset = Const("OFFSET", 100)

	# Without context
	expr = (x + offset).map(numbers)
	assert expr.to_string() == "(map x -> (x + OFFSET:100) numbers)"

	# With context
	ctx = ContainerCtx(numbers=[1, 2], values=[])
	result = expr.to_string(ctx)
	assert result == "(map x -> (x + OFFSET:100) numbers:2[1,2] -> 2[101,102])"


def test_string_representations_nested_expressions() -> None:
	"""Test string representations for nested expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Deeply nested expression
	nested = (((x + 1) * 2) - 3).map(numbers)
	expected = "(map x -> (((x + 1) * 2) - 3) numbers)"
	assert nested.to_string() == expected

	# With context
	ctx = ContainerCtx(numbers=[1, 2], values=[])
	result = nested.to_string(ctx)
	assert (
		result == "(map x -> (((x + 1) * 2) - 3) numbers:2[1,2] -> 2[1,3])"
	)  # ((1+1)*2)-3=1, ((2+1)*2)-3=3


def test_string_representations_float_operations() -> None:
	"""Test string representations for float operations."""
	f = Var[float, FloatElementCtx]("f")
	values = Var[SizedIterable[float], ContainerCtx]("values")

	# Without context
	expr = (f * 1.5).map(values)
	assert expr.to_string() == "(map f -> (f * 1.5) values)"

	# With context
	ctx = ContainerCtx(numbers=[], values=[2.0, 4.0])
	result = expr.to_string(ctx)
	assert result == "(map f -> (f * 1.5) values:2[2.0,4.0] -> 2[3.0,6.0])"


def test_string_representations_different_containers() -> None:
	"""Test string representations with different container variable names."""
	x = Var[int, ElementCtx]("x")

	class CtxA(NamedTuple):
		data: list[int]

	class CtxB(NamedTuple):
		items: list[int]

	data_var = Var[SizedIterable[int], CtxA]("data")
	items_var = Var[SizedIterable[int], CtxB]("items")

	# Same expression, different containers
	expr = x * 2

	data_mapped = expr.map(data_var)
	items_mapped = expr.map(items_var)

	assert data_mapped.to_string() == "(map x -> (x * 2) data)"
	assert items_mapped.to_string() == "(map x -> (x * 2) items)"

	# With context
	ctx_a = CtxA(data=[1, 2])
	ctx_b = CtxB(items=[3, 4])

	result_a = data_mapped.to_string(ctx_a)
	assert result_a == "(map x -> (x * 2) data:2[1,2] -> 2[2,4])"

	result_b = items_mapped.to_string(ctx_b)
	assert result_b == "(map x -> (x * 2) items:2[3,4] -> 2[6,8])"


def test_string_representations_large_containers() -> None:
	"""Test string representations for large containers."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	# Large container - should handle gracefully
	large_list = list(range(20))
	ctx = ContainerCtx(numbers=large_list, values=[])

	result = (x + 1).map(numbers).to_string(ctx)
	assert result == "(map x -> (x + 1) numbers:20[0,..19] -> 20[1,..20])"
	# Result should contain the mapped values


def test_nested_expressions() -> None:
	"""Test nested arithmetic expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	nested = x + (x * 2)
	result = nested.map(numbers).unwrap(ctx)
	assert result == [3, 6, 9]  # x + 2x = 3x


def test_duplicate_variable_usage() -> None:
	"""Test expressions that use the same variable multiple times."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[2, 3, 4], values=[])

	# x^2 + x
	expr = (x * x) + x
	result = expr.map(numbers).unwrap(ctx)
	assert result == [6, 12, 20]  # 2²+2=6, 3²+3=12, 4²+4=20


def test_constant_in_mapping() -> None:
	"""Test expressions with constants in mapping."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	# Expression with constant
	from mahonia import Const

	offset = Const("offset", 100)
	expr = x + offset

	result = expr.map(numbers).unwrap(ctx)
	assert result == [101, 102, 103]


def test_boolean_expressions() -> None:
	"""Test boolean expressions in mapping."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3, 4, 5], values=[])

	# Test greater than
	gt_expr = x > 3
	result = gt_expr.map(numbers).unwrap(ctx)
	assert result == [False, False, False, True, True]


def test_logical_operations() -> None:
	"""Test logical operations in mapping."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3, 4, 5], values=[])

	# Test compound boolean
	logic_expr = (x > 2) & (x < 5)
	result = logic_expr.map(numbers).unwrap(ctx)
	assert result == [False, False, True, True, False]


def test_float_operations() -> None:
	"""Test mapping with float operations."""
	f = Var[float, FloatElementCtx]("f")
	values = Var[SizedIterable[float], ContainerCtx]("values")
	ctx = ContainerCtx(numbers=[], values=[1.0, 2.0, 3.0])

	result = (f * 1.5).map(values).unwrap(ctx)
	assert result == [1.5, 3.0, 4.5]


def test_type_assertions() -> None:
	"""Test type safety of mapped expressions."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 2).map(numbers)
	# Test that we get a MapExpr instance
	assert isinstance(mapped, MapExpr)


def test_large_container() -> None:
	"""Test mapping over larger containers."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	large_list = list(range(100))
	ctx = ContainerCtx(numbers=large_list, values=[])

	result = (x + 1).map(numbers).unwrap(ctx)
	expected = [i + 1 for i in large_list]
	assert result == expected


def test_expression_reuse() -> None:
	"""Test reusing expressions across different containers."""
	x = Var[int, ElementCtx]("x")
	numbers1 = Var[SizedIterable[int], ContainerCtx]("numbers")

	class AltCtx(NamedTuple):
		data: list[int]

	numbers2 = Var[SizedIterable[int], AltCtx]("data")

	expr = x * 2
	mapped1 = expr.map(numbers1)
	mapped2 = expr.map(numbers2)

	ctx1 = ContainerCtx(numbers=[1, 2], values=[])
	ctx2 = AltCtx(data=[3, 4])

	assert mapped1.unwrap(ctx1) == [2, 4]
	assert mapped2.unwrap(ctx2) == [6, 8]


def test_map_expr_with_func_conversion() -> None:
	"""Test MapExpr to Func conversion preserves functionality."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x + 10).map(numbers)
	func = mapped.to_func()

	# Both should work with the same context
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])
	direct_result = mapped.unwrap(ctx)
	expected = [11, 12, 13]

	assert direct_result == expected
	assert "numbers" in func.to_string()


def test_syntax_equivalence() -> None:
	"""Test that verbose and fluent syntax produce identical results."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	# Verbose syntax
	func_verbose = (x * x).to_func()
	mapped_verbose = MapExpr(func_verbose, numbers)
	result_verbose = mapped_verbose.unwrap(ctx)

	# Fluent syntax
	result_fluent = (x * x).map(numbers).unwrap(ctx)

	assert result_verbose == result_fluent == [1, 4, 9]


@pytest.mark.mypy_testing
def test_mapexpr_int_result_type() -> None:
	"""Verify MapExpr preserves int result type."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 2).map(numbers)
	assert_type(mapped, MapExpr[Any, int, Any])

	func = (x + 10).to_func()
	assert_type(func, Func[int, ElementCtx])

	map_expr = MapExpr(func, numbers)
	assert_type(map_expr, MapExpr[int, int, ContainerCtx])

	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])
	result = mapped.unwrap(ctx)
	assert_type(result, SizedIterable[int])


@pytest.mark.mypy_testing
def test_mapexpr_float_result_type() -> None:
	"""Verify MapExpr preserves float result type."""
	f = Var[float, FloatElementCtx]("f")
	values = Var[SizedIterable[float], ContainerCtx]("values")

	mapped = (f * 1.5).map(values)
	assert_type(mapped, MapExpr[Any, float, Any])

	ctx = ContainerCtx(numbers=[], values=[1.0, 2.0])
	result = mapped.unwrap(ctx)
	assert_type(result, SizedIterable[float])


@pytest.mark.mypy_testing
def test_mapexpr_bool_result_type() -> None:
	"""Verify MapExpr preserves bool result type for predicates."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x > 5).map(numbers)
	assert_type(mapped, MapExpr[Any, bool, Any])

	ctx = ContainerCtx(numbers=[1, 10], values=[])
	result = mapped.unwrap(ctx)
	assert_type(result, SizedIterable[bool])


@pytest.mark.mypy_testing
def test_mapexpr_func_field_types() -> None:
	"""Verify MapExpr.func and MapExpr.container preserve types."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 2).map(numbers)
	assert_type(mapped.func, Func[int, Any])
	assert_type(mapped.container, Expr[SizedIterable[Any], Any, SizedIterable[Any]])


@pytest.mark.mypy_testing
def test_mapexpr_eval_returns_const() -> None:
	"""Verify MapExpr.eval returns Const with correct element type."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")

	mapped = (x * 2).map(numbers)
	ctx = ContainerCtx(numbers=[1, 2, 3], values=[])

	result = mapped.eval(ctx)
	assert_type(result, Const[SizedIterable[int]])


@pytest.mark.mypy_testing
def test_mapexpr_with_const_expression() -> None:
	"""Verify MapExpr with expressions containing Const."""
	x = Var[int, ElementCtx]("x")
	numbers = Var[SizedIterable[int], ContainerCtx]("numbers")
	offset = Const("offset", 100)

	mapped = (x + offset).map(numbers)
	assert_type(mapped, MapExpr[Any, int, Any])
