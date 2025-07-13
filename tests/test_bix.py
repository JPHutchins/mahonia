# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Final, NamedTuple, assert_type

import pytest

from binexpr import (
	Add,
	And,
	Approximately,
	BoundExpr,
	Const,
	Eq,
	Expr,
	Lt,
	Not,
	Percent,
	PlusMinus,
	Predicate,
	TSupportsComparison,
	Var,
)


@dataclass(frozen=True)
class Ctx:
	x: int
	y: int
	name: str
	flag: bool = True
	custom: object = object()
	f: float = 1.5


ctx: Final = Ctx(x=5, y=10, name="example")


def between(
	expr: Expr[TSupportsComparison, Ctx], low: TSupportsComparison, high: TSupportsComparison
) -> "And[bool, Ctx]":
	"""Example of defining some convenience to compose an expression."""
	return And(  # type: ignore[arg-type]
		Lt(Const("Low", low), expr),  # type: ignore[arg-type]
		Lt(expr, Const("High", high)),  # type: ignore[arg-type]
	)


def test_const_name_eval_and_str() -> None:
	c = Const("Forty-two", 42)
	assert c.value == 42
	assert c.unwrap(ctx) == 42
	assert c.to_string() == "Forty-two:42"
	assert c.to_string(ctx) == "Forty-two:42"


def test_const_eval_and_str() -> None:
	c = Const(None, 100)
	assert c.value == 100
	assert c.unwrap(ctx) == 100
	assert c.to_string() == "100"
	assert c.to_string(ctx) == "100"


@pytest.mark.mypy_testing
def test_const_generic_type() -> None:
	c_int = Const(None, 1)
	assert_type(c_int, Const[int])
	assert_type(c_int.value, int)

	c_str = Const(None, "test")
	assert_type(c_str, Const[str])
	assert_type(c_str.value, str)

	c_float = Const(None, 1.5)
	assert_type(c_float, Const[float])
	assert_type(c_float.value, float)

	c_complex = Const(None, 1 + 2j)
	assert_type(c_complex, Const[complex])
	assert_type(c_complex.value, complex)

	c_bool = Const(None, True)
	assert_type(c_bool, Const[bool])
	assert_type(c_bool.value, bool)

	class CustomType:
		pass

	c_custom = Const(None, CustomType())
	assert_type(c_custom, Const[CustomType])
	assert_type(c_custom.value, CustomType)


@pytest.mark.mypy_testing
def test_var_generic_type() -> None:
	v_int = Var[int, Ctx]("x")
	assert_type(v_int, Var[int, Ctx])
	assert_type(v_int.unwrap(ctx), int)

	v_str = Var[str, Ctx]("name")
	assert_type(v_str, Var[str, Ctx])
	assert_type(v_str.unwrap(ctx), str)

	v_float = Var[float, Ctx]("y")
	assert_type(v_float, Var[float, Ctx])
	assert_type(v_float.unwrap(ctx), float)

	v_bool = Var[bool, Ctx]("flag")
	assert_type(v_bool, Var[bool, Ctx])
	assert_type(v_bool.unwrap(ctx), bool)

	class CustomType:
		pass

	v_custom = Var[CustomType, Ctx]("custom")
	assert_type(v_custom, Var[CustomType, Ctx])
	assert_type(v_custom.unwrap(ctx), CustomType)


@pytest.mark.mypy_testing
def test_eq_generic_type() -> None:
	v_int = Var[int, Ctx]("x")
	c_int = Const("Five", 5)
	eq_expr = v_int == c_int
	assert_type(eq_expr, Eq[int, Ctx])
	assert_type(eq_expr.unwrap(ctx), bool)

	v_str = Var[str, Ctx]("name")
	c_str = Const("Example", "example")
	eq_expr_str = v_str == c_str
	assert_type(eq_expr_str, Eq[str, Ctx])
	assert_type(eq_expr_str.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_add_generic_type() -> None:
	v_int = Var[int, Ctx]("x")
	c_int = Const("Five", 5)
	add_expr = v_int + c_int
	assert_type(add_expr, Add[int, Ctx])
	assert_type(add_expr.unwrap(ctx), int)

	v_float = Var[float, Ctx]("y")
	c_float = Const("Two", 2.0)
	add_expr_float = v_float + c_float
	assert_type(add_expr_float, Add[float, Ctx])
	assert_type(add_expr_float.unwrap(ctx), float)


def test_var_eval_and_str() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	name = Var[str, Ctx]("name")
	assert x.unwrap(ctx) == 5
	assert y.unwrap(ctx) == 10
	assert name.unwrap(ctx) == "example"
	assert x.to_string() == "x"
	assert x.to_string(ctx) == "x:5"
	assert name.to_string(ctx) == "name:example"


def test_add_sub_mul_div() -> None:
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	c5 = Const("Five", 5.0)
	c2 = Const("Two", 2.0)
	assert (x + y).unwrap(ctx) == 15
	assert (x + c5).unwrap(ctx) == 10
	assert (y - x).unwrap(ctx) == 5
	assert (x * c2).unwrap(ctx) == 10
	assert (y / c2).unwrap(ctx) == 5
	assert (x + y).to_string() == "(x + y)"
	assert (x + c5).to_string(ctx) == "(x:5 + Five:5.0 -> 10.0)"


def test_add() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	assert (x + y).unwrap(ctx) == 15
	assert (x + 5).unwrap(ctx) == 10
	assert (x + y).to_string() == "(x + y)"
	assert (x + 5).to_string(ctx) == "(x:5 + 5 -> 10)"


def test_comparisons() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	c5 = Const("Five", 5)
	c10 = Const("Ten", 10)
	assert (x == c5).unwrap(ctx) is True
	assert (y == c5).unwrap(ctx) is False
	assert (x != c5).unwrap(ctx) is False
	assert (y != c5).unwrap(ctx) is True
	assert (x < y).unwrap(ctx) is True
	assert (y > x).unwrap(ctx) is True
	assert (x <= c5).unwrap(ctx) is True
	assert (y >= c10).unwrap(ctx) is True
	assert (x == c5).to_string() == "(x == Five:5)"
	assert (x == c5).to_string(ctx) == "(x:5 == Five:5 -> True)"


def test_logical_ops() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	c5 = Const("Five", 5)
	c10 = Const("Ten", 10)
	pred = ((x == c5) & (y == c10)) | (x != c5)
	assert pred.unwrap(ctx) is True
	assert pred.to_string() == "(((x == Five:5) & (y == Ten:10)) | (x != Five:5))"  # noqa: E501
	# Evaluate with context
	assert (
		pred.to_string(ctx)
		== "(((x:5 == Five:5 -> True) & (y:10 == Ten:10 -> True) -> True) | (x:5 != Five:5 -> False) -> True)"  # noqa: E501
	)


def test_not() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("name", 5)
	expr = ~(x == c5)
	assert expr.unwrap(ctx) is False
	assert expr.to_string() == "(not (x == name:5))"
	assert expr.to_string(ctx) == "(not (x:5 == name:5 -> True) -> False)"


def test_nested_arithmetic() -> None:
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	expr = (x * Const("name", 2.0)) + (y / Const("name", 2.0))
	assert expr.unwrap(ctx) == 15
	assert expr.to_string() == "((x * name:2.0) + (y / name:2.0))"
	assert expr.to_string(ctx) == "((x:5 * name:2.0 -> 10.0) + (y:10 / name:2.0 -> 5.0) -> 15.0)"


def test_constants_only() -> None:
	c10 = Const("name", 10.0)
	c5 = Const("name", 5.0)
	assert (c10 + c5).unwrap(None) == 15
	assert (c10 / c5).unwrap(None) == 2
	assert (c10 == c10).unwrap(None) is True
	assert (c10 == c5).unwrap(None) is False
	assert (c10 == 10.0).unwrap(None) is True
	assert (c10 != 5.0).unwrap(None) is True
	assert (c10 > c5).unwrap(None) is True
	assert (c10 + c5 * 50).unwrap(None) == 260.0
	assert (c10 + c5 * 50).to_string() == "(name:10.0 + (name:5.0 * 50))"
	assert (c10 + c5 * 50).to_string(ctx) == "(name:10.0 + (name:5.0 * 50 -> 250.0) -> 260.0)"


def test_chained_arithmetic() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y * 2 - 3
	assert expr.unwrap(ctx) == 5 + 10 * 2 - 3


def test_const_to_string_edge_cases() -> None:
	c_none = Const(None, None)
	assert c_none.to_string() == "None"
	assert c_none.unwrap(ctx) is None


def test_bool_logic() -> None:
	flag = Var[bool, Ctx]("flag")
	c_true = Const("True", True)
	c_false = Const("False", False)
	assert (flag & c_true).unwrap(ctx) is True
	assert (flag | c_false).unwrap(ctx) is True
	assert (~flag).unwrap(ctx) is False


def test_const_vs_python_literal() -> None:
	c10 = Const("name", 10.0)
	expr = c10 == 10.0
	assert expr.unwrap(None) is True
	assert (c10 != 5.0).unwrap(None) is True
	assert (c10 > 5.0).unwrap(None) is True
	assert (c10 < 20.0).unwrap(None) is True


def test_var_add_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x + c5).unwrap(ctx) == 10
	assert (x + 2).unwrap(ctx) == 7


def test_const_add_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 + x).unwrap(ctx) == 10
	assert (c5 + 2).unwrap(ctx) == 7


def test_var_sub_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x - c5).unwrap(ctx) == 0
	assert (x - 2).unwrap(ctx) == 3


def test_const_sub_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 - x).unwrap(ctx) == 0
	assert (c5 - 2).unwrap(ctx) == 3


def test_var_mul_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x * c5).unwrap(ctx) == 25
	assert (x * 2).unwrap(ctx) == 10


def test_const_mul_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 * x).unwrap(ctx) == 25
	assert (c5 * 2).unwrap(ctx) == 10


def test_var_truediv_const_and_literal() -> None:
	y = Var[float, Ctx]("y")
	c5 = Const("Five", 5.0)
	assert (y / c5).unwrap(ctx) == 2
	assert (y / 2.0).unwrap(ctx) == 5


def test_const_truediv_var_and_literal() -> None:
	y = Var[float, Ctx]("y")
	c10 = Const("Ten", 10)
	assert (c10 / y).unwrap(ctx) == 1
	assert (c10 / 2).unwrap(ctx) == 5


def test_var_eq_const() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x == c5).unwrap(ctx) is True


def test_const_eq_var() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 == x).unwrap(ctx) is True


def test_var_ne_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x != c5).unwrap(ctx) is False
	assert (x != 7).unwrap(ctx) is True


def test_const_ne_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 != x).unwrap(ctx) is False
	assert (c5 != 7).unwrap(ctx) is True


def test_var_lt_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c10 = Const("Ten", 10)
	assert (x < c10).unwrap(ctx) is True
	assert (x < 3).unwrap(ctx) is False


def test_const_lt_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c3 = Const("Three", 3)
	assert (c3 < x).unwrap(ctx) is True
	assert (c3 < 2).unwrap(ctx) is False


def test_var_le_const_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (x <= c5).unwrap(ctx) is True
	assert (x <= 4).unwrap(ctx) is False


def test_const_le_var_and_literal() -> None:
	x = Var[int, Ctx]("x")
	c5 = Const("Five", 5)
	assert (c5 <= x).unwrap(ctx) is True
	assert (c5 <= 4).unwrap(ctx) is False


def test_var_gt_const_and_literal() -> None:
	y = Var[int, Ctx]("y")
	c5 = Const("Five", 5)
	assert (y > c5).unwrap(ctx) is True
	assert (y > 20).unwrap(ctx) is False


def test_const_gt_var_and_literal() -> None:
	y = Var[int, Ctx]("y")
	c20 = Const("Twenty", 20)
	assert (c20 > y).unwrap(ctx) is True
	assert (c20 > 30).unwrap(ctx) is False


def test_var_ge_const_and_literal() -> None:
	y = Var[int, Ctx]("y")
	c10 = Const("Ten", 10)
	assert (y >= c10).unwrap(ctx) is True
	assert (y >= 20).unwrap(ctx) is False


def test_const_ge_var_and_literal() -> None:
	y = Var[int, Ctx]("y")
	c10 = Const("Ten", 10)
	assert (c10 >= y).unwrap(ctx) is True
	assert (c10 >= 20).unwrap(ctx) is False


def test_var_in_range() -> None:
	min = Const("min", 0)
	max = Const("max", 10)
	x = Var[int, Ctx]("x")

	expr = (min <= x) & (x <= max)
	assert expr.unwrap(ctx) is True
	assert expr.to_string() == "((min:0 <= x) & (x <= max:10))"
	assert expr.to_string(ctx) == "((min:0 <= x:5 -> True) & (x:5 <= max:10 -> True) -> True)"

	expr = (max >= x) & (x >= min)
	assert expr.unwrap(ctx) is True
	assert expr.to_string() == "((max:10 >= x) & (x >= min:0))"
	assert expr.to_string(ctx) == "((max:10 >= x:5 -> True) & (x:5 >= min:0 -> True) -> True)"

	expr = (min < x) & (x < max)
	assert expr.unwrap(ctx) is True
	assert expr.to_string() == "((min:0 < x) & (x < max:10))"
	assert expr.to_string(ctx) == "((min:0 < x:5 -> True) & (x:5 < max:10 -> True) -> True)"

	expr = (max > x) & (x > min)
	assert expr.unwrap(ctx) is True
	assert expr.to_string() == "((max:10 > x) & (x > min:0))"
	assert expr.to_string(ctx) == "((max:10 > x:5 -> True) & (x:5 > min:0 -> True) -> True)"


def test_deeply_nested_all_operations() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	f = Var[float, Ctx]("f")
	c5 = Const("Five", 5)
	c10 = Const("Ten", 10)
	c2 = Const("Two", 2)
	c3 = Const("Three", 3)
	c7 = Const("Seven", 7.0)
	c13 = Const("Thirteen", 13.0)

	expr = ~(
		(((x + c5) * (y - c2)) == c3)
		& ((x >= c5) | (y < c10))
		& ((x != 7) & (y <= 20))
		& ((x > 0) & (y > 0))
		| ((f / c7 + c7) < c13)
	)

	assert_type(expr, Not[Ctx])

	result = expr.unwrap(ctx)
	assert isinstance(result, bool)

	s = expr.to_string()
	print(s)
	assert (
		s
		== "(not (((((((x + Five:5) * (y - Two:2)) == Three:3) & ((x >= Five:5) | (y < Ten:10))) & ((x != 7) & (y <= 20))) & ((x > 0) & (y > 0))) | (((f / Seven:7.0) + Seven:7.0) < Thirteen:13.0)))"
	)

	s_ctx = expr.to_string(ctx)
	print(s_ctx)
	assert (
		s_ctx
		== "(not (((((((x:5 + Five:5 -> 10) * (y:10 - Two:2 -> 8) -> 80) == Three:3 -> False) & ((x:5 >= Five:5 -> True) | (y:10 < Ten:10 -> False) -> True) -> False) & ((x:5 != 7 -> True) & (y:10 <= 20 -> True) -> True) -> False) & ((x:5 > 0 -> True) & (y:10 > 0 -> True) -> True) -> False) | (((f:1.5 / Seven:7.0 -> 0.21428571428571427) + Seven:7.0 -> 7.214285714285714) < Thirteen:13.0 -> True) -> True) -> False)"  # noqa: E501
	)

	assert result is False

	# expression is immutable, so multiple evaluations should yield the same result
	for _ in range(5):
		assert expr.unwrap(ctx) is False
		assert expr.to_string() == s
		assert expr.to_string(ctx) == s_ctx
		assert expr.to_string(ctx) == s_ctx


def test_between() -> None:
	x = Var[int, Ctx]("x")

	expr = between(x, 5, 10)
	assert expr.unwrap(ctx) is False
	assert expr.to_string() == "((Low:5 < x) & (x < High:10))"
	assert expr.to_string(ctx) == "((Low:5 < x:5 -> False) & (x:5 < High:10 -> True) -> False)"

	expr = between(x, 0, 10)
	assert expr.unwrap(ctx) is True
	assert expr.to_string() == "((Low:0 < x) & (x < High:10))"
	assert expr.to_string(ctx) == "((Low:0 < x:5 -> True) & (x:5 < High:10 -> True) -> True)"

	expr = (Const("Low", 5) < x) & (x < Const("High", 10))
	print()
	print(expr.to_string())
	print()
	print(expr.to_string(ctx))


def test_manual_within() -> None:
	f = Var[float, Ctx]("f")

	expr = (f - Const("Target", 1.5)) < Const("Tolerance", 0.01)
	print()
	print(expr.to_string())
	print()
	print(expr.to_string(ctx))


def test_approximately() -> None:
	x = Var[float, Ctx]("x")

	FIVE = PlusMinus("Five", 4.9, 0.1)
	assert_type(FIVE, PlusMinus[float])

	expr = Approximately(x, FIVE)
	assert_type(expr, Approximately[float, Ctx])
	print()
	print(expr.to_string())
	print(expr.to_string(ctx))

	FIVE_ = Percent("Five", 5.0, 1.0)
	assert_type(FIVE_, Percent[float])

	expr = Approximately(x, FIVE_)
	assert_type(expr, Approximately[float, Ctx])

	print()
	print(expr.to_string())
	print(expr.to_string(ctx))


def test_approximately_composition() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	x_plus_y = x + y
	SUM = PlusMinus("Sum", 15, 0.1)

	expr = Approximately(x_plus_y, SUM)
	assert_type(expr, Approximately[int, Ctx])

	print()
	print(expr.to_string())
	print(expr.to_string(ctx))

	expr = Approximately(x * y, Percent("Product", 48.0, 5.0))
	assert_type(expr, Approximately[int, Ctx])

	print()
	print(expr.to_string())
	print(expr.to_string(ctx))


def test_composition_nested_arithmetic() -> None:
	"""Test composition of arithmetic expressions at multiple levels."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Build nested arithmetic expressions
	inner = (x + y) * 2
	outer = inner - 5
	final = outer / 3

	assert final.unwrap(ctx) == ((5 + 10) * 2 - 5) / 3
	assert final.to_string() == "((((x + y) * 2) - 5) / 3)"

	# Test with constants mixed in
	c2 = Const("Two", 2)
	c5 = Const("Five", 5)
	composed = (x + c2) * (y - c5)
	assert composed.unwrap(ctx) == (5 + 2) * (10 - 5)
	assert composed.to_string() == "((x + Two:2) * (y - Five:5))"


def test_composition_comparison_chains() -> None:
	"""Test composition of comparison expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Chain comparisons with logical operators
	range_check = (0 < x) & (x < 10) & (y > x)
	assert range_check.unwrap(ctx) is True

	complex_comparison = ((x + y) > 10) & ((x * y) < 100)
	assert complex_comparison.unwrap(ctx) is True
	assert complex_comparison.to_string() == "(((x + y) > 10) & ((x * y) < 100))"


def test_composition_mixed_operations() -> None:
	"""Test composition mixing arithmetic, comparison, & logical operations."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Arithmetic inside comparisons inside logical operations
	expr = ((x + 3) == (y - 2)) | ((x * 2) > y)
	assert expr.unwrap(ctx) is True  # (5+3) == (10-2) is True

	# Nested logical with arithmetic
	complex_expr = ~(((x + y) < 20) & ((x - y) > -10))
	assert complex_expr.unwrap(ctx) is False


def test_composition_with_constants() -> None:
	"""Test composition where constants are used throughout the expression tree."""
	x = Var[int, Ctx]("x")
	base = Const("Base", 10)
	multiplier = Const("Mult", 3)
	threshold = Const("Threshold", 25)

	# Build expression using constants at different levels
	scaled = (x + base) * multiplier
	comparison = scaled > threshold

	assert scaled.unwrap(ctx) == (5 + 10) * 3  # 45
	assert comparison.unwrap(ctx) is True  # 45 > 25
	assert comparison.to_string() == "(((x + Base:10) * Mult:3) > Threshold:25)"


def test_composition_deeply_nested_logical() -> None:
	"""Test deeply nested logical expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Build deeply nested logical expression
	a = x > 0
	b = y > 0
	c = x < 10
	d = y < 20

	nested = (a & b) | ~(c & d)
	assert nested.unwrap(ctx) is True

	# Even deeper nesting
	deep = ((a & b) | (c & d)) & ~((a | b) & (c | d))
	assert isinstance(deep.unwrap(ctx), bool)


def test_composition_with_function_calls() -> None:
	"""Test composition with special functions like between & approximately."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Use between in larger expressions
	x_in_range = between(x, 0, 10)
	y_in_range = between(y, 5, 15)
	both_in_range = x_in_range & y_in_range

	assert both_in_range.unwrap(ctx) is True

	# Compose with arithmetic
	sum_expr = x + y
	sum_in_range = between(sum_expr, 10, 20)
	assert sum_in_range.unwrap(ctx) is True


def test_composition_with_approximation() -> None:
	"""Test composition with approximation operations."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")

	# Build expressions that use approximation
	product = x * y
	target = PlusMinus("Target", 50.0, 5.0)
	approx_check = Approximately(product, target)

	# Compose with other conditions
	range_check = (x > 0) & (y > 0)
	full_check = range_check & approx_check

	assert full_check.unwrap(ctx) is True
	assert "≈" in full_check.to_string()
	assert (
		full_check.to_string(ctx)
		== "(((x:5 > 0 -> True) & (y:10 > 0 -> True) -> True) & ((x:5 * y:10 -> 50) ≈ Target:50.0 ± 5.0 -> True) -> True)"
	)  # noqa: E501


def test_composition_reuse_subexpressions() -> None:
	"""Test reusing the same subexpression in multiple places."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Create a subexpression & reuse it
	sum_expr = x + y
	diff_expr = x - y

	# Use both in different contexts
	sum_condition = sum_expr > 10
	diff_condition = diff_expr < 10
	product_expr = sum_expr * diff_expr
	product_condition = product_expr > 0

	combined = sum_condition & diff_condition & product_condition
	assert combined.unwrap(ctx) is False

	# Verify the subexpressions maintain their identity
	assert sum_expr.unwrap(ctx) == 15
	assert diff_expr.unwrap(ctx) == -5
	assert product_expr.unwrap(ctx) == -75


def test_composition_with_negation() -> None:
	"""Test composition with negation at different levels."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Negation of simple comparison
	not_equal = ~(x == y)
	assert not_equal.unwrap(ctx) is True

	# Negation of complex expression
	complex_expr = (x > 0) & (y > 0) & (x < y)
	negated_complex = ~complex_expr
	assert negated_complex.unwrap(ctx) is False

	# Double negation
	double_neg = ~~(x == 5)
	assert double_neg.unwrap(ctx) is True


def test_composition_type_consistency() -> None:
	"""Test that type consistency is maintained through composition."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	f = Var[float, Ctx]("f")

	# Mix int & float operations
	mixed_expr = (x + f) > (y * 2.0)
	assert isinstance(mixed_expr.unwrap(ctx), bool)

	# Ensure arithmetic results maintain proper types
	int_result = x + y
	float_result = f * 2.0

	assert isinstance(int_result.unwrap(ctx), int)
	assert isinstance(float_result.unwrap(ctx), float)


def test_composition_immutability() -> None:
	"""Test that composed expressions are immutable."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Create base expressions
	base_expr = x + y

	# Create composed expressions
	expr1 = base_expr > 10
	expr2 = base_expr < 20

	# Verify that using base_expr in multiple places doesn't affect results
	result1 = expr1.unwrap(ctx)
	result2 = expr2.unwrap(ctx)

	# Re-evaluate to ensure immutability
	assert expr1.unwrap(ctx) == result1
	assert expr2.unwrap(ctx) == result2
	assert base_expr.unwrap(ctx) == 15  # Original value unchanged


def test_composition_string_representation() -> None:
	"""Test that string representations work correctly for composed expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Build a complex composed expression
	expr = ((x + y) * 2) > ((x - y) + 10)

	# Test symbolic representation
	symbolic = expr.to_string()
	assert "((x + y) * 2)" in symbolic
	assert "((x - y) + 10)" in symbolic
	assert ">" in symbolic

	# Test evaluated representation
	evaluated = expr.to_string(ctx)
	assert "15" in evaluated  # x + y = 15
	assert "30" in evaluated  # (x + y) * 2 = 30
	assert "-5" in evaluated  # x - y = -5
	assert "5" in evaluated  # (x - y) + 10 = 5
	assert "True" in evaluated or "False" in evaluated

	# Repeat using sub expressions
	sum_expr = x + y
	diff_expr = x - y
	sum_expr_x2 = sum_expr * 2
	diff_expr_plus_10 = diff_expr + 10
	expr2 = sum_expr_x2 > diff_expr_plus_10
	assert expr2.to_string() == expr.to_string()
	assert expr2.to_string(ctx) == expr.to_string(ctx)


def test_predicate() -> None:
	"""Test Predicate class."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Create a predicate
	pred = Predicate("x is greater than y", x > y)

	assert pred.unwrap(ctx) is False  # 5 is not greater than 10
	assert pred.to_string() == "x is greater than y: (x > y)"
	assert pred.to_string(ctx) == "x is greater than y: False (x:5 > y:10 -> False)"

	pred2 = Predicate("x is less than y", x < y)
	assert pred2.unwrap(ctx) is True  # 5 is less than 10
	assert pred2.to_string() == "x is less than y: (x < y)"
	assert pred2.to_string(ctx) == "x is less than y: True (x:5 < y:10 -> True)"

	class Measurement(NamedTuple):
		voltage: float

	voltage = Var[float, Measurement]("voltage")

	voltage_pred = Predicate(
		"Voltage is within range", Approximately(voltage, PlusMinus("Target", 5.0, 0.1))
	)

	assert voltage_pred.unwrap(Measurement(voltage=5.05)) is True
	assert voltage_pred.to_string() == "Voltage is within range: (voltage ≈ Target:5.0 ± 0.1)"
	assert (
		voltage_pred.to_string(Measurement(voltage=5.05))
		== "Voltage is within range: True (voltage:5.05 ≈ Target:5.0 ± 0.1 -> True)"
	)
	assert voltage_pred.unwrap(Measurement(voltage=5.15)) is False
	assert (
		voltage_pred.to_string(Measurement(voltage=5.15))
		== "Voltage is within range: False (voltage:5.15 ≈ Target:5.0 ± 0.1 -> False)"
	)


@pytest.mark.mypy_testing
def test_bound_expr_type() -> None:
	"""Test Closure type with Predicate."""

	class Measurement(NamedTuple):
		voltage: float

	voltage = Var[float, Measurement]("voltage")

	m = Measurement(voltage=5.05)
	voltage_pred = Predicate(
		"Voltage is within range", Approximately(voltage, PlusMinus("Target", 5.0, 0.1))
	)

	closure = voltage_pred.bind(m)
	assert_type(closure, BoundExpr[bool, Measurement])


def test_bind() -> None:
	class Measurement(NamedTuple):
		voltage: float

	voltage = Var[float, Measurement]("voltage")

	m = Measurement(voltage=5.05)

	voltage_pred = Predicate(
		"Voltage is within range", Approximately(voltage, PlusMinus("Target", 5.0, 0.1))
	)

	closure = voltage_pred.bind(m)
	assert closure.unwrap() is True
	assert closure.expr is voltage_pred
	assert closure.ctx is m
	print(closure)
	assert str(closure) == "Voltage is within range: True (voltage:5.05 ≈ Target:5.0 ± 0.1 -> True)"


def test_bind_predicate() -> None:
	"""Test binding a Predicate to a context."""

	class Measurement(NamedTuple):
		voltage: float

	voltage = Var[float, Measurement]("voltage")

	m = Measurement(voltage=5.05)

	expr = Predicate(
		"Voltage is within range", Approximately(voltage, PlusMinus("Target", 5.0, 0.1))
	)
	predicate = expr.bind(m)

	assert predicate.unwrap() is True
	assert predicate.expr is expr
	assert predicate.ctx is m
	print(predicate.ctx)
	print(predicate)
	assert (
		str(predicate) == "Voltage is within range: True (voltage:5.05 ≈ Target:5.0 ± 0.1 -> True)"
	)
