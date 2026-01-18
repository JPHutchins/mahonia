# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Final, NamedTuple, assert_type

import pytest

from mahonia import (
	Add,
	And,
	Approximately,
	BoundExpr,
	Const,
	Div,
	Eq,
	Expr,
	Ge,
	Gt,
	Le,
	Lt,
	Max,
	MergeContextProtocol,
	Min,
	Mul,
	Ne,
	Not,
	Or,
	Percent,
	PlusMinus,
	Pow,
	Predicate,
	Sub,
	TSupportsComparison,
	Var,
	merge,
)


@dataclass(frozen=True)
class Ctx:
	x: int
	y: int
	name: str
	flag: bool = True
	custom: object = object()
	f: float = 1.5
	e: float = 2.71828


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


def test_literal_radd_var() -> None:
	x = Var[int, Ctx]("x")
	expr = 1 + x
	assert_type(expr, Add[int, Ctx])
	assert expr == Add(Const(None, 1), x)
	assert expr.unwrap(ctx) == 6


def test_literal_rsub_var() -> None:
	x = Var[int, Ctx]("x")
	expr = 10 - x
	assert_type(expr, Sub[int, Ctx])
	assert expr == Sub(Const(None, 10), x)
	assert expr.unwrap(ctx) == 5


def test_literal_rmul_var() -> None:
	x = Var[int, Ctx]("x")
	expr = 3 * x
	assert_type(expr, Mul[int, Ctx])
	assert expr == Mul(Const(None, 3), x)
	assert expr.unwrap(ctx) == 15


def test_literal_rtruediv_var() -> None:
	y = Var[float, Ctx]("y")
	expr = 20.0 / y
	assert_type(expr, Div[float, Ctx])
	assert expr == Div(Const(None, 20.0), y)
	assert expr.unwrap(ctx) == 2.0


def test_literal_rpow_var() -> None:
	x = Var[int, Ctx]("x")
	expr = 2**x
	assert_type(expr, Pow[int, Ctx])
	assert expr == Pow(Const(None, 2), x)
	assert expr.unwrap(ctx) == 32


def test_literal_rand_expr() -> None:
	x = Var[int, Ctx]("x")
	expr = True & (x > 0)
	assert_type(expr, And[Any, Ctx])
	assert expr.left == Const(None, True)
	assert expr.unwrap(ctx) is True


def test_literal_ror_expr() -> None:
	x = Var[int, Ctx]("x")
	expr = False | (x > 0)
	assert_type(expr, Or[Any, Ctx])
	assert expr.left == Const(None, False)
	assert expr.unwrap(ctx) is True


def test_literal_lt_var_reflects_to_gt() -> None:
	x = Var[int, Ctx]("x")
	expr = 3 < x
	assert_type(expr, Gt[int, Ctx])
	assert expr == Gt(x, Const(None, 3))
	assert expr.unwrap(ctx) is True


def test_literal_le_var_reflects_to_ge() -> None:
	x = Var[int, Ctx]("x")
	expr = 5 <= x
	assert_type(expr, Ge[int, Ctx])
	assert expr == Ge(x, Const(None, 5))
	assert expr.unwrap(ctx) is True


def test_literal_gt_var_reflects_to_lt() -> None:
	x = Var[int, Ctx]("x")
	expr = 10 > x
	assert_type(expr, Lt[int, Ctx])
	assert expr == Lt(x, Const(None, 10))
	assert expr.unwrap(ctx) is True


def test_literal_ge_var_reflects_to_le() -> None:
	x = Var[int, Ctx]("x")
	expr = 5 >= x
	assert_type(expr, Le[int, Ctx])
	assert expr == Le(x, Const(None, 5))
	assert expr.unwrap(ctx) is True


def test_literal_eq_var() -> None:
	x = Var[int, Ctx]("x")
	# Static type checker thinks int.__eq__ returns bool, but at runtime it returns
	# NotImplemented for unknown types, causing Python to call Var.__eq__ instead.
	expr = 5 == x  # type: ignore[comparison-overlap]
	assert isinstance(expr, Eq)
	assert expr == Eq(x, Const(None, 5))
	assert expr.unwrap(ctx) is True


def test_literal_ne_var() -> None:
	x = Var[int, Ctx]("x")
	# Static type checker thinks int.__ne__ returns bool, but at runtime it returns
	# NotImplemented for unknown types, causing Python to call Var.__ne__ instead.
	expr = 10 != x  # type: ignore[comparison-overlap]
	assert isinstance(expr, Ne)
	assert expr == Ne(x, Const(None, 10))
	assert expr.unwrap(ctx) is True


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
	f = Var[float, Ctx]("f")
	e = Var[float, Ctx]("e")

	f_plus_e = f + e
	SUM = PlusMinus("Sum", 15, 0.1)

	expr = Approximately(f_plus_e, SUM)
	assert_type(expr, Approximately[float, Ctx])

	print()
	print(expr.to_string())
	print(expr.to_string(ctx))

	expr = Approximately(f * e, Percent("Product", 48.0, 5.0))
	assert_type(expr, Approximately[float, Ctx])

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


def test_pow() -> None:
	"""Test power operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Create a power expression
	pow_expr = x**2
	assert_type(pow_expr, Pow[int, Ctx])

	assert pow_expr.unwrap(ctx) == 25
	assert pow_expr.to_string() == "(x^2)"
	assert pow_expr.to_string(ctx) == "(x:5^2 -> 25)"

	pow_expr = x**y
	assert pow_expr.unwrap(ctx) == 5**10
	assert pow_expr.to_string() == "(x^y)"
	assert pow_expr.to_string(ctx) == "(x:5^y:10 -> 9765625)"
	assert pow_expr.unwrap(ctx) == 9765625

	pow_expr1 = Const(None, 2) ** x
	assert_type(pow_expr1, Pow[int, Any])
	assert pow_expr1.unwrap(ctx) == 2**5
	assert pow_expr1.to_string() == "(2^x)"


def test_approximately_coercion() -> None:
	"""Test coercion of Approximately to BoundExpr."""
	x = Var[float, Ctx]("x")
	target = PlusMinus("Target", 5.0, 0.1)

	expr = target == x
	assert_type(expr, Approximately[float, Ctx])
	print(expr.to_string())

	expr = x == target
	assert_type(expr, Approximately[float, Ctx])
	print(expr.to_string())

	expr1 = target == 5.0
	assert_type(expr1, Approximately[float, Any])
	print(expr1.to_string())


def test_context_merge() -> None:
	"""Test merging context values with merge function."""

	class A(NamedTuple):
		a: int

	class B(NamedTuple):
		b: int

	c = merge(A(a=1), B(b=2))
	assert c.a == 1
	assert c.b == 2


def test_context_merge_arity() -> None:
	"""Test merging contexts with a single field."""

	class SingleFieldCtx(NamedTuple):
		value: int

	merged = merge(
		SingleFieldCtx(value=42),
	)

	assert merged.value == 42

	merge()


def test_context_merge_typing() -> None:
	"""Test typing of merged contexts."""

	class CtxA(NamedTuple):
		a: int

	class CtxB(NamedTuple):
		b: str

	merged = merge(CtxA(a=10), CtxB(b="test"))
	assert_type(merged, MergeContextProtocol[CtxA, CtxB])


def test_context_merge_multiple_fields() -> None:
	"""Test merging contexts with multiple fields."""

	class VoltageCtx(NamedTuple):
		voltage: float
		current: float

	class TempCtx(NamedTuple):
		temperature: float
		humidity: float

	class DurationCtx(NamedTuple):
		duration: float
		unit: str

	measurement = merge(
		VoltageCtx(voltage=3.3, current=0.5),
		TempCtx(temperature=25.0, humidity=60.0),
		DurationCtx(duration=10.0, unit="seconds"),
	)

	assert measurement.voltage == 3.3
	assert measurement.current == 0.5
	assert measurement.temperature == 25.0
	assert measurement.humidity == 60.0
	assert measurement.duration == 10.0
	assert measurement.unit == "seconds"


def test_context_merge_with_expressions() -> None:
	"""Test using merged contexts with expressions."""

	class XCtx(NamedTuple):
		x: int

	class YCtx(NamedTuple):
		y: int

	ctx = merge(XCtx(x=5), YCtx(y=10))

	x = Var[int, Any]("x")
	y = Var[int, Any]("y")

	expr = x + y

	assert expr.unwrap(ctx) == 15
	assert expr.to_string(ctx) == "(x:5 + y:10 -> 15)"


def test_context_merge_conflict() -> None:
	"""Test merging contexts with conflicting field names."""

	class A(NamedTuple):
		value: int

	class B(NamedTuple):
		value: int

	with pytest.raises(TypeError):
		merge(A(value=1), B(value=2))


def test_partial_application() -> None:
	"""Partial context resolves some vars, leaves others as Var."""

	class XCtx(NamedTuple):
		x: int

	class YCtx(NamedTuple):
		y: int

	x = Var[int, Any]("x")
	y = Var[int, Any]("y")
	expr = x + y

	partial_expr = expr.partial(XCtx(x=5))
	assert_type(partial_expr, Expr[int, Any])

	assert partial_expr.to_string() == "(x:5 + y)"

	result = partial_expr.unwrap(YCtx(y=10))
	assert_type(result, int)
	assert result == 15

	assert partial_expr.to_string(YCtx(y=10)) == "(x:5 + y:10 -> 15)"

	# prevent type narrowing from polluting above results
	assert isinstance(partial_expr, Add)
	assert isinstance(partial_expr.left, Const)
	assert isinstance(partial_expr.right, Var)


def test_partial_application_multiple() -> None:
	"""Partial application with multiple context fields."""

	class XCtx(NamedTuple):
		x: int

	class YCtx(NamedTuple):
		y: int

	class ZCtx(NamedTuple):
		z: int

	x = Var[int, Any]("x")
	y = Var[int, Any]("y")
	z = Var[int, Any]("z")
	expr = x * y + z

	partial_expr = expr.partial(merge(XCtx(x=5), YCtx(y=10)))
	assert_type(partial_expr, Expr[int, Any])

	assert partial_expr.to_string() == "((x:5 * y:10) + z)"

	result = partial_expr.unwrap(ZCtx(z=3))
	assert_type(result, int)
	assert result == 53

	assert partial_expr.to_string(ZCtx(z=3)) == "((x:5 * y:10 -> 50) + z:3 -> 53)"


def test_partial_application_exhausted() -> None:
	"""Partial application that resolves all Vars results in Const."""

	class XCtx(NamedTuple):
		x: int

	class YCtx(NamedTuple):
		y: int

	x = Var[int, Any]("x")
	y = Var[int, Any]("y")
	expr = x + y

	partial_expr = expr.partial(merge(XCtx(x=5), YCtx(y=10)))
	assert_type(partial_expr, Expr[int, Any])

	assert partial_expr.to_string() == "(x:5 + y:10)"
	assert partial_expr.to_string(()) == "(x:5 + y:10 -> 15)"

	result = partial_expr.unwrap(None)
	assert_type(result, int)
	assert result == 15


def test_partial_application_preserves_structure() -> None:
	"""Partial application preserves expression tree - no work is lost."""

	class ABCtx(NamedTuple):
		a: int
		b: int

	class CDCtx(NamedTuple):
		c: int
		d: int

	a = Var[int, Any]("a")
	b = Var[int, Any]("b")
	c = Var[int, Any]("c")
	d = Var[int, Any]("d")

	# Deeply nested: ((a + b) * (c - d)) ** 2
	expr = ((a + b) * (c - d)) ** Const(None, 2)

	# Partial with only a and b
	partial1 = expr.partial(ABCtx(a=3, b=2))
	assert_type(partial1, Expr[int, Any])
	assert partial1.to_string() == "(((a:3 + b:2) * (c - d))^2)"

	# The structure is preserved: Add and Sub are still there, not collapsed
	assert isinstance(partial1, Pow)
	assert isinstance(partial1.left, Mul)
	assert isinstance(partial1.left.left, Add)
	assert isinstance(partial1.left.right, Sub)

	# Left side fully resolved to Const, right side still has Vars
	assert isinstance(partial1.left.left.left, Const)
	assert isinstance(partial1.left.left.right, Const)
	assert isinstance(partial1.left.right.left, Var)
	assert isinstance(partial1.left.right.right, Var)

	# Complete the evaluation
	partial2 = partial1.partial(CDCtx(c=10, d=3))
	assert partial2.to_string() == "(((a:3 + b:2) * (c:10 - d:3))^2)"

	# All leaves are now Const but structure is intact
	assert isinstance(partial2, Pow)
	assert isinstance(partial2.left, Mul)
	assert isinstance(partial2.left.left, Add)
	assert isinstance(partial2.left.right, Sub)

	# Final evaluation: ((3+2) * (10-3)) ** 2 = (5 * 7) ** 2 = 35 ** 2 = 1225
	assert partial2.unwrap(()) == 1225
	assert partial2.to_string(()) == "(((a:3 + b:2 -> 5) * (c:10 - d:3 -> 7) -> 35)^2 -> 1225)"


def test_bound_expr_satisfies_expr_protocol() -> None:
	"""BoundExpr satisfies the Expr protocol as a closed term."""

	class Ctx(NamedTuple):
		x: int
		y: int

	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y
	ctx = Ctx(x=5, y=10)

	bound = expr.bind(ctx)

	# BoundExpr satisfies Expr protocol
	assert isinstance(bound, Expr)

	# eval ignores passed context, uses captured context
	assert bound.eval(()).value == 15
	assert bound.eval(None).value == 15

	# to_string ignores passed context
	assert bound.to_string() == "(x:5 + y:10 -> 15)"
	assert bound.to_string(()) == "(x:5 + y:10 -> 15)"

	# partial returns self (closed term)
	assert bound.partial(()) is bound

	# bind returns self (already bound)
	assert bound.bind(()) is bound

	# unwrap works with any context
	assert bound.unwrap() == 15
	assert bound.unwrap(None) == 15


def test_bound_expr_composable() -> None:
	"""BoundExpr can be composed with other expressions."""

	class XCtx(NamedTuple):
		x: int

	class YCtx(NamedTuple):
		y: int

	x = Var[int, XCtx]("x")
	y = Var[int, YCtx]("y")

	# Bind x to 5
	bound_x = x.bind(XCtx(x=5))

	# Compose bound expression with unbound expression
	composed = bound_x + y
	assert_type(composed, Add[int, Any])

	# The composed expression needs y's context
	# BoundExpr.to_string() always shows evaluated form since it has its context
	assert composed.to_string() == "(x:5 + y)"
	assert composed.unwrap(YCtx(y=10)) == 15
	assert composed.to_string(YCtx(y=10)) == "(x:5 + y:10 -> 15)"


def test_bound_expr_boolean_composition() -> None:
	"""BoundExpr supports boolean composition."""

	class Ctx(NamedTuple):
		x: int

	x = Var[int, Ctx]("x")

	# Bind a boolean expression - use explicit type annotation
	# Note: Gt inherits from BinaryOp[TSupportsComparison, S] so T is int, not bool
	# The eval() returns Const[bool] via override, but bind() uses the class T parameter
	bound_true: BoundExpr[bool, Ctx] = (x > 0).bind(Ctx(x=5))  # type: ignore[assignment]
	bound_false: BoundExpr[bool, Ctx] = (x < 0).bind(Ctx(x=5))  # type: ignore[assignment]

	# Compose with & and |
	combined_and = bound_true & bound_false
	combined_or = bound_true | bound_false

	assert combined_and.unwrap(()) is False
	assert combined_or.unwrap(()) is True

	# Invert
	inverted = ~bound_true
	assert inverted.unwrap(()) is False


def test_bound_expr_arithmetic_composition() -> None:
	"""BoundExpr supports arithmetic composition."""

	class Ctx(NamedTuple):
		x: int

	x = Var[int, Ctx]("x")
	bound = x.bind(Ctx(x=10))

	# Arithmetic operators (note: reverse operators like 5 + bound not supported by mixin)
	assert (bound + 5).unwrap(()) == 15
	assert (bound - 3).unwrap(()) == 7
	assert (bound * 2).unwrap(()) == 20
	assert (bound / 2).unwrap(()) == 5.0


def test_min_binary_op() -> None:
	"""Test Min binary operation."""

	class Ctx(NamedTuple):
		a: int
		b: int

	a = Var[int, Ctx]("a")
	b = Var[int, Ctx]("b")

	min_expr = Min(a, b)
	assert min_expr.to_string() == "(min a b)"

	ctx = Ctx(a=5, b=3)
	assert min_expr.unwrap(ctx) == 3
	assert min_expr.to_string(ctx) == "(min a:5 b:3 -> 3)"

	ctx2 = Ctx(a=2, b=7)
	assert min_expr.unwrap(ctx2) == 2


def test_max_binary_op() -> None:
	"""Test Max binary operation."""

	class Ctx(NamedTuple):
		a: int
		b: int

	a = Var[int, Ctx]("a")
	b = Var[int, Ctx]("b")

	max_expr = Max(a, b)
	assert max_expr.to_string() == "(max a b)"

	ctx = Ctx(a=5, b=3)
	assert max_expr.unwrap(ctx) == 5
	assert max_expr.to_string(ctx) == "(max a:5 b:3 -> 5)"

	ctx2 = Ctx(a=2, b=7)
	assert max_expr.unwrap(ctx2) == 7


def test_min_max_with_const() -> None:
	"""Test Min/Max with Const values."""

	class Ctx(NamedTuple):
		x: float

	x = Var[float, Ctx]("x")
	threshold = Const("threshold", 10.0)

	clamped_above = Max(x, threshold)
	clamped_below = Min(x, threshold)

	ctx_low = Ctx(x=5.0)
	assert clamped_above.unwrap(ctx_low) == 10.0
	assert clamped_below.unwrap(ctx_low) == 5.0

	ctx_high = Ctx(x=15.0)
	assert clamped_above.unwrap(ctx_high) == 15.0
	assert clamped_below.unwrap(ctx_high) == 10.0


def test_min_max_chained() -> None:
	"""Test chaining Min/Max operations (clamp pattern)."""

	class Ctx(NamedTuple):
		x: float

	x = Var[float, Ctx]("x")
	lower = Const("lower", 0.0)
	upper = Const("upper", 100.0)

	clamped = Min(Max(x, lower), upper)

	assert clamped.unwrap(Ctx(x=-10.0)) == 0.0
	assert clamped.unwrap(Ctx(x=50.0)) == 50.0
	assert clamped.unwrap(Ctx(x=150.0)) == 100.0
