# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Callable, Final, NamedTuple, assert_type

import pytest

from mahonia import (
	Abs,
	Add,
	AllExpr,
	And,
	AnyExpr,
	Approximately,
	BoolVar,
	BoundExpr,
	Clamp,
	Const,
	Contains,
	Div,
	Eq,
	Expr,
	FloatVar,
	FoldLExpr,
	Func,
	Ge,
	Gt,
	IntVar,
	Le,
	ListVar,
	Lt,
	MapExpr,
	Match,
	Max,
	MaxExpr,
	MergeContextProtocol,
	Min,
	MinExpr,
	Mul,
	Ne,
	Neg,
	Not,
	Or,
	Percent,
	PlusMinus,
	Pow,
	Predicate,
	SizedIterable,
	StrVar,
	Sub,
	TSupportsComparison,
	Var,
	context_vars,
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


class ElemCtx(NamedTuple):
	n: int


class ContainerCtx(NamedTuple):
	nums: list[int]


class ContainsCtx(NamedTuple):
	values: list[int]
	target: int


class FlagsCtx(NamedTuple):
	flags: list[bool]


class ValuesCtx(NamedTuple):
	values: list[int]


class PredicateCtx(NamedTuple):
	x: int


ctx: Final = Ctx(x=5, y=10, name="example")


def between(
	expr: Expr[TSupportsComparison, Ctx, TSupportsComparison],
	low: TSupportsComparison,
	high: TSupportsComparison,
) -> "And[bool, Ctx]":
	"""Example of defining some convenience to compose an expression."""
	return And(
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


def test_neg() -> None:
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr: Neg[int, Ctx] = -x
	assert_type(expr, Neg[int, Ctx])
	assert expr.unwrap(ctx) == -5
	assert expr.to_string() == "(-x)"
	assert expr.to_string(ctx) == "(-x:5 -> -5)"

	composed = -x + y
	assert composed.unwrap(ctx) == 5
	assert composed.to_string() == "((-x) + y)"

	double_neg: Neg[int, Ctx] = -(-x)
	assert double_neg.unwrap(ctx) == 5
	assert double_neg.to_string() == "(-(-x))"


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
	assert_type(expr, And[bool, Ctx])
	assert expr.left == Const(None, True)
	assert expr.unwrap(ctx) is True


def test_literal_ror_expr() -> None:
	x = Var[int, Ctx]("x")
	expr = False | (x > 0)
	assert_type(expr, Or[bool, Ctx])
	assert expr.left == Const(None, False)
	assert expr.unwrap(ctx) is True


def test_literal_lt_var_reflects_to_gt() -> None:
	x = Var[int, Ctx]("x")
	expr = 3 < x
	assert_type(expr, Gt[int, Ctx])
	assert expr == Gt(x, Const(None, 3))  # type: ignore[misc]
	assert expr.unwrap(ctx) is True


def test_literal_le_var_reflects_to_ge() -> None:
	x = Var[int, Ctx]("x")
	expr = 5 <= x
	assert_type(expr, Ge[int, Ctx])
	assert expr == Ge(x, Const(None, 5))  # type: ignore[misc]
	assert expr.unwrap(ctx) is True


def test_literal_gt_var_reflects_to_lt() -> None:
	x = Var[int, Ctx]("x")
	expr = 10 > x
	assert_type(expr, Lt[int, Ctx])
	assert expr == Lt(x, Const(None, 10))  # type: ignore[misc]
	assert expr.unwrap(ctx) is True


def test_literal_ge_var_reflects_to_le() -> None:
	x = Var[int, Ctx]("x")
	expr = 5 >= x
	assert_type(expr, Le[int, Ctx])
	assert expr == Le(x, Const(None, 5))  # type: ignore[misc]
	assert expr.unwrap(ctx) is True


def test_literal_eq_var() -> None:
	x = Var[int, Ctx]("x")
	# Static type checker thinks int.__eq__ returns bool, but at runtime it returns
	# NotImplemented for unknown types, causing Python to call Var.__eq__ instead.
	expr = 5 == x
	assert isinstance(expr, Eq)
	assert expr == Eq(x, Const(None, 5))
	assert expr.unwrap(ctx) is True


def test_literal_ne_var() -> None:
	x = Var[int, Ctx]("x")
	# Static type checker thinks int.__ne__ returns bool, but at runtime it returns
	# NotImplemented for unknown types, causing Python to call Var.__ne__ instead.
	expr = 10 != x
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
	assert_type(closure, BoundExpr[bool, Measurement, bool])


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
	assert_type(partial_expr, Expr[int, Any, int])

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
	assert_type(partial_expr, Expr[int, Any, int])

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
	assert_type(partial_expr, Expr[int, Any, int])

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
	assert_type(partial1, Expr[int, Any, int])
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
	x = Var[int, PredicateCtx]("x")

	bound_true = (x > 0).bind(PredicateCtx(x=5))
	assert_type(bound_true, BoundExpr[int, PredicateCtx, bool])
	bound_false = (x < 0).bind(PredicateCtx(x=5))
	assert_type(bound_false, BoundExpr[int, PredicateCtx, bool])

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


@pytest.mark.mypy_testing
def test_comparison_result_types() -> None:
	"""Verify comparison ops have R=bool."""
	x = Var[int, Ctx]("x")

	gt_expr = x > 5
	assert_type(gt_expr, Gt[int, Ctx])
	assert_type(gt_expr.eval(ctx), Const[bool])
	assert_type(gt_expr.unwrap(ctx), bool)

	lt_expr = x < 10
	assert_type(lt_expr, Lt[int, Ctx])
	assert_type(lt_expr.eval(ctx), Const[bool])
	assert_type(lt_expr.unwrap(ctx), bool)

	ge_expr = x >= 5
	assert_type(ge_expr, Ge[int, Ctx])
	assert_type(ge_expr.eval(ctx), Const[bool])

	le_expr = x <= 10
	assert_type(le_expr, Le[int, Ctx])
	assert_type(le_expr.eval(ctx), Const[bool])

	eq_expr = x == 5
	assert_type(eq_expr, Eq[int, Ctx])
	assert_type(eq_expr.eval(ctx), Const[bool])

	ne_expr = x != 5
	assert_type(ne_expr, Ne[int, Ctx])
	assert_type(ne_expr.eval(ctx), Const[bool])


@pytest.mark.mypy_testing
def test_arithmetic_result_types() -> None:
	"""Verify arithmetic ops preserve operand type."""
	x = Var[int, Ctx]("x")

	add_expr = x + 5
	assert_type(add_expr, Add[int, Ctx])
	assert_type(add_expr.eval(ctx), Const[int])
	assert_type(add_expr.unwrap(ctx), int)

	sub_expr = x - 3
	assert_type(sub_expr, Sub[int, Ctx])
	assert_type(sub_expr.eval(ctx), Const[int])

	mul_expr = x * 2
	assert_type(mul_expr, Mul[int, Ctx])
	assert_type(mul_expr.eval(ctx), Const[int])

	pow_expr = x**2
	assert_type(pow_expr, Pow[int, Ctx])
	assert_type(pow_expr.eval(ctx), Const[int])


@pytest.mark.mypy_testing
def test_bound_expr_result_type() -> None:
	"""Verify BoundExpr preserves R type."""
	x = Var[int, Ctx]("x")

	bound_comparison = (x > 5).bind(ctx)
	assert_type(bound_comparison, BoundExpr[int, Ctx, bool])
	assert_type(bound_comparison.unwrap(), bool)

	bound_arithmetic = (x + 5).bind(ctx)
	assert_type(bound_arithmetic, BoundExpr[int, Ctx, int])
	assert_type(bound_arithmetic.unwrap(), int)

	bound_var = x.bind(ctx)
	assert_type(bound_var, BoundExpr[int, Ctx, int])
	assert_type(bound_var.unwrap(), int)


@pytest.mark.mypy_testing
def test_logical_composition_types() -> None:
	"""Verify logical ops result type."""
	x = Var[int, Ctx]("x")

	and_expr = (x > 0) & (x < 10)
	assert_type(and_expr, And[bool, Ctx])
	assert_type(and_expr.unwrap(ctx), bool)

	or_expr = (x < 0) | (x > 0)
	assert_type(or_expr, Or[bool, Ctx])
	assert_type(or_expr.unwrap(ctx), bool)

	not_expr = ~(x > 10)
	assert_type(not_expr, Not[Ctx])
	assert_type(not_expr.unwrap(ctx), bool)


@pytest.mark.mypy_testing
def test_func_result_types() -> None:
	"""Verify Func type captures args and result type."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	expr = x + y
	func = expr.to_func()
	assert_type(func, Func[int, Ctx])

	single_var_func = (x * 2).to_func()
	assert_type(single_var_func, Func[int, Ctx])

	const_func = Const("answer", 42).to_func()
	assert_type(const_func, Func[int, Any])


@pytest.mark.mypy_testing
def test_map_expr_result_types() -> None:
	"""Verify MapExpr type preserves element and result types."""
	n = Var[int, ElemCtx]("n")
	nums = Var[SizedIterable[int], ContainerCtx]("nums")

	mapped = (n * 2).map(nums)
	assert_type(mapped, MapExpr[Any, int, Any])

	container_ctx = ContainerCtx(nums=[1, 2, 3])
	result = mapped.unwrap(container_ctx)
	assert_type(result, SizedIterable[int])


@pytest.mark.mypy_testing
def test_contains_result_types() -> None:
	"""Verify Contains returns bool."""
	values = Var[SizedIterable[int], ContainsCtx]("values")
	target = Var[int, ContainsCtx]("target")

	contains_expr = Contains(target, values)
	assert_type(contains_expr, Contains[int, ContainsCtx])

	contains_ctx = ContainsCtx(values=[1, 2, 3], target=2)
	assert_type(contains_expr.eval(contains_ctx), Const[bool])
	assert_type(contains_expr.unwrap(contains_ctx), bool)


@pytest.mark.mypy_testing
def test_any_all_expr_result_types() -> None:
	"""Verify AnyExpr and AllExpr return bool."""
	flags = Var[SizedIterable[bool], FlagsCtx]("flags")

	any_expr = AnyExpr(flags)
	assert_type(any_expr, AnyExpr[FlagsCtx])

	all_expr = AllExpr(flags)
	assert_type(all_expr, AllExpr[FlagsCtx])

	flags_ctx = FlagsCtx(flags=[True, False, True])
	assert_type(any_expr.eval(flags_ctx), Const[bool])
	assert_type(any_expr.unwrap(flags_ctx), bool)
	assert_type(all_expr.eval(flags_ctx), Const[bool])
	assert_type(all_expr.unwrap(flags_ctx), bool)


@pytest.mark.mypy_testing
def test_min_max_expr_result_types() -> None:
	"""Verify MinExpr and MaxExpr preserve element type."""
	values = Var[SizedIterable[int], ValuesCtx]("values")

	min_expr = MinExpr(values)
	assert_type(min_expr, MinExpr[int, ValuesCtx])

	max_expr = MaxExpr(values)
	assert_type(max_expr, MaxExpr[int, ValuesCtx])

	values_ctx = ValuesCtx(values=[3, 1, 4, 1, 5])
	assert_type(min_expr.eval(values_ctx), Const[int])
	assert_type(min_expr.unwrap(values_ctx), int)
	assert_type(max_expr.eval(values_ctx), Const[int])
	assert_type(max_expr.unwrap(values_ctx), int)


def test_foldl_expr_result_types() -> None:
	"""Verify FoldLExpr evaluation works correctly."""
	values = Var[SizedIterable[int], ValuesCtx]("values")

	sum_fold = FoldLExpr(Add, values)
	product_fold = FoldLExpr(Mul, values, initial=1)

	values_ctx = ValuesCtx(values=[1, 2, 3, 4])
	assert sum_fold.unwrap(values_ctx) == 10
	assert product_fold.unwrap(values_ctx) == 24


@pytest.mark.mypy_testing
def test_predicate_result_types() -> None:
	"""Verify Predicate returns bool."""
	x = Var[int, PredicateCtx]("x")

	pred = Predicate("x is positive", x > 0)
	assert_type(pred, Predicate[PredicateCtx])

	pred_ctx = PredicateCtx(x=5)
	assert_type(pred.eval(pred_ctx), Const[bool])
	assert_type(pred.unwrap(pred_ctx), bool)

	bound_pred = pred.bind(pred_ctx)
	assert_type(bound_pred, BoundExpr[bool, PredicateCtx, bool])


class PartialXCtx(NamedTuple):
	x: int


class PartialYCtx(NamedTuple):
	y: int


class PartialFullCtx(NamedTuple):
	x: int
	y: int


@pytest.mark.mypy_testing
def test_partial_preserves_result_type_int() -> None:
	"""Verify partial application preserves int result type."""
	x = Var[int, Any]("x")
	y = Var[int, Any]("y")

	expr = x + y
	assert_type(expr, Add[int, Any])

	partial_expr = expr.partial(PartialXCtx(x=5))
	assert_type(partial_expr, Expr[int, Any, int])

	result = partial_expr.unwrap(PartialYCtx(y=10))
	assert_type(result, int)
	assert result == 15


@pytest.mark.mypy_testing
def test_partial_preserves_result_type_bool() -> None:
	"""Verify partial application preserves bool result type for comparisons."""
	x = Var[int, Any]("x")
	threshold = Const("threshold", 10)

	expr = x > threshold
	assert_type(expr, Gt[int, Any])

	partial_expr = expr.partial(PartialXCtx(x=15))
	assert_type(partial_expr, Expr[int, Any, bool])

	class EmptyCtx(NamedTuple):
		pass

	result = partial_expr.unwrap(EmptyCtx())
	assert_type(result, bool)
	assert result is True


@pytest.mark.mypy_testing
def test_bind_preserves_result_type_int() -> None:
	"""Verify bind preserves int result type."""
	x = Var[int, PartialFullCtx]("x")
	y = Var[int, PartialFullCtx]("y")

	expr = x * y
	assert_type(expr, Mul[int, PartialFullCtx])

	bound = expr.bind(PartialFullCtx(x=3, y=4))
	assert_type(bound, BoundExpr[int, PartialFullCtx, int])

	result = bound.unwrap()
	assert_type(result, int)
	assert result == 12


@pytest.mark.mypy_testing
def test_bind_preserves_result_type_bool() -> None:
	"""Verify bind preserves bool result type for comparisons."""
	x = Var[int, PartialFullCtx]("x")
	y = Var[int, PartialFullCtx]("y")

	expr = x > y
	assert_type(expr, Gt[int, PartialFullCtx])

	bound = expr.bind(PartialFullCtx(x=5, y=3))
	assert_type(bound, BoundExpr[int, PartialFullCtx, bool])

	result = bound.unwrap()
	assert_type(result, bool)
	assert result is True


@pytest.mark.mypy_testing
def test_foldl_partial_preserves_type() -> None:
	"""Verify FoldLExpr.partial preserves element type."""

	class FoldPartialCtx(NamedTuple):
		values: list[int]

	values = Var[SizedIterable[int], FoldPartialCtx]("values")

	fold_expr = FoldLExpr(Add, values)
	assert_type(fold_expr, FoldLExpr[int, FoldPartialCtx, int])

	partial_fold = fold_expr.partial(FoldPartialCtx(values=[1, 2, 3]))
	assert_type(partial_fold, Expr[int, Any, int])

	class EmptyCtx(NamedTuple):
		pass

	result = partial_fold.unwrap(EmptyCtx())
	assert_type(result, int)
	assert result == 6


def test_match_expr_basic() -> None:
	"""Test basic MatchExpr operation."""

	class MatchCtx(NamedTuple):
		x: int

	x = Var[int, MatchCtx]("x")
	match_expr = Match((x > 5, Const("high", 100)), default=Const("low", 0))

	assert match_expr.to_string() == "(match (x > 5 -> high:100) else low:0)"

	ctx_high = MatchCtx(x=10)
	assert match_expr.unwrap(ctx_high) == 100
	assert (
		match_expr.to_string(ctx_high) == "(match (x:10 > 5 -> True -> high:100) else low:0 -> 100)"
	)

	ctx_low = MatchCtx(x=3)
	assert match_expr.unwrap(ctx_low) == 0
	assert match_expr.to_string(ctx_low) == "(match (x:3 > 5 -> False -> high:100) else low:0 -> 0)"


def test_match_expr_with_variables() -> None:
	"""Test MatchExpr with variables in branches."""

	class BranchCtx(NamedTuple):
		x: int
		y: int
		z: int

	x = Var[int, BranchCtx]("x")
	y = Var[int, BranchCtx]("y")
	z = Var[int, BranchCtx]("z")

	match_expr = Match((x > 0, y), default=z)

	ctx_positive = BranchCtx(x=5, y=10, z=20)
	assert match_expr.unwrap(ctx_positive) == 10

	ctx_negative = BranchCtx(x=-5, y=10, z=20)
	assert match_expr.unwrap(ctx_negative) == 20


def test_match_expr_with_expressions_in_branches() -> None:
	"""Test MatchExpr with expressions in branches."""

	class ExprCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, ExprCtx]("x")
	y = Var[int, ExprCtx]("y")

	match_expr = Match((x > y, x + y), default=x * y)

	ctx = ExprCtx(x=10, y=5)
	assert match_expr.unwrap(ctx) == 15

	ctx2 = ExprCtx(x=3, y=5)
	assert match_expr.unwrap(ctx2) == 15


def test_match_expr_multi_branch() -> None:
	"""Test MatchExpr with multiple branches."""

	class MultiCtx(NamedTuple):
		x: int

	x = Var[int, MultiCtx]("x")

	match_expr = Match(
		(x > 10, Const("large", "large")),
		(x > 5, Const("medium", "medium")),
		default=Const("small", "small"),
	)

	assert match_expr.unwrap(MultiCtx(x=15)) == "large"
	assert match_expr.unwrap(MultiCtx(x=7)) == "medium"
	assert match_expr.unwrap(MultiCtx(x=3)) == "small"


def test_match_expr_with_boolean_result() -> None:
	"""Test MatchExpr that returns boolean values."""

	class BoolCtx(NamedTuple):
		x: int

	x = Var[int, BoolCtx]("x")

	match_expr = Match((x > 0, Const("positive", True)), default=Const("non_positive", False))

	assert match_expr.unwrap(BoolCtx(x=5)) is True
	assert match_expr.unwrap(BoolCtx(x=-5)) is False


def test_match_expr_partial() -> None:
	"""Test MatchExpr partial application."""

	class PartialMatchCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, PartialMatchCtx]("x")
	y = Var[int, PartialMatchCtx]("y")

	match_expr = Match((x > 0, y * 2), default=y * 3)

	class XOnlyCtx(NamedTuple):
		x: int

	partial_expr = match_expr.partial(XOnlyCtx(x=5))
	assert partial_expr.to_string() == "(match (x:5 > 0 -> (y * 2)) else (y * 3))"

	class YOnlyCtx(NamedTuple):
		y: int

	assert partial_expr.unwrap(YOnlyCtx(y=10)) == 20


def test_match_expr_composition() -> None:
	"""Test MatchExpr in expression composition."""

	class ComposeCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, ComposeCtx]("x")
	y = Var[int, ComposeCtx]("y")

	match_expr = Match((x > 0, x), default=Const("zero", 0))
	composed = match_expr + y

	ctx = ComposeCtx(x=5, y=10)
	assert composed.unwrap(ctx) == 15

	ctx2 = ComposeCtx(x=-5, y=10)
	assert composed.unwrap(ctx2) == 10


def test_match_expr_with_complex_condition() -> None:
	"""Test MatchExpr with complex condition."""

	class ComplexCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, ComplexCtx]("x")
	y = Var[int, ComplexCtx]("y")

	match_expr = Match(((x > 0) & (y > 0), x + y), default=Const("invalid", -1))

	ctx = ComplexCtx(x=5, y=10)
	assert match_expr.unwrap(ctx) == 15

	ctx2 = ComplexCtx(x=-5, y=10)
	assert match_expr.unwrap(ctx2) == -1

	ctx3 = ComplexCtx(x=5, y=-10)
	assert match_expr.unwrap(ctx3) == -1


def test_match_expr_extract_vars() -> None:
	"""Test that _extract_vars handles MatchExpr correctly."""

	class VarsCtx(NamedTuple):
		x: int
		y: int
		z: int

	x = Var[int, VarsCtx]("x")
	y = Var[int, VarsCtx]("y")
	z = Var[int, VarsCtx]("z")

	match_expr = Match((x > 0, y), default=z)
	func = match_expr.to_func()

	assert len(func.args) == 3


@pytest.mark.mypy_testing
def test_match_expr_type_preservation() -> None:
	"""Verify match() preserves result type."""

	class TypeCtx(NamedTuple):
		x: int

	x = Var[int, TypeCtx]("x")
	match_expr = Match((x > 5, Const("high", 100)), default=Const("low", 0))

	ctx = TypeCtx(x=10)
	result = match_expr.eval(ctx)
	assert_type(result, Const[int])
	assert_type(match_expr.unwrap(ctx), int)


def test_context_vars_basic() -> None:
	"""Test context_vars creates context class and matching vars."""
	TestCtx, x, y = context_vars(("x", int), ("y", float))

	assert x.name == "x"
	assert_type(x, Var[int, tuple[int, float]])
	assert y.name == "y"
	assert_type(y, Var[float, tuple[int, float]])
	assert_type(TestCtx, Callable[[int, float], tuple[int, float]])

	expr = x + y
	assert expr.to_string() == "(x + y)"

	test_ctx = TestCtx(1, 2.5)
	assert_type(test_ctx, tuple[int, float])
	assert expr.unwrap(test_ctx) == 3.5
	assert_type(expr.unwrap(test_ctx), float)
	assert expr.to_string(test_ctx) == "(x:1 + y:2.5 -> 3.5)"


def test_context_vars_single_field() -> None:
	"""Test context_vars with a single field."""
	SingleCtx, value = context_vars(("value", int))

	assert value.name == "value"
	assert_type(value, Var[int, tuple[int]])
	assert_type(SingleCtx, Callable[[int], tuple[int]])
	expr = value * 2
	single_ctx = SingleCtx(21)
	assert_type(single_ctx, tuple[int])
	assert expr.unwrap(single_ctx) == 42


def test_context_vars_with_predicates() -> None:
	"""Test context_vars with predicate expressions."""
	Measurements, voltage, current = context_vars(("voltage", float), ("current", float))

	power = voltage * current
	is_safe = (voltage < 50) & (current < 10)

	measurement_ctx = Measurements(12.0, 2.0)
	assert power.unwrap(measurement_ctx) == 24.0
	assert is_safe.unwrap(measurement_ctx) is True

	ctx_unsafe = Measurements(100.0, 5.0)
	assert is_safe.unwrap(ctx_unsafe) is False


def test_context_vars_with_approximation() -> None:
	"""Test context_vars with approximate comparisons."""
	SensorCtx, reading = context_vars(("reading", float))

	target = PlusMinus("Target", 5.0, 0.1)
	check = Approximately(reading, target)

	ctx_pass = SensorCtx(5.05)
	assert check.unwrap(ctx_pass) is True

	ctx_fail = SensorCtx(5.2)
	assert check.unwrap(ctx_fail) is False


@pytest.mark.mypy_testing
def test_context_vars_type_inference() -> None:
	"""Test that context_vars preserves Var and context types for type checker."""
	Ctx, x, y = context_vars(("x", float), ("y", int))

	assert_type(Ctx, Callable[[float, int], tuple[float, int]])
	assert_type(x, Var[float, tuple[float, int]])
	assert_type(y, Var[int, tuple[float, int]])

	ctx = Ctx(1.5, 42)
	assert_type(ctx, tuple[float, int])

	expr = x + 1.0
	assert_type(expr, Add[float, tuple[float, int]])


def test_type_alias_float_var() -> None:
	"""Test FloatVar type alias."""

	class TempCtx(NamedTuple):
		temperature: float

	temp: FloatVar[TempCtx] = Var("temperature")
	assert temp.name == "temperature"
	assert temp.unwrap(TempCtx(temperature=25.5)) == 25.5


def test_type_alias_int_var() -> None:
	"""Test IntVar type alias."""

	class CountCtx(NamedTuple):
		count_val: int

	count_var: IntVar[CountCtx] = Var("count_val")
	assert count_var.name == "count_val"
	assert count_var.unwrap(CountCtx(count_val=42)) == 42


def test_type_alias_bool_var() -> None:
	"""Test BoolVar type alias."""

	class FlagCtx(NamedTuple):
		flag: bool

	flag_var: BoolVar[FlagCtx] = Var("flag")
	assert flag_var.name == "flag"
	assert flag_var.unwrap(FlagCtx(flag=True)) is True


def test_type_alias_str_var() -> None:
	"""Test StrVar type alias."""

	class NameCtx(NamedTuple):
		name: str

	name_var: StrVar[NameCtx] = Var("name")
	assert name_var.name == "name"
	assert name_var.unwrap(NameCtx(name="test")) == "test"


def test_type_alias_list_var() -> None:
	"""Test ListVar type alias."""

	class ListValuesCtx(NamedTuple):
		values: list[int]

	values_var: ListVar[int, ListValuesCtx] = Var("values")
	assert values_var.name == "values"
	assert values_var.unwrap(ListValuesCtx(values=[1, 2, 3])) == [1, 2, 3]


def test_type_aliases_in_expressions() -> None:
	"""Test type aliases work in expression composition."""

	class CompositeCtx(NamedTuple):
		x: float
		y: float
		items: list[int]

	x: FloatVar[CompositeCtx] = Var("x")
	y: FloatVar[CompositeCtx] = Var("y")
	items: ListVar[int, CompositeCtx] = Var("items")

	sum_expr = x + y
	composite_ctx = CompositeCtx(x=1.5, y=2.5, items=[1, 2, 3])

	assert sum_expr.unwrap(composite_ctx) == 4.0

	contains_expr = Contains(Const(None, 2), items)
	assert contains_expr.unwrap(composite_ctx) is True


def test_context_vars_manufacturing_example() -> None:
	"""Test context_vars with a realistic manufacturing test scenario."""
	Measurement, voltage, current, temp = context_vars(
		("voltage", float),
		("current", float),
		("temp", float),
	)

	power = voltage * current
	voltage_ok = Approximately(voltage, PlusMinus("V_nom", 12.0, 0.5))
	current_ok = (current > 0.1) & (current < 2.0)
	temp_ok = (temp > 0) & (temp < 85)
	all_pass = voltage_ok & current_ok & temp_ok

	good = Measurement(12.1, 0.5, 25.0)
	assert power.unwrap(good) == pytest.approx(6.05)  # pyright: ignore[reportUnknownMemberType]
	assert voltage_ok.unwrap(good) is True
	assert current_ok.unwrap(good) is True
	assert temp_ok.unwrap(good) is True
	assert all_pass.unwrap(good) is True

	bad_voltage = Measurement(15.0, 0.5, 25.0)
	assert all_pass.unwrap(bad_voltage) is False
	assert voltage_ok.unwrap(bad_voltage) is False


def test_context_vars_with_max_min() -> None:
	"""Test context_vars with MaxExpr and MinExpr operations."""

	class DataCtx(NamedTuple):
		samples: list[float]

	samples = Var[SizedIterable[float], DataCtx]("samples")
	sample_max = MaxExpr(samples)
	sample_min = MinExpr(samples)

	ctx = DataCtx(samples=[1.0, 2.0, 3.0, 4.0, 5.0])

	assert sample_max.unwrap(ctx) == 5.0
	assert sample_min.unwrap(ctx) == 1.0


def test_context_vars_with_conditionals() -> None:
	"""Test context_vars with MatchExpr conditional logic."""
	StatusCtx, value, threshold = context_vars(("value", int), ("threshold", int))

	status = Match((value > threshold, Const("status", "HIGH")), default=Const("status", "LOW"))

	high_ctx = StatusCtx(100, 50)
	assert status.unwrap(high_ctx) == "HIGH"

	low_ctx = StatusCtx(30, 50)
	assert status.unwrap(low_ctx) == "LOW"


def test_context_vars_with_any_all() -> None:
	"""Test context_vars with AnyExpr and AllExpr on boolean containers."""

	class ChecksCtx(NamedTuple):
		flags: list[bool]

	flags = Var[SizedIterable[bool], ChecksCtx]("flags")
	any_true = AnyExpr(flags)
	all_true = AllExpr(flags)

	all_pass_ctx = ChecksCtx(flags=[True, True, True])
	assert any_true.unwrap(all_pass_ctx) is True
	assert all_true.unwrap(all_pass_ctx) is True

	mixed_ctx = ChecksCtx(flags=[True, False, True])
	assert any_true.unwrap(mixed_ctx) is True
	assert all_true.unwrap(mixed_ctx) is False

	all_fail_ctx = ChecksCtx(flags=[False, False, False])
	assert any_true.unwrap(all_fail_ctx) is False
	assert all_true.unwrap(all_fail_ctx) is False


def test_context_vars_complex_expression_composition() -> None:
	"""Test composing multiple expressions from context_vars."""
	SensorCtx, v1, v2, v3, v4 = context_vars(
		("v1", float),
		("v2", float),
		("v3", float),
		("v4", float),
	)

	avg = (v1 + v2 + v3 + v4) / 4
	spread = Max(v1, Max(v2, Max(v3, v4))) - Min(v1, Min(v2, Min(v3, v4)))
	balanced = spread < 1.0

	ctx = SensorCtx(10.0, 10.2, 10.1, 10.3)
	assert avg.unwrap(ctx) == pytest.approx(10.15)  # pyright: ignore[reportUnknownMemberType]
	assert spread.unwrap(ctx) == pytest.approx(0.3)  # pyright: ignore[reportUnknownMemberType]
	assert balanced.unwrap(ctx) is True

	unbalanced_ctx = SensorCtx(5.0, 10.0, 15.0, 20.0)
	assert spread.unwrap(unbalanced_ctx) == 15.0
	assert balanced.unwrap(unbalanced_ctx) is False


def test_context_vars_to_string_output() -> None:
	"""Test that context_vars expressions produce good string representations."""
	TestCtx, x, y = context_vars(("x", int), ("y", int))

	expr = (x + y) * 2
	assert expr.to_string() == "((x + y) * 2)"

	ctx = TestCtx(3, 4)
	assert expr.to_string(ctx) == "((x:3 + y:4 -> 7) * 2 -> 14)"


def test_context_vars_bound_expr() -> None:
	"""Test binding context_vars expressions for repeated evaluation."""
	ConfigCtx, base, multiplier = context_vars(("base", float), ("multiplier", float))

	formula = base * multiplier

	bound = formula.bind(ConfigCtx(100.0, 1.5))
	assert bound.unwrap() == 150.0
	assert str(bound) == "(base:100.0 * multiplier:1.5 -> 150.0)"


def test_abs() -> None:
	"""Test Abs unary operation."""
	x = Var[int, Ctx]("x")
	abs_x = Abs(x)

	assert abs_x.to_string() == "(abs x)"
	assert abs_x.unwrap(ctx) == 5
	assert abs_x.to_string(ctx) == "(abs x:5 -> 5)"

	neg_ctx = Ctx(x=-5, y=10, name="test")
	assert abs_x.unwrap(neg_ctx) == 5
	assert abs_x.to_string(neg_ctx) == "(abs x:-5 -> 5)"

	zero_ctx = Ctx(x=0, y=10, name="test")
	assert abs_x.unwrap(zero_ctx) == 0


def test_abs_with_float() -> None:
	"""Test Abs with float values."""

	class FloatCtx(NamedTuple):
		x: float

	x = Var[float, FloatCtx]("x")
	abs_x = Abs(x)

	assert abs_x.unwrap(FloatCtx(x=-3.14)) == 3.14
	assert abs_x.unwrap(FloatCtx(x=2.71)) == 2.71
	assert abs_x.to_string(FloatCtx(x=-3.14)) == "(abs x:-3.14 -> 3.14)"


@pytest.mark.mypy_testing
def test_abs_types() -> None:
	"""Verify Abs type inference."""
	x = Var[int, Ctx]("x")
	abs_x = Abs(x)

	assert_type(abs_x, Abs[int, Ctx])
	assert_type(abs_x.eval(ctx), Const[int])
	assert_type(abs_x.unwrap(ctx), int)

	f = Var[float, Ctx]("f")
	abs_f = Abs(f)
	assert_type(abs_f, Abs[float, Ctx])
	assert_type(abs_f.unwrap(ctx), float)


def test_abs_composition() -> None:
	"""Test Abs composition with other expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	taxicab = Abs(x) + Abs(y)
	assert taxicab.to_string() == "((abs x) + (abs y))"
	neg_ctx = Ctx(x=-3, y=-4, name="test")
	assert taxicab.unwrap(neg_ctx) == 7
	assert taxicab.to_string(neg_ctx) == "((abs x:-3 -> 3) + (abs y:-4 -> 4) -> 7)"

	doubled = Abs(x) * 2
	assert doubled.unwrap(neg_ctx) == 6
	assert doubled.to_string() == "((abs x) * 2)"

	compared = Abs(x) > 2
	assert compared.unwrap(neg_ctx) is True
	assert compared.unwrap(Ctx(x=-1, y=0, name="test")) is False


def test_abs_nested() -> None:
	"""Test nested Abs and Abs of expressions."""
	x = Var[int, Ctx]("x")

	double_abs = Abs(Abs(x))
	neg_ctx = Ctx(x=-5, y=10, name="test")
	assert double_abs.unwrap(neg_ctx) == 5

	abs_of_expr = Abs(x - 10)
	assert abs_of_expr.unwrap(ctx) == 5
	assert abs_of_expr.to_string() == "(abs (x - 10))"
	assert abs_of_expr.to_string(ctx) == "(abs (x:5 - 10 -> -5) -> 5)"


def test_clamp() -> None:
	"""Test Clamp operation."""

	class ClampCtx(NamedTuple):
		x: int

	x = Var[int, ClampCtx]("x")
	clamped = Clamp(0, 10)(x)

	assert clamped.to_string() == "(clamp 0 10 x)"
	assert clamped.unwrap(ClampCtx(x=5)) == 5
	assert clamped.unwrap(ClampCtx(x=-5)) == 0
	assert clamped.unwrap(ClampCtx(x=15)) == 10
	assert clamped.to_string(ClampCtx(x=15)) == "(clamp 0 10 x:15 -> 10)"
	assert clamped.to_string(ClampCtx(x=-5)) == "(clamp 0 10 x:-5 -> 0)"


def test_clamp_with_float() -> None:
	"""Test Clamp with float values."""

	class FloatCtx(NamedTuple):
		x: float

	x = Var[float, FloatCtx]("x")
	clamped = Clamp(0.0, 100.0)(x)

	assert clamped.unwrap(FloatCtx(x=50.5)) == 50.5
	assert clamped.unwrap(FloatCtx(x=-10.5)) == 0.0
	assert clamped.unwrap(FloatCtx(x=150.5)) == 100.0


def test_clamp_edge_cases() -> None:
	"""Test Clamp at boundary values."""

	class ClampCtx(NamedTuple):
		x: int

	x = Var[int, ClampCtx]("x")
	clamped = Clamp(0, 10)(x)

	assert clamped.unwrap(ClampCtx(x=0)) == 0
	assert clamped.unwrap(ClampCtx(x=10)) == 10


@pytest.mark.mypy_testing
def test_clamp_types() -> None:
	"""Verify ClampExpr type inference."""
	from mahonia import ClampExpr

	class ClampCtx(NamedTuple):
		x: int

	x = Var[int, ClampCtx]("x")
	clamped = Clamp(0, 10)(x)

	assert_type(clamped, ClampExpr[int, ClampCtx])
	assert_type(clamped.eval(ClampCtx(x=5)), Const[int])
	assert_type(clamped.unwrap(ClampCtx(x=5)), int)


def test_clamp_composition() -> None:
	"""Test Clamp composition with other expressions."""

	class ClampCtx(NamedTuple):
		x: int

	x = Var[int, ClampCtx]("x")

	clamped_doubled = Clamp(0, 10)(x) * 2
	assert clamped_doubled.to_string() == "((clamp 0 10 x) * 2)"
	assert clamped_doubled.unwrap(ClampCtx(x=15)) == 20
	assert clamped_doubled.unwrap(ClampCtx(x=3)) == 6

	clamped_compared = Clamp(0, 10)(x) > 5
	assert clamped_compared.unwrap(ClampCtx(x=15)) is True
	assert clamped_compared.unwrap(ClampCtx(x=3)) is False


def test_clamp_of_expression() -> None:
	"""Test Clamp applied to complex expressions."""

	class ClampCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, ClampCtx]("x")
	y = Var[int, ClampCtx]("y")

	clamped_sum = Clamp(0, 100)(x + y)
	assert clamped_sum.to_string() == "(clamp 0 100 (x + y))"
	assert clamped_sum.unwrap(ClampCtx(x=30, y=40)) == 70
	assert clamped_sum.unwrap(ClampCtx(x=60, y=60)) == 100
	assert clamped_sum.unwrap(ClampCtx(x=-50, y=20)) == 0


def test_abs_clamp_combined() -> None:
	"""Test combining Abs and Clamp."""

	class Ctx(NamedTuple):
		x: int

	x = Var[int, Ctx]("x")

	clamped_abs = Clamp(5, 10)(Abs(x))
	assert clamped_abs.to_string() == "(clamp 5 10 (abs x))"
	assert clamped_abs.unwrap(Ctx(x=-3)) == 5
	assert clamped_abs.unwrap(Ctx(x=-7)) == 7
	assert clamped_abs.unwrap(Ctx(x=-15)) == 10
	assert clamped_abs.unwrap(Ctx(x=8)) == 8

	abs_clamped = Abs(Clamp(-5, 5)(x))
	assert abs_clamped.unwrap(Ctx(x=-10)) == 5
	assert abs_clamped.unwrap(Ctx(x=-3)) == 3
	assert abs_clamped.unwrap(Ctx(x=3)) == 3


def test_abs_partial() -> None:
	"""Test Abs.partial() preserves structure."""

	class FullCtx(NamedTuple):
		x: int
		y: int

	class PartialCtx(NamedTuple):
		x: int

	x = Var[int, FullCtx]("x")
	y = Var[int, FullCtx]("y")

	expr = Abs(x) + y
	partial_ctx = PartialCtx(x=-5)
	partial_expr = expr.partial(partial_ctx)

	assert partial_expr.to_string() == "((abs x:-5) + y)"
	assert partial_expr.to_string(FullCtx(x=-5, y=3)) == "((abs x:-5 -> 5) + y:3 -> 8)"


def test_clamp_partial() -> None:
	"""Test Clamp.partial() preserves structure."""

	class FullCtx(NamedTuple):
		x: int
		y: int

	class PartialCtx(NamedTuple):
		x: int

	x = Var[int, FullCtx]("x")
	y = Var[int, FullCtx]("y")

	expr = Clamp(0, 10)(x) + y
	partial_ctx = PartialCtx(x=15)
	partial_expr = expr.partial(partial_ctx)

	assert partial_expr.to_string() == "((clamp 0 10 x:15) + y)"
	assert partial_expr.to_string(FullCtx(x=15, y=5)) == "((clamp 0 10 x:15 -> 10) + y:5 -> 15)"


def test_clamp_factory_reuse() -> None:
	"""Test that Clamp(lo, hi) factory can be reused."""

	class Ctx(NamedTuple):
		x: int
		y: int

	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	normalize = Clamp(0, 100)
	clamped_x = normalize(x)
	clamped_y = normalize(y)

	ctx = Ctx(x=-10, y=150)
	assert clamped_x.unwrap(ctx) == 0
	assert clamped_y.unwrap(ctx) == 100

	assert repr(normalize) == "Clamp(0, 100)"
	assert clamped_x.to_string() == "(clamp 0 100 x)"
	assert clamped_y.to_string() == "(clamp 0 100 y)"
