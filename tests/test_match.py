# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import NamedTuple, assert_type

import pytest

from mahonia import (
	Add,
	And,
	Const,
	Div,
	Eq,
	Ge,
	Gt,
	Le,
	Lt,
	Match,
	MatchExpr,
	Mod,
	Mul,
	Neg,
	Not,
	Or,
	Sub,
	Var,
)
from mahonia.latex import LatexCtx, Show, latex


class FizzBuzzCtx(NamedTuple):
	n: int


def test_fizzbuzz() -> None:
	"""Demonstrate MatchExpr with FizzBuzz logic."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	assert fizzbuzz.unwrap(FizzBuzzCtx(n=15)) == "FizzBuzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=9)) == "Fizz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=10)) == "Buzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=7)) == 7
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=1)) == 1
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=30)) == "FizzBuzz"


def test_fizzbuzz_to_string() -> None:
	"""Test FizzBuzz to_string output."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, Const("fb", "FizzBuzz")),
		((n % 3) == 0, Const("f", "Fizz")),
		((n % 5) == 0, Const("b", "Buzz")),
		default=n,
	)

	assert "(n % 15)" in fizzbuzz.to_string()
	assert "fb:FizzBuzz" in fizzbuzz.to_string()
	assert "f:Fizz" in fizzbuzz.to_string()
	assert "b:Buzz" in fizzbuzz.to_string()
	assert "else n" in fizzbuzz.to_string()

	ctx = FizzBuzzCtx(n=15)
	result_str = fizzbuzz.to_string(ctx)
	assert "-> FizzBuzz" in result_str


def test_fizzbuzz_sequence() -> None:
	"""Test FizzBuzz for sequence 1-20."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	expected = [
		1,
		2,
		"Fizz",
		4,
		"Buzz",
		"Fizz",
		7,
		8,
		"Fizz",
		"Buzz",
		11,
		"Fizz",
		13,
		14,
		"FizzBuzz",
		16,
		17,
		"Fizz",
		19,
		"Buzz",
	]

	for i, exp in enumerate(expected, 1):
		assert fizzbuzz.unwrap(FizzBuzzCtx(n=i)) == exp


@pytest.mark.mypy_testing
def test_match_expr_type_inference() -> None:
	"""Verify match() correctly infers union types."""

	class Ctx(NamedTuple):
		x: int

	x = Var[int, Ctx]("x")

	m1 = Match((x > 5, "high"), default=0)
	assert_type(m1, MatchExpr[str | int, Ctx])
	assert_type(m1.unwrap(Ctx(x=10)), str | int)
	assert_type(m1.eval(Ctx(x=10)), Const[str | int])

	m2 = Match((x > 10, "large"), (x > 5, "medium"), default=0)
	assert_type(m2, MatchExpr[str | int, Ctx])

	m3 = Match((x > 10, "large"), (x > 5, 3.14), (x > 0, True), default=0)
	assert_type(m3, MatchExpr[str | float | bool | int, Ctx])
	assert_type(m3.unwrap(Ctx(x=1)), str | float | bool | int)


@pytest.mark.mypy_testing
def test_match_expr_fizzbuzz_types() -> None:
	"""Verify FizzBuzz has correct str | int union type."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	assert_type(fizzbuzz, MatchExpr[str | int, FizzBuzzCtx])
	assert_type(fizzbuzz.unwrap(FizzBuzzCtx(n=1)), str | int)
	assert_type(fizzbuzz.eval(FizzBuzzCtx(n=1)), Const[str | int])


@pytest.mark.mypy_testing
def test_match_expr_homogeneous_type() -> None:
	"""Verify Match() with same type in all branches."""

	class Ctx(NamedTuple):
		x: int

	x = Var[int, Ctx]("x")

	m = Match((x > 10, 100), (x > 5, 50), default=0)
	assert_type(m, MatchExpr[int, Ctx])
	assert_type(m.unwrap(Ctx(x=1)), int)


@pytest.mark.mypy_testing
def test_match_overload_coverage() -> None:
	"""Test all Match overload categories for type inference.

	Coverage for:
	- Named vs inline variable patterns (mypy handles these differently)
	- All three overload categories: Const+Const, Const+Expr, Expr+Expr
	- 5-6 branch overloads
	- Var as both default and branch value
	"""

	class Ctx(NamedTuple):
		x: int
		y: str

	x = Var[int, Ctx]("x")
	y = Var[str, Ctx]("y")

	cond = x > 10
	val = Const("a", "high")
	default_const = Const("c", 0)

	named_match = Match((cond, val), default=default_const)
	assert_type(named_match, MatchExpr[str | int, Ctx])

	inline_match = Match((x > 10, "high"), default=0)
	assert_type(inline_match, MatchExpr[str | int, Ctx])

	const_expr_match = Match((x > 10, "high"), (x > 5, "med"), default=x)
	assert_type(const_expr_match, MatchExpr[str | int, Ctx])

	expr_expr_match = Match((x > 10, y), default=x)
	assert_type(expr_expr_match, MatchExpr[str | int, Ctx])

	five_branch = Match((x > 50, 5), (x > 40, 4), (x > 30, 3), (x > 20, 2), (x > 10, 1), default=0)
	assert_type(five_branch, MatchExpr[int, Ctx])

	six_branch = Match(
		(x > 60, 6), (x > 50, 5), (x > 40, 4), (x > 30, 3), (x > 20, 2), (x > 10, 1), default=0
	)
	assert_type(six_branch, MatchExpr[int, Ctx])

	var_as_branch_value = Match((x > 5, x), default=0)
	assert_type(var_as_branch_value, MatchExpr[int, Ctx])


def test_fizzbuzz_eval_returns_const() -> None:
	"""Verify eval() returns Const objects with correct values."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	result_15 = fizzbuzz.eval(FizzBuzzCtx(n=15))
	assert isinstance(result_15, Const)
	assert result_15.value == "FizzBuzz"

	result_9 = fizzbuzz.eval(FizzBuzzCtx(n=9))
	assert isinstance(result_9, Const)
	assert result_9.value == "Fizz"

	result_10 = fizzbuzz.eval(FizzBuzzCtx(n=10))
	assert isinstance(result_10, Const)
	assert result_10.value == "Buzz"

	result_7 = fizzbuzz.eval(FizzBuzzCtx(n=7))
	assert isinstance(result_7, Const)
	assert result_7.value == 7


def test_fizzbuzz_bind() -> None:
	"""Test binding FizzBuzz to a context creates BoundExpr."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	bound_15 = fizzbuzz.bind(FizzBuzzCtx(n=15))
	assert bound_15.unwrap() == "FizzBuzz"
	assert bound_15.ctx == FizzBuzzCtx(n=15)
	assert bound_15.expr is fizzbuzz

	bound_7 = fizzbuzz.bind(FizzBuzzCtx(n=7))
	assert bound_7.unwrap() == 7

	assert "(n:15 % 15 -> 0)" in str(bound_15)
	assert "-> FizzBuzz" in str(bound_15)


def test_fizzbuzz_to_string_full_serialization() -> None:
	"""Test complete to_string output for all branches."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, Const("fb", "FizzBuzz")),
		((n % 3) == 0, Const("f", "Fizz")),
		((n % 5) == 0, Const("b", "Buzz")),
		default=n,
	)

	unbound_str = fizzbuzz.to_string()
	assert unbound_str == (
		"(match ((n % 15) == 0 -> fb:FizzBuzz) "
		"((n % 3) == 0 -> f:Fizz) "
		"((n % 5) == 0 -> b:Buzz) else n)"
	)

	bound_str_15 = fizzbuzz.to_string(FizzBuzzCtx(n=15))
	assert "(n:15 % 15 -> 0)" in bound_str_15
	assert "-> FizzBuzz" in bound_str_15

	bound_str_9 = fizzbuzz.to_string(FizzBuzzCtx(n=9))
	assert "(n:9 % 3 -> 0)" in bound_str_9
	assert "-> Fizz" in bound_str_9

	bound_str_10 = fizzbuzz.to_string(FizzBuzzCtx(n=10))
	assert "(n:10 % 5 -> 0)" in bound_str_10
	assert "-> Buzz" in bound_str_10

	bound_str_7 = fizzbuzz.to_string(FizzBuzzCtx(n=7))
	assert "-> 7" in bound_str_7


def test_fizzbuzz_latex() -> None:
	"""Test LaTeX conversion of FizzBuzz expression."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, Const("fb", "FizzBuzz")),
		((n % 3) == 0, Const("f", "Fizz")),
		((n % 5) == 0, Const("b", "Buzz")),
		default=n,
	)

	result = latex(fizzbuzz)
	assert "\\begin{cases}" in result
	assert "\\end{cases}" in result
	assert "\\text{if }" in result
	assert "fb" in result
	assert "otherwise" in result


def test_fizzbuzz_latex_with_context() -> None:
	"""Test LaTeX conversion with evaluated context."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	result = latex(fizzbuzz, LatexCtx(FizzBuzzCtx(n=15), Show.VALUES | Show.WORK))
	assert "\\rightarrow" in result


def test_fizzbuzz_edge_cases() -> None:
	"""Test FizzBuzz with edge case values."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"),
		((n % 3) == 0, "Fizz"),
		((n % 5) == 0, "Buzz"),
		default=n,
	)

	assert fizzbuzz.unwrap(FizzBuzzCtx(n=0)) == "FizzBuzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=-15)) == "FizzBuzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=-3)) == "Fizz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=-5)) == "Buzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=-7)) == -7
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=45)) == "FizzBuzz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=99)) == "Fizz"
	assert fizzbuzz.unwrap(FizzBuzzCtx(n=100)) == "Buzz"


def test_fizzbuzz_composition_with_arithmetic() -> None:
	"""Test FizzBuzz composed with arithmetic expressions."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(((n % 15) == 0, 15), ((n % 3) == 0, 3), ((n % 5) == 0, 5), default=n)

	doubled = fizzbuzz * 2
	assert doubled.unwrap(FizzBuzzCtx(n=15)) == 30
	assert doubled.unwrap(FizzBuzzCtx(n=9)) == 6
	assert doubled.unwrap(FizzBuzzCtx(n=10)) == 10
	assert doubled.unwrap(FizzBuzzCtx(n=7)) == 14

	plus_one = fizzbuzz + 1
	assert plus_one.unwrap(FizzBuzzCtx(n=15)) == 16
	assert plus_one.unwrap(FizzBuzzCtx(n=7)) == 8


def test_fizzbuzz_composition_with_comparison() -> None:
	"""Test FizzBuzz result in comparison expressions."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz_numeric = Match(((n % 15) == 0, 15), ((n % 3) == 0, 3), ((n % 5) == 0, 5), default=n)

	is_large = fizzbuzz_numeric > 10
	assert is_large.unwrap(FizzBuzzCtx(n=15)) is True
	assert is_large.unwrap(FizzBuzzCtx(n=9)) is False
	assert is_large.unwrap(FizzBuzzCtx(n=22)) is True


@pytest.mark.mypy_testing
def test_fizzbuzz_composition_with_logic() -> None:
	"""Test FizzBuzz conditions in logical expressions."""
	n = Var[int, FizzBuzzCtx]("n")

	mod_3 = n % 3
	assert_type(mod_3, Mod[int, FizzBuzzCtx])

	is_fizz = (n % 3) == 0
	is_buzz = (n % 5) == 0
	assert_type(is_fizz, Eq[int, FizzBuzzCtx])
	assert_type(is_buzz, Eq[int, FizzBuzzCtx])

	is_fizzbuzz = is_fizz & is_buzz
	assert_type(is_fizzbuzz, And[bool, FizzBuzzCtx])
	assert_type(is_fizzbuzz.unwrap(FizzBuzzCtx(n=15)), bool)

	assert is_fizzbuzz.unwrap(FizzBuzzCtx(n=15)) is True
	assert is_fizzbuzz.unwrap(FizzBuzzCtx(n=9)) is False
	assert is_fizzbuzz.unwrap(FizzBuzzCtx(n=10)) is False
	assert is_fizzbuzz.unwrap(FizzBuzzCtx(n=7)) is False

	is_special = is_fizz | is_buzz
	assert_type(is_special, Or[bool, FizzBuzzCtx])
	assert_type(is_special.unwrap(FizzBuzzCtx(n=15)), bool)
	assert is_special.unwrap(FizzBuzzCtx(n=15)) is True
	assert is_special.unwrap(FizzBuzzCtx(n=9)) is True
	assert is_special.unwrap(FizzBuzzCtx(n=10)) is True
	assert is_special.unwrap(FizzBuzzCtx(n=7)) is False


def test_fizzbuzz_bound_expr_composition() -> None:
	"""Test composing BoundExpr from FizzBuzz with other expressions."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz_numeric = Match(
		((n % 15) == 0, 15),
		((n % 3) == 0, 3),
		((n % 5) == 0, 5),
		default=n,
	)

	bound = fizzbuzz_numeric.bind(FizzBuzzCtx(n=15))
	composed = bound + 100
	assert composed.unwrap(()) == 115

	composed_mul = bound * 10
	assert composed_mul.unwrap(()) == 150


@pytest.mark.mypy_testing
def test_fizzbuzz_bind_type() -> None:
	"""Verify bind() returns correctly typed BoundExpr."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"), ((n % 3) == 0, "Fizz"), ((n % 5) == 0, "Buzz"), default=n
	)

	bound = fizzbuzz.bind(FizzBuzzCtx(n=15))
	assert_type(bound.unwrap(), str | int)
	assert_type(bound.eval(()), Const[str | int])


def test_fizzbuzz_repr() -> None:
	"""Test repr of FizzBuzz MatchExpr."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz = Match(
		((n % 15) == 0, "FizzBuzz"), ((n % 3) == 0, "Fizz"), ((n % 5) == 0, "Buzz"), default=n
	)

	repr_str = repr(fizzbuzz)
	assert "MatchExpr" in repr_str or "branches" in repr_str


def test_fizzbuzz_branches_attribute() -> None:
	"""Test direct access to branches and default attributes."""
	n = Var[int, FizzBuzzCtx]("n")

	fb_cond = (n % 15) == 0
	f_cond = (n % 3) == 0
	b_cond = (n % 5) == 0
	fb_val = Const("fb", "FizzBuzz")
	f_val = Const("f", "Fizz")
	b_val = Const("b", "Buzz")

	fizzbuzz = Match(
		(fb_cond, fb_val),
		(f_cond, f_val),
		(b_cond, b_val),
		default=n,
	)

	assert len(fizzbuzz.branches) == 3
	assert fizzbuzz.branches[0] == (fb_cond, fb_val)
	assert fizzbuzz.branches[1] == (f_cond, f_val)
	assert fizzbuzz.branches[2] == (b_cond, b_val)
	assert fizzbuzz.default is n


class GradeCtx(NamedTuple):
	score: int


@pytest.mark.mypy_testing
def test_composed_grading_system() -> None:
	"""Demonstrate composing Match expressions for a grading system."""
	score = Var[int, GradeCtx]("score")

	letter_grade = Match(
		(score >= 90, Const("A", "A")),
		(score >= 80, Const("B", "B")),
		(score >= 70, Const("C", "C")),
		(score >= 60, Const("D", "D")),
		default=Const("F", "F"),
	)
	assert_type(letter_grade, MatchExpr[str, GradeCtx])
	gpa_points = Match(
		(score >= 90, Const("A_pts", 4.0)),
		(score >= 80, Const("B_pts", 3.0)),
		(score >= 70, Const("C_pts", 2.0)),
		(score >= 60, Const("D_pts", 1.0)),
		default=Const("F_pts", 0.0),
	)
	assert_type(gpa_points, MatchExpr[float, GradeCtx])
	assert letter_grade.unwrap(GradeCtx(score=95)) == "A"
	assert letter_grade.unwrap(GradeCtx(score=85)) == "B"
	assert letter_grade.unwrap(GradeCtx(score=55)) == "F"

	assert gpa_points.unwrap(GradeCtx(score=95)) == 4.0
	assert gpa_points.unwrap(GradeCtx(score=75)) == 2.0

	weighted_gpa = gpa_points * Const("credit_hours", 3.0)
	assert weighted_gpa.unwrap(GradeCtx(score=95)) == 12.0
	assert weighted_gpa.unwrap(GradeCtx(score=85)) == 9.0

	assert letter_grade.to_string() == (
		"(match (score >= 90 -> A:A) (score >= 80 -> B:B) "
		"(score >= 70 -> C:C) (score >= 60 -> D:D) else F:F)"
	)
	assert letter_grade.to_string(GradeCtx(score=95)) == (
		"(match (score:95 >= 90 -> True -> A:A) (score:95 >= 80 -> True -> B:B) "
		"(score:95 >= 70 -> True -> C:C) (score:95 >= 60 -> True -> D:D) else F:F -> A)"
	)
	assert letter_grade.to_string(GradeCtx(score=55)) == (
		"(match (score:55 >= 90 -> False -> A:A) (score:55 >= 80 -> False -> B:B) "
		"(score:55 >= 70 -> False -> C:C) (score:55 >= 60 -> False -> D:D) else F:F -> F)"
	)


class TaxCtx(NamedTuple):
	income: float
	filing_status: str


@pytest.mark.mypy_testing
def test_composed_tax_bracket_calculator() -> None:
	"""Demonstrate nested Match for tax bracket calculation."""
	income = Var[float, TaxCtx]("income")

	tax_rate = Match(
		(income > 500000, Const("rate_37", 0.37)),
		(income > 200000, Const("rate_32", 0.32)),
		(income > 80000, Const("rate_22", 0.22)),
		(income > 40000, Const("rate_12", 0.12)),
		default=Const("rate_10", 0.10),
	)
	assert_type(tax_rate, MatchExpr[float, TaxCtx])
	tax_owed = income * tax_rate
	assert_type(tax_owed, Mul[float, TaxCtx])

	assert tax_rate.unwrap(TaxCtx(income=600000, filing_status="single")) == 0.37
	assert tax_rate.unwrap(TaxCtx(income=100000, filing_status="single")) == 0.22

	assert tax_owed.unwrap(TaxCtx(income=100000, filing_status="single")) == 22000.0
	assert tax_owed.unwrap(TaxCtx(income=50000, filing_status="single")) == 6000.0

	assert tax_owed.to_string() == (
		"(income * (match (income > 500000 -> rate_37:0.37) "
		"(income > 200000 -> rate_32:0.32) (income > 80000 -> rate_22:0.22) "
		"(income > 40000 -> rate_12:0.12) else rate_10:0.1))"
	)
	assert tax_owed.to_string(TaxCtx(income=100000, filing_status="single")) == (
		"(income:100000 * (match (income:100000 > 500000 -> False -> rate_37:0.37) "
		"(income:100000 > 200000 -> False -> rate_32:0.32) "
		"(income:100000 > 80000 -> True -> rate_22:0.22) "
		"(income:100000 > 40000 -> True -> rate_12:0.12) else rate_10:0.1 -> 0.22) -> 22000.0)"
	)


class ShippingCtx(NamedTuple):
	weight: float
	distance: int
	is_express: bool


def test_composed_shipping_calculator() -> None:
	"""Demonstrate composing Match with arithmetic for shipping cost."""
	weight = Var[float, ShippingCtx]("weight")
	distance = Var[int, ShippingCtx]("distance")
	is_express = Var[bool, ShippingCtx]("is_express")

	base_rate = Match(
		(weight > 50, 15.0),
		(weight > 20, 10.0),
		(weight > 5, 5.0),
		default=2.0,
	)

	distance_multiplier = Match(
		(distance > 1000, 2.0),
		(distance > 500, 1.5),
		default=1.0,
	)

	express_surcharge = Match(
		(is_express, 10.0),
		default=0.0,
	)

	shipping_cost = (base_rate * distance_multiplier) + express_surcharge

	standard_near = ShippingCtx(weight=10.0, distance=100, is_express=False)
	assert shipping_cost.unwrap(standard_near) == 5.0

	heavy_far_express = ShippingCtx(weight=60.0, distance=1500, is_express=True)
	assert shipping_cost.unwrap(heavy_far_express) == 40.0

	medium_mid_standard = ShippingCtx(weight=30.0, distance=750, is_express=False)
	assert shipping_cost.unwrap(medium_mid_standard) == 15.0

	assert "(weight > 50 ->" in shipping_cost.to_string()
	assert "(distance > 1000 ->" in shipping_cost.to_string()


class DiscountCtx(NamedTuple):
	subtotal: float
	is_member: bool
	coupon_code: str


def test_composed_discount_pipeline() -> None:
	"""Demonstrate sequential discount application using composed expressions."""
	subtotal = Var[float, DiscountCtx]("subtotal")
	is_member = Var[bool, DiscountCtx]("is_member")
	coupon_code = Var[str, DiscountCtx]("coupon_code")

	member_discount = Match(
		(is_member & (subtotal >= 100), 0.15),
		(is_member, 0.10),
		default=0.0,
	)

	coupon_discount = Match(
		(coupon_code == "SAVE20", 0.20),
		(coupon_code == "SAVE10", 0.10),
		default=0.0,
	)

	after_member = subtotal * (Const(None, 1.0) - member_discount)
	final_price = after_member * (Const(None, 1.0) - coupon_discount)

	no_discounts = DiscountCtx(subtotal=50.0, is_member=False, coupon_code="")
	assert final_price.unwrap(no_discounts) == 50.0

	member_only = DiscountCtx(subtotal=50.0, is_member=True, coupon_code="")
	assert final_price.unwrap(member_only) == 45.0

	gold_member = DiscountCtx(subtotal=100.0, is_member=True, coupon_code="")
	assert final_price.unwrap(gold_member) == 85.0

	gold_with_coupon = DiscountCtx(subtotal=100.0, is_member=True, coupon_code="SAVE20")
	assert final_price.unwrap(gold_with_coupon) == 68.0


class GameCtx(NamedTuple):
	health: int
	has_shield: bool
	damage: int


def test_composed_damage_calculation() -> None:
	"""Demonstrate game-like damage calculation with multiple modifiers."""
	health = Var[int, GameCtx]("health")
	has_shield = Var[bool, GameCtx]("has_shield")
	damage = Var[int, GameCtx]("damage")

	damage_reduction = Match(
		(has_shield, 0.5),
		default=1.0,
	)

	actual_damage = damage * damage_reduction

	status = Match(
		(health <= 0, "DEAD"),
		(health < 20, "CRITICAL"),
		(health < 50, "WOUNDED"),
		default="HEALTHY",
	)

	assert status.unwrap(GameCtx(health=100, has_shield=False, damage=0)) == "HEALTHY"
	assert status.unwrap(GameCtx(health=30, has_shield=False, damage=0)) == "WOUNDED"
	assert status.unwrap(GameCtx(health=10, has_shield=False, damage=0)) == "CRITICAL"
	assert status.unwrap(GameCtx(health=0, has_shield=False, damage=0)) == "DEAD"

	assert actual_damage.unwrap(GameCtx(health=100, has_shield=True, damage=20)) == 10.0
	assert actual_damage.unwrap(GameCtx(health=100, has_shield=False, damage=20)) == 20.0


def test_match_as_condition_in_another_match() -> None:
	"""Demonstrate using a Match result as a condition in another Match."""
	n = Var[int, FizzBuzzCtx]("n")

	category = Match(
		((n % 15) == 0, "fizzbuzz"),
		((n % 3) == 0, "fizz"),
		((n % 5) == 0, "buzz"),
		default="number",
	)

	priority = Match(
		(category == "fizzbuzz", 3),
		(category == "fizz", 2),
		(category == "buzz", 2),
		default=1,
	)

	assert priority.unwrap(FizzBuzzCtx(n=15)) == 3
	assert priority.unwrap(FizzBuzzCtx(n=9)) == 2
	assert priority.unwrap(FizzBuzzCtx(n=10)) == 2
	assert priority.unwrap(FizzBuzzCtx(n=7)) == 1


def test_fizzbuzz_with_score_multiplier() -> None:
	"""Demonstrate FizzBuzz as part of a scoring system."""
	n = Var[int, FizzBuzzCtx]("n")

	base_points = Match(
		((n % 15) == 0, 100),
		((n % 3) == 0, 30),
		((n % 5) == 0, 50),
		default=n,
	)

	bonus_multiplier = Match(
		(n > 50, 3),
		(n > 20, 2),
		default=1,
	)

	total_score = base_points * bonus_multiplier

	assert total_score.unwrap(FizzBuzzCtx(n=15)) == 100
	assert total_score.unwrap(FizzBuzzCtx(n=30)) == 200
	assert total_score.unwrap(FizzBuzzCtx(n=60)) == 300
	assert total_score.unwrap(FizzBuzzCtx(n=9)) == 30
	assert total_score.unwrap(FizzBuzzCtx(n=25)) == 100
	assert total_score.unwrap(FizzBuzzCtx(n=7)) == 7
	assert total_score.unwrap(FizzBuzzCtx(n=37)) == 74

	assert "match" in total_score.to_string()


def test_full_serialization_of_composed_program() -> None:
	"""Verify serialization captures the full composed expression tree."""
	n = Var[int, FizzBuzzCtx]("n")

	fizzbuzz_numeric = Match(
		((n % 15) == 0, Const("fb", 15)),
		((n % 3) == 0, Const("f", 3)),
		((n % 5) == 0, Const("b", 5)),
		default=n,
	)

	doubled = fizzbuzz_numeric * 2
	is_big = doubled > 20

	assert is_big.to_string() == (
		"(((match ((n % 15) == 0 -> fb:15) ((n % 3) == 0 -> f:3) "
		"((n % 5) == 0 -> b:5) else n) * 2) > 20)"
	)

	assert is_big.to_string(FizzBuzzCtx(n=30)) == (
		"(((match ((n:30 % 15 -> 0) == 0 -> True -> fb:15) "
		"((n:30 % 3 -> 0) == 0 -> True -> f:3) "
		"((n:30 % 5 -> 0) == 0 -> True -> b:5) else n -> 15) * 2 -> 30) > 20 -> True)"
	)

	assert is_big.to_string(FizzBuzzCtx(n=7)) == (
		"(((match ((n:7 % 15 -> 7) == 0 -> False -> fb:15) "
		"((n:7 % 3 -> 1) == 0 -> False -> f:3) "
		"((n:7 % 5 -> 2) == 0 -> False -> b:5) else n -> 7) * 2 -> 14) > 20 -> False)"
	)


class CollatzCtx(NamedTuple):
	n: float


@pytest.mark.mypy_testing
def test_collatz_step() -> None:
	"""Collatz conjecture: if even, n/2; if odd, 3n+1.

	This is a classic FP example demonstrating Match as a conditional expression.
	In Haskell: collatz n = if even n then n `div` 2 else 3*n + 1
	"""
	n = Var[float, CollatzCtx]("n")

	collatz_step = Match(
		((n % 2) == 0, n / 2),
		default=(n * 3) + 1,
	)
	assert_type(collatz_step, MatchExpr[float | int, CollatzCtx])

	assert collatz_step.unwrap(CollatzCtx(n=16)) == 8
	assert collatz_step.unwrap(CollatzCtx(n=8)) == 4
	assert collatz_step.unwrap(CollatzCtx(n=4)) == 2
	assert collatz_step.unwrap(CollatzCtx(n=2)) == 1
	assert collatz_step.unwrap(CollatzCtx(n=7)) == 22
	assert collatz_step.unwrap(CollatzCtx(n=22)) == 11
	assert collatz_step.unwrap(CollatzCtx(n=11)) == 34

	assert collatz_step.to_string() == "(match ((n % 2) == 0 -> (n / 2)) else ((n * 3) + 1))"
	assert collatz_step.to_string(CollatzCtx(n=16)) == (
		"(match ((n:16 % 2 -> 0) == 0 -> True -> (n / 2)) else ((n * 3) + 1) -> 8.0)"
	)


class Point2DCtx(NamedTuple):
	x: int
	y: int


@pytest.mark.mypy_testing
def test_quadrant_classifier() -> None:
	"""Classify 2D points into quadrants - nested conditionals.

	Equivalent Haskell:
	quadrant x y
	  | x > 0 && y > 0 = 1
	  | x < 0 && y > 0 = 2
	  | x < 0 && y < 0 = 3
	  | x > 0 && y < 0 = 4
	  | otherwise = 0  -- on axis
	"""
	x = Var[int, Point2DCtx]("x")
	y = Var[int, Point2DCtx]("y")

	q1_cond = (x > 0) & (y > 0)
	assert_type(q1_cond, And[bool, Point2DCtx])

	quadrant = Match(
		(q1_cond, 1),
		((x < 0) & (y > 0), 2),
		((x < 0) & (y < 0), 3),
		((x > 0) & (y < 0), 4),
		default=0,
	)
	assert_type(quadrant, MatchExpr[int, Point2DCtx])

	assert quadrant.unwrap(Point2DCtx(x=5, y=10)) == 1
	assert quadrant.unwrap(Point2DCtx(x=-3, y=7)) == 2
	assert quadrant.unwrap(Point2DCtx(x=-2, y=-8)) == 3
	assert quadrant.unwrap(Point2DCtx(x=4, y=-1)) == 4
	assert quadrant.unwrap(Point2DCtx(x=0, y=5)) == 0
	assert quadrant.unwrap(Point2DCtx(x=3, y=0)) == 0

	assert quadrant.to_string() == (
		"(match ((x > 0) & (y > 0) -> 1) ((x < 0) & (y > 0) -> 2) "
		"((x < 0) & (y < 0) -> 3) ((x > 0) & (y < 0) -> 4) else 0)"
	)


@pytest.mark.mypy_testing
def test_nested_match_distance_from_origin() -> None:
	"""Compute taxicab distance with sign-aware absolute value via nested Match.

	abs(x) + abs(y) where abs is itself a Match expression.
	This demonstrates Match expressions as the *result values* of branches.
	"""
	x = Var[int, Point2DCtx]("x")
	y = Var[int, Point2DCtx]("y")

	neg_x = -x
	assert_type(neg_x, Neg[int, Point2DCtx])

	abs_x = Match((x >= 0, x), default=-x)
	abs_y = Match((y >= 0, y), default=-y)
	assert_type(abs_x, MatchExpr[int, Point2DCtx])

	taxicab_distance = abs_x + abs_y
	assert_type(taxicab_distance, Add[int, Point2DCtx])

	assert taxicab_distance.unwrap(Point2DCtx(x=3, y=4)) == 7
	assert taxicab_distance.unwrap(Point2DCtx(x=-3, y=4)) == 7
	assert taxicab_distance.unwrap(Point2DCtx(x=3, y=-4)) == 7
	assert taxicab_distance.unwrap(Point2DCtx(x=-3, y=-4)) == 7
	assert taxicab_distance.unwrap(Point2DCtx(x=0, y=0)) == 0

	assert abs_x.to_string() == "(match (x >= 0 -> x) else (-x))"
	assert abs_x.to_string(Point2DCtx(x=-3, y=0)) == (
		"(match (x:-3 >= 0 -> False -> x) else (-x) -> 3)"
	)
	assert taxicab_distance.to_string() == (
		"((match (x >= 0 -> x) else (-x)) + (match (y >= 0 -> y) else (-y)))"
	)


@pytest.mark.mypy_testing
def test_deeply_nested_match_in_match_arms() -> None:
	"""Match expressions as branch values - true Haskell case-of-case.

	This is the key test: Match results are themselves Match expressions,
	creating deeply nested conditional logic without explicit branching.

	Equivalent Haskell:
	classify x y = case (x > 0, y > 0) of
	  (True, _)  -> case y > 0 of
	                  True  -> "Q1: both positive"
	                  False -> case y == 0 of
	                             True  -> "positive x-axis"
	                             False -> "Q4: x+, y-"
	  (False, _) -> case x == 0 of
	                  True  -> case y > 0 of ...
	                  False -> case y > 0 of ...
	"""
	x = Var[int, Point2DCtx]("x")
	y = Var[int, Point2DCtx]("y")

	classify_positive_x = Match(
		(y > 0, "Q1: both positive"),
		(y == 0, "positive x-axis"),
		default="Q4: x+, y-",
	)
	assert_type(classify_positive_x, MatchExpr[str, Point2DCtx])

	classify_on_y_axis = Match(
		(y > 0, "positive y-axis"),
		(y == 0, "origin"),
		default="negative y-axis",
	)

	classify_negative_x = Match(
		(y > 0, "Q2: x-, y+"),
		(y == 0, "negative x-axis"),
		default="Q3: both negative",
	)

	full_classifier = Match(
		(x > 0, classify_positive_x),
		(x == 0, classify_on_y_axis),
		default=classify_negative_x,
	)
	assert_type(full_classifier, MatchExpr[str, Point2DCtx])

	assert full_classifier.unwrap(Point2DCtx(x=5, y=3)) == "Q1: both positive"
	assert full_classifier.unwrap(Point2DCtx(x=5, y=0)) == "positive x-axis"
	assert full_classifier.unwrap(Point2DCtx(x=5, y=-2)) == "Q4: x+, y-"
	assert full_classifier.unwrap(Point2DCtx(x=0, y=7)) == "positive y-axis"
	assert full_classifier.unwrap(Point2DCtx(x=0, y=0)) == "origin"
	assert full_classifier.unwrap(Point2DCtx(x=0, y=-4)) == "negative y-axis"
	assert full_classifier.unwrap(Point2DCtx(x=-3, y=2)) == "Q2: x-, y+"
	assert full_classifier.unwrap(Point2DCtx(x=-3, y=0)) == "negative x-axis"
	assert full_classifier.unwrap(Point2DCtx(x=-3, y=-1)) == "Q3: both negative"

	assert full_classifier.to_string() == (
		"(match (x > 0 -> (match (y > 0 -> Q1: both positive) "
		"(y == 0 -> positive x-axis) else Q4: x+, y-)) "
		"(x == 0 -> (match (y > 0 -> positive y-axis) "
		"(y == 0 -> origin) else negative y-axis)) "
		"else (match (y > 0 -> Q2: x-, y+) "
		"(y == 0 -> negative x-axis) else Q3: both negative))"
	)


class RationalCtx(NamedTuple):
	num: int
	denom: int


@pytest.mark.mypy_testing
def test_sign_function_composition() -> None:
	"""Compose sign functions: signum(a) * signum(b) for rational sign.

	Demonstrates composing Match results with arithmetic.
	signum x = case compare x 0 of { LT -> -1; EQ -> 0; GT -> 1 }
	"""
	num = Var[int, RationalCtx]("num")
	denom = Var[int, RationalCtx]("denom")

	signum_num = Match((num > 0, 1), (num == 0, 0), default=-1)
	signum_denom = Match((denom > 0, 1), (denom == 0, 0), default=-1)
	assert_type(signum_num, MatchExpr[int, RationalCtx])

	rational_sign = signum_num * signum_denom
	assert_type(rational_sign, Mul[int, RationalCtx])

	assert rational_sign.unwrap(RationalCtx(num=5, denom=3)) == 1
	assert rational_sign.unwrap(RationalCtx(num=-5, denom=3)) == -1
	assert rational_sign.unwrap(RationalCtx(num=5, denom=-3)) == -1
	assert rational_sign.unwrap(RationalCtx(num=-5, denom=-3)) == 1
	assert rational_sign.unwrap(RationalCtx(num=0, denom=3)) == 0
	assert rational_sign.unwrap(RationalCtx(num=5, denom=0)) == 0

	assert rational_sign.to_string() == (
		"((match (num > 0 -> 1) (num == 0 -> 0) else -1) * "
		"(match (denom > 0 -> 1) (denom == 0 -> 0) else -1))"
	)


class TriangleCtx(NamedTuple):
	a: int
	b: int
	c: int


@pytest.mark.mypy_testing
def test_triangle_classifier_deep_composition() -> None:
	"""Classify triangles: first check validity, then classify type.

	Demonstrates: Match on Match result, composed predicates, nested logic.

	Haskell:
	classify a b c
	  | not (a + b > c && b + c > a && a + c > b) = "invalid"
	  | a == b && b == c = "equilateral"
	  | a == b || b == c || a == c = "isoceles"
	  | otherwise = "scalene"
	"""
	a = Var[int, TriangleCtx]("a")
	b = Var[int, TriangleCtx]("b")
	c = Var[int, TriangleCtx]("c")

	valid = (a + b > c) & (b + c > a) & (a + c > b)
	assert_type(valid, And[bool, TriangleCtx])

	equilateral = (a == b) & (b == c)
	isoceles = (a == b) | (b == c) | (a == c)
	assert_type(isoceles, Or[bool, TriangleCtx])

	not_valid = ~valid
	assert_type(not_valid, Not[TriangleCtx])

	triangle_type = Match(
		(not_valid, "invalid"),
		(equilateral, "equilateral"),
		(isoceles, "isoceles"),
		default="scalene",
	)
	assert_type(triangle_type, MatchExpr[str, TriangleCtx])

	assert triangle_type.unwrap(TriangleCtx(a=3, b=3, c=3)) == "equilateral"
	assert triangle_type.unwrap(TriangleCtx(a=3, b=3, c=4)) == "isoceles"
	assert triangle_type.unwrap(TriangleCtx(a=3, b=4, c=5)) == "scalene"
	assert triangle_type.unwrap(TriangleCtx(a=1, b=1, c=10)) == "invalid"
	assert triangle_type.unwrap(TriangleCtx(a=5, b=5, c=5)) == "equilateral"

	assert triangle_type.to_string() == (
		"(match (not ((((a + b) > c) & ((b + c) > a)) & ((a + c) > b)) -> invalid) "
		"((a == b) & (b == c) -> equilateral) "
		"(((a == b) | (b == c)) | (a == c) -> isoceles) else scalene)"
	)


class ExprTreeCtx(NamedTuple):
	op: str
	left_val: float
	right_val: float


@pytest.mark.mypy_testing
def test_simple_interpreter_pattern() -> None:
	"""A tiny expression interpreter - the essence of Lisp eval.

	Interprets (op left right) where op is "+", "-", "*", "/".
	This demonstrates using Match for dispatch based on a discriminant.
	"""
	op = Var[str, ExprTreeCtx]("op")
	left = Var[float, ExprTreeCtx]("left_val")
	right = Var[float, ExprTreeCtx]("right_val")

	op_eq_plus = op == "+"
	assert_type(op_eq_plus, Eq[str, ExprTreeCtx])

	left_plus_right = left + right
	assert_type(left_plus_right, Add[float, ExprTreeCtx])

	left_minus_right = left - right
	assert_type(left_minus_right, Sub[float, ExprTreeCtx])

	left_div_right = left / right
	assert_type(left_div_right, Div[float, ExprTreeCtx])

	interpret = Match(
		(op_eq_plus, left_plus_right),
		(op == "-", left_minus_right),
		(op == "*", left * right),
		(op == "/", left_div_right),
		default=0.0,
	)
	assert_type(interpret, MatchExpr[float, ExprTreeCtx])

	assert interpret.unwrap(ExprTreeCtx(op="+", left_val=10, right_val=3)) == 13
	assert interpret.unwrap(ExprTreeCtx(op="-", left_val=10, right_val=3)) == 7
	assert interpret.unwrap(ExprTreeCtx(op="*", left_val=10, right_val=3)) == 30
	assert interpret.unwrap(ExprTreeCtx(op="/", left_val=12, right_val=3)) == 4
	assert interpret.unwrap(ExprTreeCtx(op="^", left_val=10, right_val=3)) == 0

	assert interpret.to_string() == (
		"(match (op == + -> (left_val + right_val)) "
		"(op == - -> (left_val - right_val)) "
		"(op == * -> (left_val * right_val)) "
		"(op == / -> (left_val / right_val)) else 0.0)"
	)


class ColorCtx(NamedTuple):
	r: float
	g: float
	b: float


@pytest.mark.mypy_testing
def test_color_classifier_with_luminance() -> None:
	"""Classify colors by computing luminance and categorizing.

	Demonstrates: arithmetic on variables, then Match on result.
	Luminance ~= 0.299*R + 0.587*G + 0.114*B
	"""
	r = Var[float, ColorCtx]("r")
	g = Var[float, ColorCtx]("g")
	b = Var[float, ColorCtx]("b")

	luminance = r * 0.299 + g * 0.587 + b * 0.114
	assert_type(luminance, Add[float, ColorCtx])

	lum_gt_200 = luminance > 200
	assert_type(lum_gt_200, Gt[int, ColorCtx])

	brightness = Match(
		(lum_gt_200, "very bright"),
		(luminance > 128, "bright"),
		(luminance > 64, "medium"),
		(luminance > 32, "dark"),
		default="very dark",
	)
	assert_type(brightness, MatchExpr[str, ColorCtx])

	assert brightness.unwrap(ColorCtx(r=255, g=255, b=255)) == "very bright"
	assert brightness.unwrap(ColorCtx(r=200, g=200, b=200)) == "bright"
	assert brightness.unwrap(ColorCtx(r=100, g=100, b=100)) == "medium"
	assert brightness.unwrap(ColorCtx(r=50, g=50, b=50)) == "dark"
	assert brightness.unwrap(ColorCtx(r=10, g=10, b=10)) == "very dark"
	assert brightness.unwrap(ColorCtx(r=255, g=0, b=0)) == "medium"


@pytest.mark.mypy_testing
def test_match_with_arithmetic_in_both_condition_and_result() -> None:
	"""Complex expressions in both guard and body - full FP style.

	The condition involves computation, and the result involves more computation.
	"""
	x = Var[int, Point2DCtx]("x")
	y = Var[int, Point2DCtx]("y")

	distance_squared = x * x + y * y
	assert_type(distance_squared, Add[int, Point2DCtx])

	scaled_result = Match(
		(distance_squared > 100, (x + y) * 2),
		(distance_squared > 25, x + y),
		default=(x + y) * Const("half", 0.5),
	)
	assert_type(scaled_result, MatchExpr[int | float, Point2DCtx])

	assert scaled_result.unwrap(Point2DCtx(x=8, y=8)) == 32
	assert scaled_result.unwrap(Point2DCtx(x=4, y=4)) == 8
	assert scaled_result.unwrap(Point2DCtx(x=2, y=2)) == 2.0

	assert scaled_result.to_string() == (
		"(match (((x * x) + (y * y)) > 100 -> ((x + y) * 2)) "
		"(((x * x) + (y * y)) > 25 -> (x + y)) else ((x + y) * half:0.5))"
	)


class ChainCtx(NamedTuple):
	a: int
	b: int
	c: int


@pytest.mark.mypy_testing
def test_three_way_min_max_via_nested_match() -> None:
	"""Compute min and max of three values using only nested Match.

	This is a classic functional programming exercise: implement min3/max3
	using only binary comparisons and conditionals.

	min3 a b c = min a (min b c)  -- but expressed as nested case
	"""
	a = Var[int, ChainCtx]("a")
	b = Var[int, ChainCtx]("b")
	c = Var[int, ChainCtx]("c")

	b_le_c = b <= c
	assert_type(b_le_c, Le[int, ChainCtx])

	min_bc = Match((b_le_c, b), default=c)
	assert_type(min_bc, MatchExpr[int, ChainCtx])

	a_le_min_bc = a <= min_bc
	assert_type(a_le_min_bc, Le[int, ChainCtx])

	min_abc = Match((a_le_min_bc, a), default=min_bc)

	b_ge_c = b >= c
	assert_type(b_ge_c, Ge[int, ChainCtx])

	max_bc = Match((b_ge_c, b), default=c)
	max_abc = Match((a >= max_bc, a), default=max_bc)

	assert min_abc.unwrap(ChainCtx(a=1, b=2, c=3)) == 1
	assert min_abc.unwrap(ChainCtx(a=3, b=1, c=2)) == 1
	assert min_abc.unwrap(ChainCtx(a=2, b=3, c=1)) == 1
	assert min_abc.unwrap(ChainCtx(a=5, b=5, c=5)) == 5

	assert max_abc.unwrap(ChainCtx(a=1, b=2, c=3)) == 3
	assert max_abc.unwrap(ChainCtx(a=3, b=1, c=2)) == 3
	assert max_abc.unwrap(ChainCtx(a=2, b=3, c=1)) == 3
	assert max_abc.unwrap(ChainCtx(a=5, b=5, c=5)) == 5

	assert min_abc.to_string() == (
		"(match (a <= (match (b <= c -> b) else c) -> a) else (match (b <= c -> b) else c))"
	)


@pytest.mark.mypy_testing
def test_clamp_via_composed_min_max() -> None:
	"""Clamp a value to [lo, hi] using composed min/max Match expressions.

	clamp lo hi x = max lo (min x hi)
	"""
	a = Var[int, ChainCtx]("a")
	b = Var[int, ChainCtx]("b")
	c = Var[int, ChainCtx]("c")

	a_lt_b = a < b
	assert_type(a_lt_b, Lt[int, ChainCtx])

	clamped = Match(
		(a_lt_b, b),
		(a > c, c),
		default=a,
	)
	assert_type(clamped, MatchExpr[int, ChainCtx])

	assert clamped.unwrap(ChainCtx(a=5, b=0, c=10)) == 5
	assert clamped.unwrap(ChainCtx(a=-5, b=0, c=10)) == 0
	assert clamped.unwrap(ChainCtx(a=15, b=0, c=10)) == 10
	assert clamped.unwrap(ChainCtx(a=0, b=0, c=10)) == 0
	assert clamped.unwrap(ChainCtx(a=10, b=0, c=10)) == 10

	assert clamped.to_string() == "(match (a < b -> b) (a > c -> c) else a)"


class MaybeCtx(NamedTuple):
	is_just: bool
	value: int


@pytest.mark.mypy_testing
def test_maybe_monad_simulation() -> None:
	"""Simulate Maybe monad's fmap and bind-like behavior.

	Since we can't have true sum types, we use a tagged record.
	fmap f Nothing = Nothing
	fmap f (Just x) = Just (f x)
	"""
	is_just = Var[bool, MaybeCtx]("is_just")
	value = Var[int, MaybeCtx]("value")

	fmap_double = Match(
		(is_just, value * 2),
		default=0,
	)
	assert_type(fmap_double, MatchExpr[int, MaybeCtx])

	fmap_result_is_just = Match(
		(is_just & (value > 0), True),
		default=False,
	)
	assert_type(fmap_result_is_just, MatchExpr[bool, MaybeCtx])

	assert fmap_double.unwrap(MaybeCtx(is_just=True, value=21)) == 42
	assert fmap_double.unwrap(MaybeCtx(is_just=False, value=21)) == 0

	assert fmap_result_is_just.unwrap(MaybeCtx(is_just=True, value=5)) is True
	assert fmap_result_is_just.unwrap(MaybeCtx(is_just=True, value=-5)) is False
	assert fmap_result_is_just.unwrap(MaybeCtx(is_just=False, value=5)) is False

	assert fmap_double.to_string() == "(match (is_just -> (value * 2)) else 0)"


class LeapYearCtx(NamedTuple):
	year: int


@pytest.mark.mypy_testing
def test_leap_year_complex_predicate() -> None:
	"""Determine leap year - classic nested conditional logic.

	A year is a leap year if:
	- divisible by 4 AND
	- (not divisible by 100 OR divisible by 400)

	isLeapYear y = (y `mod` 4 == 0) && ((y `mod` 100 /= 0) || (y `mod` 400 == 0))
	"""
	year = Var[int, LeapYearCtx]("year")

	div_by_4 = (year % 4) == 0
	div_by_100 = (year % 100) == 0
	div_by_400 = (year % 400) == 0
	assert_type(div_by_4, Eq[int, LeapYearCtx])

	not_div_by_100 = ~div_by_100
	assert_type(not_div_by_100, Not[LeapYearCtx])

	is_leap = Match(
		(div_by_4 & (not_div_by_100 | div_by_400), True),
		default=False,
	)
	assert_type(is_leap, MatchExpr[bool, LeapYearCtx])

	assert is_leap.unwrap(LeapYearCtx(year=2000)) is True
	assert is_leap.unwrap(LeapYearCtx(year=2004)) is True
	assert is_leap.unwrap(LeapYearCtx(year=1900)) is False
	assert is_leap.unwrap(LeapYearCtx(year=2001)) is False
	assert is_leap.unwrap(LeapYearCtx(year=2024)) is True
	assert is_leap.unwrap(LeapYearCtx(year=2100)) is False

	assert is_leap.to_string() == (
		"(match (((year % 4) == 0) & ((not ((year % 100) == 0)) | ((year % 400) == 0)) -> True) "
		"else False)"
	)


class RomanCtx(NamedTuple):
	n: int


@pytest.mark.mypy_testing
def test_roman_numeral_digit_selector() -> None:
	"""Select Roman numeral representation for single digits.

	This tests many-branch Match with string results and nested default.
	"""
	n = Var[int, RomanCtx]("n")

	n_eq_1 = n == 1
	assert_type(n_eq_1, Eq[int, RomanCtx])

	inner_match = Match(
		(n == 7, "VII"),
		(n == 8, "VIII"),
		(n == 9, "IX"),
		default="?",
	)
	assert_type(inner_match, MatchExpr[str, RomanCtx])

	roman_digit = Match(
		(n_eq_1, "I"),
		(n == 2, "II"),
		(n == 3, "III"),
		(n == 4, "IV"),
		(n == 5, "V"),
		(n == 6, "VI"),
		default=inner_match,
	)
	assert_type(roman_digit, MatchExpr[str, RomanCtx])

	assert roman_digit.unwrap(RomanCtx(n=1)) == "I"
	assert roman_digit.unwrap(RomanCtx(n=4)) == "IV"
	assert roman_digit.unwrap(RomanCtx(n=5)) == "V"
	assert roman_digit.unwrap(RomanCtx(n=7)) == "VII"
	assert roman_digit.unwrap(RomanCtx(n=9)) == "IX"
	assert roman_digit.unwrap(RomanCtx(n=0)) == "?"
	assert roman_digit.unwrap(RomanCtx(n=10)) == "?"

	assert roman_digit.to_string() == (
		"(match (n == 1 -> I) (n == 2 -> II) (n == 3 -> III) (n == 4 -> IV) "
		"(n == 5 -> V) (n == 6 -> VI) else (match (n == 7 -> VII) "
		"(n == 8 -> VIII) (n == 9 -> IX) else ?))"
	)


class ChessCtx(NamedTuple):
	piece: str
	from_rank: int
	from_file: int
	to_rank: int
	to_file: int


@pytest.mark.mypy_testing
def test_chess_move_validator_complex() -> None:
	"""Validate basic chess piece movements.

	Demonstrates complex multi-variable predicates with nested Match for abs.
	"""
	piece = Var[str, ChessCtx]("piece")
	from_rank = Var[int, ChessCtx]("from_rank")
	from_file = Var[int, ChessCtx]("from_file")
	to_rank = Var[int, ChessCtx]("to_rank")
	to_file = Var[int, ChessCtx]("to_file")

	rank_diff = Match((to_rank >= from_rank, to_rank - from_rank), default=from_rank - to_rank)
	file_diff = Match((to_file >= from_file, to_file - from_file), default=from_file - to_file)
	assert_type(rank_diff, MatchExpr[int, ChessCtx])

	is_rook_move = (rank_diff == 0) | (file_diff == 0)
	is_bishop_move = rank_diff == file_diff
	is_queen_move = is_rook_move | is_bishop_move
	is_king_move = (rank_diff <= 1) & (file_diff <= 1)
	is_knight_move = ((rank_diff == 2) & (file_diff == 1)) | ((rank_diff == 1) & (file_diff == 2))
	assert_type(is_knight_move, Or[bool, ChessCtx])

	is_valid_move = Match(
		((piece == "R") & is_rook_move, True),
		((piece == "B") & is_bishop_move, True),
		((piece == "Q") & is_queen_move, True),
		((piece == "K") & is_king_move, True),
		((piece == "N") & is_knight_move, True),
		default=False,
	)
	assert_type(is_valid_move, MatchExpr[bool, ChessCtx])

	assert is_valid_move.unwrap(ChessCtx(piece="R", from_rank=1, from_file=1, to_rank=1, to_file=8))
	assert is_valid_move.unwrap(ChessCtx(piece="R", from_rank=1, from_file=1, to_rank=8, to_file=1))
	assert not is_valid_move.unwrap(
		ChessCtx(piece="R", from_rank=1, from_file=1, to_rank=2, to_file=2)
	)

	assert is_valid_move.unwrap(ChessCtx(piece="B", from_rank=1, from_file=1, to_rank=4, to_file=4))
	assert not is_valid_move.unwrap(
		ChessCtx(piece="B", from_rank=1, from_file=1, to_rank=1, to_file=4)
	)

	assert is_valid_move.unwrap(ChessCtx(piece="N", from_rank=1, from_file=2, to_rank=3, to_file=3))
	assert is_valid_move.unwrap(ChessCtx(piece="N", from_rank=1, from_file=2, to_rank=2, to_file=4))
	assert not is_valid_move.unwrap(
		ChessCtx(piece="N", from_rank=1, from_file=1, to_rank=2, to_file=2)
	)


class RecursivePatternCtx(NamedTuple):
	depth: int
	value: int


@pytest.mark.mypy_testing
def test_tower_of_hanoi_move_count() -> None:
	"""Compute moves needed for Tower of Hanoi at various depths.

	The formula is 2^n - 1, but we'll compute it via Match-based selection
	for small n to demonstrate pattern matching on discrete cases.
	"""
	depth = Var[int, RecursivePatternCtx]("depth")

	hanoi_moves = Match(
		(depth == 0, 0),
		(depth == 1, 1),
		(depth == 2, 3),
		(depth == 3, 7),
		(depth == 4, 15),
		(depth == 5, 31),
		default=(2**depth) - 1,
	)
	assert_type(hanoi_moves, MatchExpr[int, RecursivePatternCtx])

	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=0, value=0)) == 0
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=1, value=0)) == 1
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=2, value=0)) == 3
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=3, value=0)) == 7
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=4, value=0)) == 15
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=5, value=0)) == 31
	assert hanoi_moves.unwrap(RecursivePatternCtx(depth=10, value=0)) == 1023

	assert hanoi_moves.to_string() == (
		"(match (depth == 0 -> 0) (depth == 1 -> 1) (depth == 2 -> 3) "
		"(depth == 3 -> 7) (depth == 4 -> 15) (depth == 5 -> 31) else ((2^depth) - 1))"
	)


class FibonacciSelectCtx(NamedTuple):
	n: int


@pytest.mark.mypy_testing
def test_fibonacci_selector() -> None:
	"""Select from precomputed Fibonacci numbers - demonstrates many-branch Match."""
	n = Var[int, FibonacciSelectCtx]("n")

	fib = Match(
		(n == 0, 0),
		(n == 1, 1),
		(n == 2, 1),
		(n == 3, 2),
		(n == 4, 3),
		(n == 5, 5),
		default=Match(
			(n == 6, 8),
			(n == 7, 13),
			(n == 8, 21),
			(n == 9, 34),
			(n == 10, 55),
			default=None,
		),
	)
	assert_type(fib, MatchExpr[int | None, FibonacciSelectCtx])

	assert fib.unwrap(FibonacciSelectCtx(n=0)) == 0
	assert fib.unwrap(FibonacciSelectCtx(n=1)) == 1
	assert fib.unwrap(FibonacciSelectCtx(n=5)) == 5
	assert fib.unwrap(FibonacciSelectCtx(n=10)) == 55
	assert fib.unwrap(FibonacciSelectCtx(n=11)) is None

	assert fib.to_string() == (
		"(match (n == 0 -> 0) (n == 1 -> 1) (n == 2 -> 1) (n == 3 -> 2) "
		"(n == 4 -> 3) (n == 5 -> 5) else (match (n == 6 -> 8) (n == 7 -> 13) "
		"(n == 8 -> 21) (n == 9 -> 34) (n == 10 -> 55) else None))"
	)


@pytest.mark.mypy_testing
def test_expression_tree_depth_classification() -> None:
	"""Classify expression complexity based on depth of nesting."""
	n = Var[int, FizzBuzzCtx]("n")

	level_1 = n + 1
	assert_type(level_1, Add[int, FizzBuzzCtx])

	level_2 = level_1 * 2
	assert_type(level_2, Mul[int, FizzBuzzCtx])

	level_3 = Match((level_2 > 10, level_2), default=level_1)
	assert_type(level_3, MatchExpr[int, FizzBuzzCtx])

	level_4 = Match(
		(level_3 > 20, level_3 * 3),
		(level_3 > 10, level_3 * 2),
		default=level_3,
	)
	assert_type(level_4, MatchExpr[int, FizzBuzzCtx])

	level_5 = Match(
		(level_4 > 100, "huge"),
		(level_4 > 50, "large"),
		(level_4 > 20, "medium"),
		default="small",
	)
	assert_type(level_5, MatchExpr[str, FizzBuzzCtx])

	assert level_5.unwrap(FizzBuzzCtx(n=1)) == "small"
	assert level_5.unwrap(FizzBuzzCtx(n=5)) == "medium"
	assert level_5.unwrap(FizzBuzzCtx(n=10)) == "large"
	assert level_5.unwrap(FizzBuzzCtx(n=20)) == "huge"


class DayCtx(NamedTuple):
	day: int
	month: int
	year: int


@pytest.mark.mypy_testing
def test_day_of_year_calculation() -> None:
	"""Calculate day of year - complex multi-branch with arithmetic.

	This demonstrates cumulative computation via deeply nested Match.
	"""
	day = Var[int, DayCtx]("day")
	month = Var[int, DayCtx]("month")
	year = Var[int, DayCtx]("year")

	is_leap = ((year % 4) == 0) & (~((year % 100) == 0) | ((year % 400) == 0))
	assert_type(is_leap, And[bool, DayCtx])

	days_before_month = Match(
		(month == 1, 0),
		(month == 2, 31),
		(month == 3, Match((is_leap, 60), default=59)),
		(month == 4, Match((is_leap, 91), default=90)),
		(month == 5, Match((is_leap, 121), default=120)),
		(month == 6, Match((is_leap, 152), default=151)),
		default=Match(
			(month == 7, Match((is_leap, 182), default=181)),
			(month == 8, Match((is_leap, 213), default=212)),
			(month == 9, Match((is_leap, 244), default=243)),
			(month == 10, Match((is_leap, 274), default=273)),
			(month == 11, Match((is_leap, 305), default=304)),
			(month == 12, Match((is_leap, 335), default=334)),
			default=0,
		),
	)
	assert_type(days_before_month, MatchExpr[int | MatchExpr[int, DayCtx], DayCtx])

	day_of_year = days_before_month + day
	assert_type(day_of_year, Add[int, DayCtx])

	assert day_of_year.unwrap(DayCtx(day=1, month=1, year=2024)) == 1
	assert day_of_year.unwrap(DayCtx(day=31, month=1, year=2024)) == 31
	assert day_of_year.unwrap(DayCtx(day=1, month=2, year=2024)) == 32
	assert day_of_year.unwrap(DayCtx(day=29, month=2, year=2024)) == 60
	assert day_of_year.unwrap(DayCtx(day=1, month=3, year=2024)) == 61
	assert day_of_year.unwrap(DayCtx(day=1, month=3, year=2023)) == 60
	assert day_of_year.unwrap(DayCtx(day=31, month=12, year=2024)) == 366
	assert day_of_year.unwrap(DayCtx(day=31, month=12, year=2023)) == 365
