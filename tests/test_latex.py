# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import NamedTuple

from mahonia import (
	Approximately,
	Const,
	Percent,
	PlusMinus,
	Predicate,
	Var,
)
from mahonia.latex import LatexCtx, Show, latex
from mahonia.stats import Count, Mean, Median, Percentile, Range, SizedIterable, StdDev


class Ctx(NamedTuple):
	x: int
	y: int
	f: float = 1.5
	name: str = "test"
	alpha: float = 2.5
	beta: float = 3.0
	gamma: float = 4.0
	delta: float = 5.0
	theta: float = 6.0
	pi: float = 3.14159
	sigma: float = 7.0
	omega: float = 8.0


ctx = Ctx(x=5, y=10)


class StatsCtx(NamedTuple):
	values: SizedIterable[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
	measurements: SizedIterable[float] = [10.0, 15.0, 20.0, 25.0, 30.0]


stats_ctx = StatsCtx()


# Basic Expressions Tests


def test_var_simple() -> None:
	"""Test simple variable conversion."""
	x = Var[int, Ctx]("x")
	assert latex(x) == "x"


def test_var_subscript_single() -> None:
	"""Test variable with single character subscript."""
	x1 = Var[int, Ctx]("x_1")
	assert latex(x1) == "x_1"


def test_var_subscript_multiple() -> None:
	"""Test variable with multiple character subscript."""
	x_max = Var[int, Ctx]("x_max")
	assert latex(x_max) == "x_{max}"


def test_var_greek_letters() -> None:
	"""Test Greek letter variable names."""
	alpha = Var[float, Ctx]("alpha")
	beta = Var[float, Ctx]("beta")
	gamma = Var[float, Ctx]("gamma")
	delta = Var[float, Ctx]("delta")
	epsilon = Var[float, Ctx]("epsilon")
	theta = Var[float, Ctx]("theta")
	pi = Var[float, Ctx]("pi")
	sigma = Var[float, Ctx]("sigma")
	omega = Var[float, Ctx]("omega")

	assert latex(alpha) == "\\alpha"
	assert latex(beta) == "\\beta"
	assert latex(gamma) == "\\gamma"
	assert latex(delta) == "\\delta"
	assert latex(epsilon) == "\\epsilon"
	assert latex(theta) == "\\theta"
	assert latex(pi) == "\\pi"
	assert latex(sigma) == "\\sigma"
	assert latex(omega) == "\\omega"


def test_const_with_name() -> None:
	"""Test constant with name."""
	c = Const("Pi", 3.14159)
	assert latex(c) == "\\pi"  # "Pi" gets converted to Greek letter π


def test_const_without_name() -> None:
	"""Test constant without name."""
	c = Const(None, 42)
	assert latex(c) == "42"


def test_const_float() -> None:
	"""Test float constant."""
	c = Const(None, 3.14)
	assert latex(c) == "3.14"


def test_const_negative() -> None:
	"""Test negative constant."""
	c = Const(None, -5)
	assert latex(c) == "-5"


# Arithmetic Operations Tests


def test_addition() -> None:
	"""Test addition operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y
	assert latex(expr) == "x + y"


def test_addition_with_constant() -> None:
	"""Test addition with constant."""
	x = Var[int, Ctx]("x")
	c = Const("Five", 5)
	expr = x + c
	assert latex(expr) == "x + Five"


def test_subtraction() -> None:
	"""Test subtraction operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x - y
	assert latex(expr) == "x - y"


def test_subtraction_with_parentheses() -> None:
	"""Test subtraction that requires parentheses."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr = x - (y + z)
	assert latex(expr) == "x - (y + z)"


def test_multiplication() -> None:
	"""Test multiplication operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x * y
	assert latex(expr) == "x \\cdot y"


def test_multiplication_with_parentheses() -> None:
	"""Test multiplication that requires parentheses."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr = (x + y) * (z - x)
	assert latex(expr) == "(x + y) \\cdot (z - x)"


def test_division() -> None:
	"""Test division operation."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	expr = x / y
	assert latex(expr) == "\\frac{x}{y}"


def test_division_complex() -> None:
	"""Test division with complex expressions."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	z = Var[float, Ctx]("z")
	expr = (x + y) / (z - x)
	assert latex(expr) == "\\frac{x + y}{z - x}"


def test_power_simple() -> None:
	"""Test power operation."""
	x = Var[int, Ctx]("x")
	expr = x**2
	assert latex(expr) == "x^2"


def test_power_variable_exponent() -> None:
	"""Test power with variable exponent."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x**y
	assert latex(expr) == "x^{y}"


def test_power_complex_exponent() -> None:
	"""Test power with complex exponent."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x ** (y + 1)
	assert latex(expr) == "x^{y + 1}"


def test_power_with_parentheses() -> None:
	"""Test power that requires parentheses for base."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x + y) ** 2
	assert latex(expr) == "(x + y)^2"


# Comparison Operations Tests


def test_equality() -> None:
	"""Test equality operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x == y
	assert latex(expr) == "x = y"


def test_inequality() -> None:
	"""Test inequality operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x != y
	assert latex(expr) == "x \\neq y"


def test_less_than() -> None:
	"""Test less than operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x < y
	assert latex(expr) == "x < y"


def test_less_equal() -> None:
	"""Test less than or equal operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x <= y
	assert latex(expr) == "x \\leq y"


def test_greater_than() -> None:
	"""Test greater than operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x > y
	assert latex(expr) == "x > y"


def test_greater_equal() -> None:
	"""Test greater than or equal operator."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x >= y
	assert latex(expr) == "x \\geq y"


def test_comparison_with_expression() -> None:
	"""Test comparison with arithmetic expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x + 5) > (y * 2)
	assert latex(expr) == "x + 5 > y \\cdot 2"


# Logical Operations Tests


def test_and_operation() -> None:
	"""Test logical AND operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x > 5) & (y < 10)
	assert latex(expr) == "x > 5 \\land y < 10"


def test_or_operation() -> None:
	"""Test logical OR operation."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x > 5) | (y < 10)
	assert latex(expr) == "x > 5 \\lor y < 10"


def test_not_operation() -> None:
	"""Test logical NOT operation."""
	x = Var[int, Ctx]("x")
	expr = ~(x > 5)
	assert latex(expr) == "\\neg x > 5"


def test_not_with_parentheses() -> None:
	"""Test NOT operation that requires parentheses."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = ~((x > 5) & (y < 10))
	assert latex(expr) == "\\neg (x > 5 \\land y < 10)"


def test_complex_logical_expression() -> None:
	"""Test complex logical expression."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr = ((x > 0) & (y > 0)) | (z < 0)
	assert latex(expr) == "x > 0 \\land y > 0 \\lor z < 0"


# Special Expressions Tests


def test_plus_minus() -> None:
	"""Test plus-minus expression."""
	pm = PlusMinus("measurement", 5.0, 0.1)
	assert latex(pm) == "measurement \\pm 0.1"


def test_plus_minus_without_name() -> None:
	"""Test plus-minus without name."""
	pm = PlusMinus(None, 5.0, 0.1)
	assert latex(pm) == "5.0 \\pm 0.1"


def test_percent() -> None:
	"""Test percent expression."""
	pct = Percent("measurement", 5.0, 2.0)
	assert latex(pct) == "measurement \\pm 2.0\\%"


def test_percent_without_name() -> None:
	"""Test percent without name."""
	pct = Percent(None, 5.0, 2.0)
	assert latex(pct) == "5.0 \\pm 2.0\\%"


def test_approximately() -> None:
	"""Test approximately operation."""
	x = Var[float, Ctx]("x")
	target = PlusMinus("target", 5.0, 0.1)
	expr = Approximately(x, target)
	assert latex(expr) == "x \\approx target \\pm 0.1"


def test_predicate_with_name() -> None:
	"""Test predicate with name."""
	x = Var[int, Ctx]("x")
	pred = Predicate("condition", x > 5)
	assert latex(pred) == "\\text{condition}: x > 5"


def test_predicate_without_name() -> None:
	"""Test predicate without name."""
	x = Var[int, Ctx]("x")
	pred = Predicate(None, x > 5)
	assert latex(pred) == "x > 5"


# Complex Expressions Tests


def test_nested_arithmetic() -> None:
	"""Test deeply nested arithmetic expressions."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	z = Var[float, Ctx]("z")
	expr = ((x + y) * z) / (x - y)
	assert latex(expr) == "\\frac{(x + y) \\cdot z}{x - y}"


def test_polynomial_expression() -> None:
	"""Test polynomial-like expression."""
	x = Var[float, Ctx]("x")
	two = Const("Two", 2)
	one = Const("One", 1)
	expr = x**3 + two * x**2 - x + one
	assert latex(expr) == "x^3 + Two \\cdot x^2 - x + One"


def test_quadratic_formula() -> None:
	"""Test quadratic formula representation."""
	a = Var[float, Ctx]("a")
	b = Var[float, Ctx]("b")
	c = Var[float, Ctx]("c")
	four = Const("Four", 4)
	two = Const("Two", 2)
	# x = (-b ± sqrt(b² - 4ac)) / (2a)
	discriminant = b**2 - four * a * c
	neg_b = Const("NegB", -1) * b
	expr = (neg_b + discriminant) / (two * a)  # Just the positive part
	assert latex(expr) == "\\frac{NegB \\cdot b + b^2 - Four \\cdot a \\cdot c}{Two \\cdot a}"


def test_complex_logical_with_arithmetic() -> None:
	"""Test complex expression mixing logical and arithmetic operations."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	expr = ((x + y) > 10) & ((x * y) < 50) | (x / y > 2)
	assert latex(expr) == "x + y > 10 \\land x \\cdot y < 50 \\lor \\frac{x}{y} > 2"


def test_nested_function_composition() -> None:
	"""Test nested function-like expressions."""
	x = Var[float, Ctx]("x")
	target1 = PlusMinus("target1", 5.0, 0.1)
	target2 = PlusMinus("target2", 10.0, 0.2)
	expr1: Approximately[float, Ctx] = Approximately(x, target1)
	two = Const(None, 2.0)  # Use float constant to maintain float type
	power_expr = x**two
	expr2: Approximately[float, Ctx] = Approximately(power_expr, target2)
	combined = expr1 & expr2
	assert latex(combined) == "x \\approx target1 \\pm 0.1 \\land x^{2.0} \\approx target2 \\pm 0.2"


def test_deeply_nested_parentheses() -> None:
	"""Test expression that requires multiple levels of parentheses."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	z = Var[float, Ctx]("z")
	w = Var[float, Ctx]("w")
	expr = (x + (y - (z * w))) / (x - (y + z))
	assert latex(expr) == "\\frac{x + y - z \\cdot w}{x - (y + z)}"


# Parentheses Logic Tests


def test_addition_in_multiplication() -> None:
	"""Test addition inside multiplication needs parentheses."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr = (x + y) * z
	assert latex(expr) == "(x + y) \\cdot z"


def test_multiplication_in_addition() -> None:
	"""Test multiplication inside addition doesn't need parentheses."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr = x + y * z
	assert latex(expr) == "x + y \\cdot z"


def test_subtraction_right_side_parentheses() -> None:
	"""Test right side of subtraction needs parentheses for addition/subtraction."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	expr1 = x - (y + z)
	expr2 = x - (y - z)
	assert latex(expr1) == "x - (y + z)"
	assert latex(expr2) == "x - (y - z)"


def test_division_right_side_parentheses() -> None:
	"""Test right side of division needs parentheses for multiplication/division."""
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	z = Var[float, Ctx]("z")
	expr1 = x / (y * z)
	expr2 = x / (y / z)
	assert latex(expr1) == "\\frac{x}{y \\cdot z}"
	assert latex(expr2) == "\\frac{x}{\\frac{y}{z}}"


def test_power_base_parentheses() -> None:
	"""Test power base needs parentheses for lower precedence operations."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x + y) ** 2
	assert latex(expr) == "(x + y)^2"


def test_logical_precedence() -> None:
	"""Test logical operator precedence."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	z = Var[int, Ctx]("z")
	# OR has lower precedence than AND
	expr = (x > 0) | (y > 0) & (z > 0)
	assert latex(expr) == "x > 0 \\lor y > 0 \\land z > 0"


# Edge Cases Tests


def test_unknown_expression_type() -> None:
	"""Test handling of unknown expression types."""
	# This is tricky to test since we can't easily create unknown types
	# In real usage, this would be handled by the default case
	pass


def test_empty_variable_name() -> None:
	"""Test variable with empty name."""
	x = Var[int, Ctx]("")
	assert latex(x) == ""


def test_special_characters_in_names() -> None:
	"""Test special characters in variable names."""
	x = Var[int, Ctx]("x_prime")
	assert latex(x) == "x_{prime}"


def test_zero_values() -> None:
	"""Test expressions with zero values."""
	zero = Const(None, 0)
	x = Var[int, Ctx]("x")
	expr1 = x + zero
	expr2 = x * zero
	assert latex(expr1) == "x + 0"
	assert latex(expr2) == "x \\cdot 0"


def test_negative_constants_in_expressions() -> None:
	"""Test negative constants in complex expressions."""
	x = Var[int, Ctx]("x")
	neg = Const(None, -5)
	expr = x + neg
	assert latex(expr) == "x + -5"


def test_very_long_variable_names() -> None:
	"""Test very long variable names."""
	long_name = "very_long_variable_name_with_many_characters"
	x = Var[int, Ctx](long_name)
	assert latex(x) == "very_{long_variable_name_with_many_characters}"


# Docstring Examples Tests


def test_docstring_examples() -> None:
	"""Test the examples from the module docstring."""
	from typing import NamedTuple

	class TestCtx(NamedTuple):
		x: int
		y: int

	x = Var[int, TestCtx]("x")
	y = Var[int, TestCtx]("y")

	expr = x + y * 2
	assert latex(expr) == "x + y \\cdot 2"

	expr2 = x > 5
	assert latex(expr2) == "x > 5"


def test_function_docstring_examples() -> None:
	"""Test the examples from the latex function docstring."""
	from typing import NamedTuple

	class TestCtx(NamedTuple):
		x: float
		y: float

	x = Var[float, TestCtx]("x")
	y = Var[float, TestCtx]("y")

	# Test basic addition
	assert latex(x + y) == "x + y"

	# Test multiplication
	assert latex(x * y) == "x \\cdot y"

	# Test division
	assert latex(x / y) == "\\frac{x}{y}"

	# Test power
	assert latex(x**2 + y**2) == "x^2 + y^2"


# Evaluated Expression Tests


def test_latex_with_context_variables() -> None:
	"""Test LaTeX output of variables with context."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Without context
	assert latex(x) == "x"
	assert latex(y) == "y"

	# With context - shows expression -> result
	assert latex(x, LatexCtx(ctx)) == "(x:5 \\rightarrow 5)"
	assert latex(y, LatexCtx(ctx)) == "(y:10 \\rightarrow 10)"


def test_latex_with_context_arithmetic() -> None:
	"""Test LaTeX output of arithmetic expressions with context."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Addition
	add_expr = x + y
	assert latex(add_expr) == "x + y"
	assert latex(add_expr, LatexCtx(ctx)) == "(x:5 + y:10 \\rightarrow 15)"

	# Subtraction
	sub_expr = y - x
	assert latex(sub_expr) == "y - x"
	assert latex(sub_expr, LatexCtx(ctx)) == "(y:10 - x:5 \\rightarrow 5)"

	# Multiplication
	mul_expr = x * y
	assert latex(mul_expr) == "x \\cdot y"
	assert latex(mul_expr, LatexCtx(ctx)) == "(x:5 \\cdot y:10 \\rightarrow 50)"

	# Division (need floats for proper division)
	x_float = Var[float, Ctx]("f")  # f = 1.5
	y_float = Var[float, Ctx]("beta")  # beta = 3.0
	div_expr = x_float / y_float
	assert latex(div_expr) == "\\frac{f}{\\beta}"
	assert latex(div_expr, LatexCtx(ctx)) == "(\\frac{f:1.5}{\\beta:3.0} \\rightarrow 0.5)"


def test_latex_with_context_power() -> None:
	"""Test LaTeX output of power expressions with context."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Simple power
	power_expr = x**2
	assert latex(power_expr) == "x^2"
	assert latex(power_expr, LatexCtx(ctx)) == "(x:5^2 \\rightarrow 25)"

	# Variable exponent
	var_power_expr = x**y
	assert latex(var_power_expr) == "x^{y}"
	assert latex(var_power_expr, LatexCtx(ctx)) == "(x:5^{y:10} \\rightarrow 9765625)"


def test_latex_with_context_comparisons() -> None:
	"""Test LaTeX output of comparison expressions with context."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Equal
	eq_expr = x == 5
	assert latex(eq_expr) == "x = 5"
	assert latex(eq_expr, LatexCtx(ctx)) == "(x:5 = 5 \\rightarrow \\text{True})"

	# Not equal
	ne_expr = x != y
	assert latex(ne_expr) == "x \\neq y"
	assert latex(ne_expr, LatexCtx(ctx)) == "(x:5 \\neq y:10 \\rightarrow \\text{True})"

	# Less than
	lt_expr = x < y
	assert latex(lt_expr) == "x < y"
	assert latex(lt_expr, LatexCtx(ctx)) == "(x:5 < y:10 \\rightarrow \\text{True})"

	# Greater than
	gt_expr = x > y
	assert latex(gt_expr) == "x > y"
	assert latex(gt_expr, LatexCtx(ctx)) == "(x:5 > y:10 \\rightarrow \\text{False})"


def test_latex_with_context_logical() -> None:
	"""Test LaTeX output of logical expressions with context."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# AND
	and_expr = (x > 0) & (y > 0)
	assert latex(and_expr) == "x > 0 \\land y > 0"
	assert (
		latex(and_expr, LatexCtx(ctx))
		== "((x:5 > 0 \\rightarrow \\text{True}) \\land (y:10 > 0 \\rightarrow \\text{True}) \\rightarrow \\text{True})"
	)

	# OR
	or_expr = (x > 100) | (y > 0)
	assert latex(or_expr) == "x > 100 \\lor y > 0"
	assert (
		latex(or_expr, LatexCtx(ctx))
		== "((x:5 > 100 \\rightarrow \\text{False}) \\lor (y:10 > 0 \\rightarrow \\text{True}) \\rightarrow \\text{True})"
	)

	# NOT
	not_expr = ~(x > 100)
	assert latex(not_expr) == "\\neg x > 100"
	assert (
		latex(not_expr, LatexCtx(ctx))
		== "(\\neg (x:5 > 100 \\rightarrow \\text{False}) \\rightarrow \\text{True})"
	)


def test_latex_with_context_special_expressions() -> None:
	"""Test LaTeX output of special expressions with context."""
	x = Var[float, Ctx]("f")  # Using f=1.5 from ctx

	# PlusMinus (evaluates to itself)
	pm = PlusMinus("measurement", 5.0, 0.1)
	assert latex(pm) == "measurement \\pm 0.1"
	assert latex(pm, LatexCtx(ctx)) == "(measurement \\pm 0.1 \\rightarrow measurement \\pm 0.1)"

	# Percent (evaluates to itself)
	pct = Percent("tolerance", 100.0, 5.0)
	assert latex(pct) == "tolerance \\pm 5.0\\%"
	assert latex(pct, LatexCtx(ctx)) == "(tolerance \\pm 5.0\\% \\rightarrow tolerance \\pm 5.0\\%)"

	# Approximately
	target = PlusMinus("target", 1.5, 0.1)
	approx = Approximately(x, target)
	assert latex(approx) == "f \\approx target \\pm 0.1"
	assert (
		latex(approx, LatexCtx(ctx)) == "(f:1.5 \\approx target \\pm 0.1 \\rightarrow \\text{True})"
	)

	# Predicate
	pred = Predicate("condition", x > 1.0)
	assert latex(pred) == "\\text{condition}: f > 1.0"
	assert (
		latex(pred, LatexCtx(ctx))
		== "(\\text{condition}: (f:1.5 > 1.0 \\rightarrow \\text{True}) \\rightarrow \\text{True})"
	)


def test_latex_with_context_complex_expressions() -> None:
	"""Test LaTeX output of complex expressions with context."""
	x = Var[float, Ctx]("f")  # f=1.5
	y = Var[int, Ctx]("x")  # x=5
	z = Var[int, Ctx]("y")  # y=10

	# Nested arithmetic with proper parentheses handling
	complex_expr = (x + y) * z
	assert latex(complex_expr) == "(f + x) \\cdot y"
	assert (
		latex(complex_expr, LatexCtx(ctx))
		== "(((f:1.5 + x:5 \\rightarrow 6.5)) \\cdot y:10 \\rightarrow 65.0)"
	)

	# Polynomial-like expression
	poly_expr = y**2 + y + 1
	assert latex(poly_expr) == "x^2 + x + 1"
	assert (
		latex(poly_expr, LatexCtx(ctx))
		== "(((x:5^2 \\rightarrow 25) + x:5 \\rightarrow 30) + 1 \\rightarrow 31)"
	)


def test_latex_with_context_edge_cases() -> None:
	"""Test LaTeX output with context for edge cases."""
	x = Var[int, Ctx]("x")

	# Zero result
	zero_expr = x - x
	assert latex(zero_expr) == "x - x"
	assert latex(zero_expr, LatexCtx(ctx)) == "(x:5 - x:5 \\rightarrow 0)"

	# Negative values
	neg_expr = x - 10
	assert latex(neg_expr) == "x - 10"
	assert latex(neg_expr, LatexCtx(ctx)) == "(x:5 - 10 \\rightarrow -5)"

	# Float values
	x_float = Var[float, Ctx]("f")  # f=1.5
	float_expr = x_float * 2
	assert latex(float_expr) == "f \\cdot 2"
	assert latex(float_expr, LatexCtx(ctx)) == "(f:1.5 \\cdot 2 \\rightarrow 3.0)"


def test_latex_boolean_expressions_with_context() -> None:
	"""Test LaTeX output of expressions that evaluate to booleans."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")

	# Boolean expressions should show their values as \\text{True/False} when evaluated
	true_expr = x < y  # 5 < 10 is True
	false_expr = x > y  # 5 > 10 is False

	# Without context - shows structure
	assert latex(true_expr) == "x < y"
	assert latex(false_expr) == "x > y"

	# With context - shows expression -> boolean result
	assert latex(true_expr, LatexCtx(ctx)) == "(x:5 < y:10 \\rightarrow \\text{True})"
	assert latex(false_expr, LatexCtx(ctx)) == "(x:5 > y:10 \\rightarrow \\text{False})"


# Show Flags Tests


def test_show_flags_no_flags() -> None:
	"""Test LaTeX output with no Show flags."""

	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y

	# No flags - shows structure with result
	assert latex(expr, LatexCtx(ctx, Show(0))) == "(x + y \\rightarrow 15)"


def test_show_flags_values_only() -> None:
	"""Test LaTeX output with VALUES flag only."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y

	# VALUES only - shows values with final result wrapper
	assert latex(expr, LatexCtx(ctx, Show.VALUES)) == "(x:5 + y:10 \\rightarrow 15)"


def test_show_flags_work_only() -> None:
	"""Test LaTeX output with WORK flag only."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y

	# WORK only - shows structure and result
	assert latex(expr, LatexCtx(ctx, Show.WORK)) == "(x + y \\rightarrow 15)"


def test_show_flags_values_and_work() -> None:
	"""Test LaTeX output with both VALUES and WORK flags."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x + y

	# Both VALUES and WORK - shows values and result
	assert latex(expr, LatexCtx(ctx, Show.VALUES | Show.WORK)) == "(x:5 + y:10 \\rightarrow 15)"


def test_show_flags_complex_expressions() -> None:
	"""Test Show flags with complex expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = (x + y) * 2

	# Structure only - shows structure with result
	assert latex(expr, LatexCtx(ctx, Show(0))) == "((x + y) \\cdot 2 \\rightarrow 30)"

	# VALUES only
	assert latex(expr, LatexCtx(ctx, Show.VALUES)) == "((x:5 + y:10) \\cdot 2 \\rightarrow 30)"

	# WORK only
	assert (
		latex(expr, LatexCtx(ctx, Show.WORK))
		== "(((x + y \\rightarrow 15)) \\cdot 2 \\rightarrow 30)"
	)

	# Both FLAGS
	assert (
		latex(expr, LatexCtx(ctx, Show.VALUES | Show.WORK))
		== "(((x:5 + y:10 \\rightarrow 15)) \\cdot 2 \\rightarrow 30)"
	)


def test_show_flags_boolean_expressions() -> None:
	"""Test Show flags with boolean expressions."""
	x = Var[int, Ctx]("x")
	y = Var[int, Ctx]("y")
	expr = x < y

	# Structure only - shows structure with result
	assert latex(expr, LatexCtx(ctx, Show(0))) == "(x < y \\rightarrow \\text{True})"

	# VALUES only - shows values with result
	assert latex(expr, LatexCtx(ctx, Show.VALUES)) == "(x:5 < y:10 \\rightarrow \\text{True})"

	# WORK only
	assert latex(expr, LatexCtx(ctx, Show.WORK)) == "(x < y \\rightarrow \\text{True})"

	# Both FLAGS
	assert (
		latex(expr, LatexCtx(ctx, Show.VALUES | Show.WORK))
		== "(x:5 < y:10 \\rightarrow \\text{True})"
	)


def test_show_flags_with_constants() -> None:
	"""Test Show flags with named constants."""
	x = Var[int, Ctx]("x")
	Max = Const("Max", 20)
	expr = x + Max

	# Structure only shows constant name with result
	assert latex(expr, LatexCtx(ctx, Show(0))) == "(x + Max \\rightarrow 25)"

	# VALUES shows constant name:value with result
	assert latex(expr, LatexCtx(ctx, Show.VALUES)) == "(x:5 + Max:20 \\rightarrow 25)"

	# WORK shows structure and result
	assert latex(expr, LatexCtx(ctx, Show.WORK)) == "(x + Max \\rightarrow 25)"

	# Both show values and result
	assert latex(expr, LatexCtx(ctx, Show.VALUES | Show.WORK)) == "(x:5 + Max:20 \\rightarrow 25)"


# Statistical Operations Tests


def test_mean_latex() -> None:
	"""Test mean operation LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	mean_expr = Mean(values)

	# Without context - shows mathematical notation
	assert latex(mean_expr) == "\\bar{values}"

	# With context - shows values and result
	assert latex(mean_expr, LatexCtx(stats_ctx)) == "(\\bar{values:5[1.0,..5.0]} \\rightarrow 3.0)"


def test_stddev_latex() -> None:
	"""Test standard deviation LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	stddev_expr = StdDev(values)

	# Without context - shows mathematical notation
	assert latex(stddev_expr) == "\\sigma_{values}"

	# With context - shows values and result
	result_str = latex(stddev_expr, LatexCtx(stats_ctx))
	assert result_str.startswith("(\\sigma_{values:5[1.0,..5.0]} \\rightarrow")
	assert "1.58" in result_str  # Standard deviation of [1,2,3,4,5]


def test_median_latex() -> None:
	"""Test median LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	median_expr = Median(values)

	# Without context - shows mathematical notation
	assert latex(median_expr) == "\\text{median}(values)"

	# With context - shows values and result
	assert (
		latex(median_expr, LatexCtx(stats_ctx))
		== "(\\text{median}(values:5[1.0,..5.0]) \\rightarrow 3.0)"
	)


def test_percentile_latex() -> None:
	"""Test percentile LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	p95_expr = Percentile(95, values)
	p50_expr = Percentile(50.5, values)

	# Without context - shows mathematical notation
	assert latex(p95_expr) == "P_{95}(values)"
	assert latex(p50_expr) == "P_{50.5}(values)"

	# With context - shows values and result
	assert latex(p95_expr, LatexCtx(stats_ctx)) == "(P_{95}(values:5[1.0,..5.0]) \\rightarrow 4.8)"


def test_range_latex() -> None:
	"""Test range LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	range_expr = Range(values)

	# Without context - shows mathematical notation
	assert latex(range_expr) == "\\text{range}(values)"

	# With context - shows values and result
	assert (
		latex(range_expr, LatexCtx(stats_ctx))
		== "(\\text{range}(values:5[1.0,..5.0]) \\rightarrow 4.0)"
	)


def test_count_latex() -> None:
	"""Test count LaTeX formatting."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	count_expr = Count(values)

	# Without context - shows mathematical notation
	assert latex(count_expr) == "|values|"

	# With context - shows values and result
	assert latex(count_expr, LatexCtx(stats_ctx)) == "(|values:5[1.0,..5.0]| \\rightarrow 5)"


def test_statistical_operations_with_show_flags() -> None:
	"""Test statistical operations with different Show flags."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	mean_expr = Mean(values)

	# No flags - shows structure with result
	assert latex(mean_expr, LatexCtx(stats_ctx, Show(0))) == "(\\bar{values} \\rightarrow 3.0)"

	# VALUES only - shows values with result
	assert (
		latex(mean_expr, LatexCtx(stats_ctx, Show.VALUES))
		== "(\\bar{values:5[1.0,..5.0]} \\rightarrow 3.0)"
	)

	# WORK only - shows structure with result
	assert latex(mean_expr, LatexCtx(stats_ctx, Show.WORK)) == "(\\bar{values} \\rightarrow 3.0)"

	# Both flags - shows values with result
	assert (
		latex(mean_expr, LatexCtx(stats_ctx, Show.VALUES | Show.WORK))
		== "(\\bar{values:5[1.0,..5.0]} \\rightarrow 3.0)"
	)


def test_statistical_operations_with_greek_variables() -> None:
	"""Test statistical operations with Greek letter variable names."""

	# Create a context with Greek variable names
	class GreekCtx(NamedTuple):
		sigma: SizedIterable[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
		alpha: SizedIterable[float] = [1.0, 2.0, 3.0, 4.0, 5.0]

	greek_ctx = GreekCtx()
	sigma_data = Var[SizedIterable[float], GreekCtx]("sigma")
	mean_expr = Mean(sigma_data)

	# Variable name should be converted to Greek letter
	assert latex(mean_expr) == "\\bar{\\sigma}"
	assert (
		latex(mean_expr, LatexCtx(greek_ctx, Show.VALUES))
		== "(\\bar{\\sigma:5[1.0,..5.0]} \\rightarrow 3.0)"
	)


def test_complex_statistical_expressions() -> None:
	"""Test complex expressions involving statistical operations."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	mean_expr = Mean(values)
	stddev_expr = StdDev(values)

	# Statistical operations in arithmetic
	range_normalized = (mean_expr - 1) / stddev_expr

	# Without context
	assert latex(range_normalized) == "\\frac{\\bar{values} - 1}{\\sigma_{values}}"

	# With context (showing intermediate results)
	result_str = latex(range_normalized, LatexCtx(stats_ctx, Show.VALUES | Show.WORK))
	assert "\\bar{values:5[1.0,..5.0]}" in result_str
	assert "\\sigma_{values:5[1.0,..5.0]}" in result_str


def test_statistical_operations_in_comparisons() -> None:
	"""Test statistical operations used in comparisons."""
	values = Var[SizedIterable[float], StatsCtx]("values")
	mean_expr = Mean(values)
	target = 3.0

	# Comparison with statistical operation
	quality_check = mean_expr > target

	assert latex(quality_check) == "\\bar{values} > 3.0"
	assert (
		latex(quality_check, LatexCtx(stats_ctx, Show.VALUES))
		== "(\\bar{values:5[1.0,..5.0]} > 3.0 \\rightarrow \\text{False})"
	)


def test_statistical_operations_with_subscripts() -> None:
	"""Test statistical operations with subscripted variable names."""
	x_measurements = Var[SizedIterable[float], StatsCtx]("measurements")
	mean_expr = Mean(x_measurements)

	# Test with measurements variable
	assert latex(mean_expr) == "\\bar{measurements}"

	# With context using measurements data
	measurements_ctx = StatsCtx(measurements=[10.0, 15.0, 20.0, 25.0, 30.0])
	assert (
		latex(mean_expr, LatexCtx(measurements_ctx, Show.VALUES))
		== "(\\bar{measurements:5[10.0,..30.0]} \\rightarrow 20.0)"
	)
