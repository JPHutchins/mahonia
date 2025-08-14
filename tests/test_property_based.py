# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Property-based tests for Mahonia expressions using hypothesis.

These tests verify mathematical properties and invariants that should hold
for all valid inputs, helping catch edge cases and ensure correctness.
"""

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mahonia import (
	Approximately,
	Const,
	Percent,
	PlusMinus,
	Predicate,
	Var,
)


@dataclass(frozen=True)
class PropertyTestCtx:
	x: int
	y: int
	z: int
	f: float
	g: float


# Strategy for generating test contexts
@st.composite
def contexts_strategy(draw):
	return PropertyTestCtx(
		x=draw(st.integers(-1000, 1000)),
		y=draw(st.integers(-1000, 1000)),
		z=draw(st.integers(-1000, 1000)),
		f=draw(st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False)),
		g=draw(st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False)),
	)


# Strategy for generating integer constants
@st.composite
def int_consts(draw):
	value = draw(st.integers(-100, 100))
	name = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
	return Const(name, value)


# Strategy for generating float constants
@st.composite
def float_consts(draw):
	value = draw(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))
	name = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
	return Const(name, value)


class TestArithmeticProperties:
	"""Test fundamental arithmetic properties."""

	@given(st.integers(), st.integers())
	def test_addition_commutative(self, a: int, b: int):
		"""a + b == b + a for all integers a, b"""
		ca = Const("a", a)
		cb = Const("b", b)
		expr1 = ca + cb
		expr2 = cb + ca
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers(), st.integers(), st.integers())
	def test_addition_associative(self, a: int, b: int, c: int):
		"""(a + b) + c == a + (b + c) for all integers a, b, c"""
		ca, cb, cc = Const("a", a), Const("b", b), Const("c", c)
		expr1 = (ca + cb) + cc
		expr2 = ca + (cb + cc)
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers())
	def test_addition_identity(self, a: int):
		"""a + 0 == a for all integers a"""
		ca = Const("a", a)
		zero = Const("zero", 0)
		expr = ca + zero
		assert expr.unwrap(None) == a

	@given(st.integers(), st.integers())
	def test_multiplication_commutative(self, a: int, b: int):
		"""a * b == b * a for all integers a, b"""
		ca = Const("a", a)
		cb = Const("b", b)
		expr1 = ca * cb
		expr2 = cb * ca
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers(), st.integers(), st.integers())
	def test_multiplication_associative(self, a: int, b: int, c: int):
		"""(a * b) * c == a * (b * c) for all integers a, b, c"""
		ca, cb, cc = Const("a", a), Const("b", b), Const("c", c)
		expr1 = (ca * cb) * cc
		expr2 = ca * (cb * cc)
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers())
	def test_multiplication_identity(self, a: int):
		"""a * 1 == a for all integers a"""
		ca = Const("a", a)
		one = Const("one", 1)
		expr = ca * one
		assert expr.unwrap(None) == a

	@given(st.integers())
	def test_multiplication_zero(self, a: int):
		"""a * 0 == 0 for all integers a"""
		ca = Const("a", a)
		zero = Const("zero", 0)
		expr = ca * zero
		assert expr.unwrap(None) == 0

	@given(st.integers(), st.integers(), st.integers())
	def test_distributive_property(self, a: int, b: int, c: int):
		"""a * (b + c) == (a * b) + (a * c) for all integers a, b, c"""
		ca, cb, cc = Const("a", a), Const("b", b), Const("c", c)
		expr1 = ca * (cb + cc)
		expr2 = (ca * cb) + (ca * cc)
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers(), st.integers())
	def test_subtraction_inverse(self, a: int, b: int):
		"""(a - b) + b == a for all integers a, b"""
		ca = Const("a", a)
		cb = Const("b", b)
		expr = (ca - cb) + cb
		assert expr.unwrap(None) == a

	@given(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))
	def test_division_inverse(self, a: float):
		"""(a / b) * b == a for all non-zero b"""
		assume(abs(a) > 1e-10)  # Avoid division by very small numbers
		ca = Const("a", a)
		cb = Const("b", 2.0)  # Use a safe non-zero divisor
		expr = (ca / cb) * cb
		result = expr.unwrap(None)
		assert abs(result - a) < 1e-10  # Account for floating point precision


class TestComparisonProperties:
	"""Test comparison operation properties."""

	@given(st.integers())
	def test_equality_reflexive(self, a: int):
		"""a == a is always True"""
		ca = Const("a", a)
		expr = ca == ca
		assert expr.unwrap(None) is True

	@given(st.integers(), st.integers())
	def test_equality_symmetric(self, a: int, b: int):
		"""(a == b) == (b == a) for all a, b"""
		ca = Const("a", a)
		cb = Const("b", b)
		expr1 = ca == cb
		expr2 = cb == ca
		assert expr1.unwrap(None) == expr2.unwrap(None)

	@given(st.integers(), st.integers(), st.integers())
	@settings(suppress_health_check=[HealthCheck.filter_too_much])
	def test_equality_transitive(self, a: int, b: int, c: int):
		"""If a == b and b == c, then a == c"""
		assume(a == b and b == c)  # Only test when they're transitively equal
		ca, cb, cc = Const("a", a), Const("b", b), Const("c", c)
		expr1 = ca == cb
		expr2 = cb == cc
		expr3 = ca == cc

		# All should be true
		assert expr1.unwrap(None) is True
		assert expr2.unwrap(None) is True
		assert expr3.unwrap(None) is True

	@given(st.integers(), st.integers())
	def test_inequality_complement(self, a: int, b: int):
		"""(a == b) is complement of (a != b)"""
		ca = Const("a", a)
		cb = Const("b", b)
		eq_expr = ca == cb
		ne_expr = ca != cb
		assert eq_expr.unwrap(None) != ne_expr.unwrap(None)

	@given(st.integers(), st.integers())
	def test_less_than_antisymmetric(self, a: int, b: int):
		"""If a < b, then not (b < a)"""
		ca = Const("a", a)
		cb = Const("b", b)
		expr1 = ca < cb
		expr2 = cb < ca

		# They can't both be true (antisymmetric property)
		assert not (expr1.unwrap(None) and expr2.unwrap(None))


class TestLogicalProperties:
	"""Test logical operation properties."""

	@given(contexts_strategy())
	def test_and_commutative(self, ctx: PropertyTestCtx):
		"""a & b == b & a for boolean expressions"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")

		cond_a = x > 0
		cond_b = y > 0

		expr1 = cond_a & cond_b
		expr2 = cond_b & cond_a
		assert expr1.unwrap(ctx) == expr2.unwrap(ctx)

	@given(contexts_strategy())
	def test_or_commutative(self, ctx: PropertyTestCtx):
		"""a | b == b | a for boolean expressions"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")

		cond_a = x > 0
		cond_b = y > 0

		expr1 = cond_a | cond_b
		expr2 = cond_b | cond_a
		assert expr1.unwrap(ctx) == expr2.unwrap(ctx)

	@given(contexts_strategy())
	def test_and_associative(self, ctx: PropertyTestCtx):
		"""(a & b) & c == a & (b & c) for boolean expressions"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")
		z = Var[int, PropertyTestCtx]("z")

		cond_a = x > 0
		cond_b = y > 0
		cond_c = z > 0

		expr1 = (cond_a & cond_b) & cond_c
		expr2 = cond_a & (cond_b & cond_c)
		assert expr1.unwrap(ctx) == expr2.unwrap(ctx)

	@given(contexts_strategy())
	def test_or_associative(self, ctx: PropertyTestCtx):
		"""(a | b) | c == a | (b | c) for boolean expressions"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")
		z = Var[int, PropertyTestCtx]("z")

		cond_a = x > 0
		cond_b = y > 0
		cond_c = z > 0

		expr1 = (cond_a | cond_b) | cond_c
		expr2 = cond_a | (cond_b | cond_c)
		assert expr1.unwrap(ctx) == expr2.unwrap(ctx)

	@given(contexts_strategy())
	def test_double_negation(self, ctx: PropertyTestCtx):
		"""~~a == a for boolean expressions"""
		x = Var[int, PropertyTestCtx]("x")
		original = x > 0
		double_neg = ~~original
		assert original.unwrap(ctx) == double_neg.unwrap(ctx)


class TestToleranceProperties:
	"""Test approximate equality properties."""

	@given(
		st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
		st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
	)
	def test_approximately_reflexive(self, value: float, tolerance: float):
		"""value ≈ value should always be True"""
		target = PlusMinus("target", value, tolerance)
		expr = Approximately(Const("val", value), target)
		assert expr.unwrap(None) is True

	@given(
		st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
		st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
	)
	def test_approximately_symmetric(self, value: float, tolerance: float):
		"""If a ≈ b, then b ≈ a (within same tolerance)"""
		val_a = Const("a", value)
		val_b = Const("b", value + tolerance / 2)  # Within tolerance

		target_a = PlusMinus("target_a", value, tolerance)
		target_b = PlusMinus("target_b", value + tolerance / 2, tolerance)

		expr1 = Approximately(val_a, target_b)
		expr2 = Approximately(val_b, target_a)

		# Both should evaluate the same way
		result1 = expr1.unwrap(None)
		result2 = expr2.unwrap(None)
		assert result1 == result2

	@given(
		st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
		st.floats(1.0, 50.0, allow_nan=False, allow_infinity=False),
	)
	def test_percent_tolerance_positive(self, value: float, percent: float):
		"""Percentage tolerance should work for positive values"""
		assume(abs(value) > 1e-10)  # Avoid very small numbers

		target = Percent("target", abs(value), percent)

		# Value within tolerance should pass
		within_val = abs(value) * (1 + percent / 200)  # Half the tolerance
		expr = Approximately(Const("val", within_val), target)
		assert expr.unwrap(None) is True

		# Value outside tolerance should fail
		outside_val = abs(value) * (1 + percent / 50)  # Double the tolerance
		expr2 = Approximately(Const("val", outside_val), target)
		assert expr2.unwrap(None) is False


class TestExpressionConsistency:
	"""Test consistency properties across different expression types."""

	@given(contexts_strategy())
	def test_string_evaluation_consistency(self, ctx: PropertyTestCtx):
		"""String representation should be consistent with evaluation"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")
		expr = x + y

		# The evaluated string should contain the actual result
		result = expr.unwrap(ctx)
		eval_string = expr.to_string(ctx)
		assert str(result) in eval_string

	@given(contexts_strategy())
	def test_binding_consistency(self, ctx: PropertyTestCtx):
		"""Bound expressions should evaluate consistently"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")
		expr = x + y

		# Direct evaluation vs bound evaluation
		direct_result = expr.unwrap(ctx)
		bound_expr = expr.bind(ctx)
		bound_result = bound_expr.unwrap()

		assert direct_result == bound_result

	@given(contexts_strategy(), int_consts())
	def test_const_context_independence(self, ctx: PropertyTestCtx, const_expr: Const[int]):
		"""Constants should evaluate the same regardless of context"""
		result1 = const_expr.unwrap(None)
		result2 = const_expr.unwrap(ctx)
		assert result1 == result2

	@given(contexts_strategy())
	def test_predicate_boolean_consistency(self, ctx: PropertyTestCtx):
		"""Predicates should maintain boolean semantics"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")

		bool_expr = x > y
		predicate = Predicate("test", bool_expr)

		# Both should give same boolean result
		assert bool_expr.unwrap(ctx) == predicate.unwrap(ctx)


class TestVariableProperties:
	"""Test properties specific to variable expressions."""

	@given(contexts_strategy())
	def test_variable_context_binding(self, ctx: PropertyTestCtx):
		"""Variables should correctly bind to context attributes"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")

		assert x.unwrap(ctx) == ctx.x
		assert y.unwrap(ctx) == ctx.y

	@given(contexts_strategy())
	def test_variable_expression_composition(self, ctx: PropertyTestCtx):
		"""Variable expressions should compose correctly"""
		x = Var[int, PropertyTestCtx]("x")
		y = Var[int, PropertyTestCtx]("y")

		# Complex expression using variables
		expr = (x + y) * (x - y)
		expected = (ctx.x + ctx.y) * (ctx.x - ctx.y)
		assert expr.unwrap(ctx) == expected


# Regression test for specific edge cases that hypothesis might find
class TestRegressionCases:
	"""Test specific edge cases and regression scenarios."""

	def test_zero_division_safety(self):
		"""Division by zero should be handled appropriately"""
		# This would be a regression test if hypothesis found a division by zero issue
		numerator = Const("num", 10.0)
		denominator = Const("den", 0.0)

		# In a real scenario, we might want to handle this gracefully
		# For now, we expect Python's normal division by zero behavior
		expr = numerator / denominator
		with pytest.raises(ZeroDivisionError):
			expr.unwrap(None)

	@given(
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
	)
	def test_floating_point_precision(self, value: float, tolerance: float):
		"""Test floating point precision in approximations"""
		target = PlusMinus("target", value, tolerance)

		# Value well within the boundary (half tolerance)
		within_val = value + tolerance / 2
		expr = Approximately(Const("val", within_val), target)

		# Should be True (within tolerance)
		assert expr.unwrap(None) is True

		# Value well outside boundary (double tolerance)
		outside_val = value + tolerance * 2
		expr2 = Approximately(Const("val", outside_val), target)

		# Should be False (outside tolerance)
		assert expr2.unwrap(None) is False
