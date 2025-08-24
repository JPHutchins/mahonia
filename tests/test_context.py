# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for the Context class and automatic vars generation."""

from dataclasses import dataclass

import pytest

from mahonia import Const, Var
from mahonia.context import Context


@dataclass
class SimpleContext(Context):
	"""Simple context for testing basic functionality."""

	x: int
	y: str


@dataclass
class ComplexContext(Context):
	"""More complex context with various types."""

	value: float
	name: str
	enabled: bool
	count: int


def test_context_vars_generation():
	"""Test that Context classes automatically generate vars attribute."""
	# Check that vars exists and is a NamedTuple
	assert hasattr(SimpleContext, "vars")
	assert hasattr(ComplexContext, "vars")

	# Check the vars structure for SimpleContext
	assert len(SimpleContext.vars) == 2
	x, y = SimpleContext.vars
	assert isinstance(x, Var)
	assert isinstance(y, Var)
	assert x.name == "x"
	assert y.name == "y"

	# Check the vars structure for ComplexContext
	assert len(ComplexContext.vars) == 4
	value, name, enabled, count = ComplexContext.vars
	assert isinstance(value, Var)
	assert isinstance(name, Var)
	assert isinstance(enabled, Var)
	assert isinstance(count, Var)
	assert value.name == "value"
	assert name.name == "name"
	assert enabled.name == "enabled"
	assert count.name == "count"


def test_context_vars_unpacking():
	"""Test that vars can be unpacked into individual variables."""
	x, y = SimpleContext.vars

	# Test that the unpacked vars work correctly
	ctx = SimpleContext(x=42, y="hello")

	assert x.unwrap(ctx) == 42
	assert y.unwrap(ctx) == "hello"

	# Test string representation
	assert x.to_string() == "x"
	assert y.to_string() == "y"
	assert x.to_string(ctx) == "x:42"
	assert y.to_string(ctx) == "y:hello"


def test_context_integration_with_expressions():
	"""Test that Context-generated vars integrate with mahonia expressions."""
	x, y = SimpleContext.vars
	ctx = SimpleContext(x=10, y="test")

	# Test arithmetic expressions
	expr1 = x + 5
	assert expr1.unwrap(ctx) == 15
	assert expr1.to_string(ctx) == "(x:10 + 5 -> 15)"

	# Test comparison expressions
	expr2 = x > 5
	assert expr2.unwrap(ctx)
	assert expr2.to_string(ctx) == "(x:10 > 5 -> True)"

	# Test complex expressions
	MAX = Const("MAX", 100)
	expr3 = (x > 0) & (x < MAX)
	assert expr3.unwrap(ctx)
	expected = "((x:10 > 0 -> True) & (x:10 < MAX:100 -> True) -> True)"
	assert expr3.to_string(ctx) == expected


def test_context_dataclass_behavior():
	"""Test that Context classes behave like dataclasses."""
	ctx = SimpleContext(x=42, y="hello")

	# Test attribute access
	assert ctx.x == 42
	assert ctx.y == "hello"

	# Test mutability (since we're not using frozen=True)
	ctx.x = 100
	assert ctx.x == 100


def test_context_with_various_types():
	"""Test Context with different Python types."""
	value, name, enabled, count = ComplexContext.vars
	ctx = ComplexContext(value=3.14, name="test", enabled=True, count=5)

	assert value.unwrap(ctx) == 3.14
	assert name.unwrap(ctx) == "test"
	assert enabled.unwrap(ctx)
	assert count.unwrap(ctx) == 5

	# Test type coercion in expressions
	expr = value > 3.0
	assert expr.unwrap(ctx)

	bool_expr = enabled & (count > 0)
	assert bool_expr.unwrap(ctx)


def test_empty_context():
	"""Test Context class with no fields."""

	@dataclass
	class EmptyContext(Context):
		pass

	# Should have empty vars
	assert len(EmptyContext.vars) == 0

	# Should still be creatable
	ctx = EmptyContext()
	assert isinstance(ctx, EmptyContext)


def test_context_inheritance_works():
	"""Test that Context inheritance merges fields correctly."""

	@dataclass
	class BaseContext(Context):
		base_field: int

	@dataclass
	class ExtendedContext(BaseContext):
		extra_field: str = "default"

	# Base should only have its own field
	(base_field,) = BaseContext.vars
	assert base_field.name == "base_field"

	# Extended should have both inherited and own fields
	assert len(ExtendedContext.vars) == 2
	base_field_ext, extra_field = ExtendedContext.vars
	assert base_field_ext.name == "base_field"
	assert extra_field.name == "extra_field"


@pytest.mark.mypy_testing
def test_context_type_annotations():
	"""Test that Context preserves type information for mypy."""
	# This test will be enhanced when the mypy plugin is complete
	x, y = SimpleContext.vars

	# Basic runtime type checking - mypy plugin will enhance this
	assert isinstance(x, Var)
	assert isinstance(y, Var)
	assert x.name == "x"
	assert y.name == "y"


def test_context_vs_namedtuple_compatibility():
	"""Test that Context works alongside traditional NamedTuple contexts."""
	from typing import NamedTuple

	# Traditional approach should still work
	class TraditionalCtx(NamedTuple):
		x: int
		y: str

	# Manual var creation (old way)
	old_x = Var[int, TraditionalCtx]("x")
	old_y = Var[str, TraditionalCtx]("y")

	# New Context approach
	new_x, new_y = SimpleContext.vars

	# Both should work with their respective context types
	old_ctx = TraditionalCtx(x=42, y="hello")
	new_ctx = SimpleContext(x=42, y="hello")

	assert old_x.unwrap(old_ctx) == 42
	assert old_y.unwrap(old_ctx) == "hello"

	assert new_x.unwrap(new_ctx) == 42
	assert new_y.unwrap(new_ctx) == "hello"

	# Cross-usage with different field names should fail
	old_z = Var[int, TraditionalCtx]("z")  # Field that doesn't exist in TraditionalCtx
	from mahonia import EvalError

	with pytest.raises(EvalError):
		old_z.unwrap(old_ctx)  # Should fail since TraditionalCtx has no 'z' field
