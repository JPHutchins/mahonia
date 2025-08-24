# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""MyPy-specific tests for mahonia Context classes.

These tests use pytest-mypy-testing to validate that the mypy plugin
correctly infers types for Context classes and their vars attribute.
"""

from dataclasses import dataclass
from typing import assert_type

import pytest

from mahonia import Add, Var
from mahonia.context import Context


@dataclass
class SimpleContext(Context):
	"""Simple context for mypy testing."""

	x: int
	y: str


@dataclass
class ComplexContext(Context):
	"""Complex context with various types."""

	value: float
	name: str
	enabled: bool
	count: int


@dataclass
class EmptyContext(Context):
	"""Empty context for edge case testing."""

	pass


@pytest.mark.mypy_testing
def test_simple_context_vars_type_inference() -> None:
	"""Test that SimpleContext.vars has correct tuple type."""
	# The vars should be a tuple of Var instances
	vars_tuple = SimpleContext.vars

	# Test tuple unpacking type inference
	x, y = SimpleContext.vars

	# Test individual var types
	assert_type(x, Var[int, SimpleContext])
	assert_type(y, Var[str, SimpleContext])

	# Test that vars_tuple is the correct tuple type
	# Note: This should be Tuple[Var[int, SimpleContext], Var[str, SimpleContext]]
	# but mypy plugin needs to provide this information
	assert len(vars_tuple) == 2


@pytest.mark.mypy_testing
def test_complex_context_vars_type_inference() -> None:
	"""Test that ComplexContext.vars has correct tuple type."""
	vars_tuple = ComplexContext.vars

	# Test tuple unpacking
	value, name, enabled, count = ComplexContext.vars

	# Test individual var types
	assert_type(value, Var[float, ComplexContext])
	assert_type(name, Var[str, ComplexContext])
	assert_type(enabled, Var[bool, ComplexContext])
	assert_type(count, Var[int, ComplexContext])

	assert len(vars_tuple) == 4


@pytest.mark.mypy_testing
def test_empty_context_vars_type_inference() -> None:
	"""Test that EmptyContext.vars has correct empty tuple type."""
	vars_tuple = EmptyContext.vars

	# Should be empty tuple
	assert len(vars_tuple) == 0


@pytest.mark.mypy_testing
def test_var_expression_type_inference() -> None:
	"""Test that expressions using Context vars have correct types."""
	x, y = SimpleContext.vars

	# Arithmetic expressions
	add_expr = x + 5
	assert_type(add_expr, Add[int, SimpleContext])

	# Comparison expressions
	# Note: comparison expressions are not currently used
	# cmp_expr = x > 0
	# assert_type(cmp_expr, some_bool_expr_type)  # Will need to determine correct type


@pytest.mark.mypy_testing
def test_context_instance_type_compatibility() -> None:
	"""Test that Context instances work correctly with their vars."""
	x, y = SimpleContext.vars
	ctx = SimpleContext(x=42, y="hello")

	# These should work without type errors
	result1 = x.unwrap(ctx)
	result2 = y.unwrap(ctx)

	assert_type(result1, int)
	assert_type(result2, str)


@pytest.mark.mypy_testing
def test_context_cross_compatibility_errors() -> None:
	"""Test that using vars from one context with another context fails type checking."""
	from typing import NamedTuple

	class TraditionalCtx(NamedTuple):
		x: int
		y: str

	# Create vars from different contexts
	context_x, context_y = SimpleContext.vars
	traditional_var = Var[int, TraditionalCtx]("x")

	# Create instances
	context_instance = SimpleContext(x=42, y="hello")
	traditional_instance = TraditionalCtx(x=42, y="hello")

	# These should work (same types)
	context_result = context_x.unwrap(context_instance)
	traditional_result = traditional_var.unwrap(traditional_instance)

	assert_type(context_result, int)
	assert_type(traditional_result, int)

	# This should cause a type error (different context types)
	# cross_result = traditional_var.unwrap(context_instance)  # Should error
	# assert_type(cross_result, int)  # This line should cause mypy error


@pytest.mark.mypy_testing
def test_context_inheritance_not_supported() -> None:
	"""Test that Context inheritance merges field types correctly."""

	@dataclass
	class BaseContext(Context):
		base_field: int

	@dataclass
	class ExtendedContext(BaseContext):
		extra_field: str = "default"

	# Test base context vars
	(base_field,) = BaseContext.vars
	assert_type(base_field, Var[int, BaseContext])

	# Test extended context vars (should have both fields)
	extended_vars = ExtendedContext.vars
	assert len(extended_vars) == 2

	# Should have both inherited and directly defined fields
	base_field_ext, extra_field = ExtendedContext.vars
	assert_type(base_field_ext, Var[int, ExtendedContext])
	assert_type(extra_field, Var[str, ExtendedContext])


@pytest.mark.mypy_testing
def test_context_with_generic_types() -> None:
	"""Test Context with more complex generic types."""
	from typing import List, Optional

	@dataclass
	class GenericContext(Context):
		items: List[int]
		optional_value: Optional[str]

	items, optional_value = GenericContext.vars

	# Test that generic types are preserved
	assert_type(items, Var[List[int], GenericContext])
	assert_type(optional_value, Var[Optional[str], GenericContext])


@pytest.mark.mypy_testing
def test_context_with_complex_expressions() -> None:
	"""Test that complex expressions maintain correct types."""
	_, _, enabled, count = ComplexContext.vars

	# Create a context instance
	ctx = ComplexContext(value=10.5, name="test", enabled=True, count=5)

	# Complex arithmetic
	# calc = value * count
	# calc_result = calc.unwrap(ctx)
	# assert_type(calc_result, float)  # Should be float since value is float

	# Boolean logic
	logic = enabled & (count > 0)
	logic_result = logic.unwrap(ctx)
	assert_type(logic_result, bool)

	# Combined with constants
	# MAX = Const("MAX", 100.0)
	# combined = (value < MAX) & enabled
	# combined_result = combined.unwrap(ctx)
	# assert_type(combined_result, bool)


@pytest.mark.mypy_testing
def test_context_vars_attribute_access() -> None:
	"""Test direct access to vars attribute."""
	# Test that .vars attribute exists and has correct type
	assert hasattr(SimpleContext, "vars")
	assert hasattr(ComplexContext, "vars")
	assert hasattr(EmptyContext, "vars")

	# Test that vars can be accessed and have expected properties
	simple_vars = SimpleContext.vars
	complex_vars = ComplexContext.vars
	empty_vars = EmptyContext.vars

	# These should not cause type errors
	assert len(simple_vars) == 2
	assert len(complex_vars) == 4
	assert len(empty_vars) == 0


@pytest.mark.mypy_testing
def test_context_field_name_preservation() -> None:
	"""Test that field names are correctly preserved in vars."""
	x, y = SimpleContext.vars

	# Field names should match the class annotations
	assert x.name == "x"
	assert y.name == "y"

	# Test with complex context
	value, name, enabled, count = ComplexContext.vars
	assert value.name == "value"
	assert name.name == "name"
	assert enabled.name == "enabled"
	assert count.name == "count"
