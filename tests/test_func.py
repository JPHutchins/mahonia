# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test functional programming features with Func class."""

from typing import NamedTuple, assert_type

from mahonia import Const, Func, Var, _extract_vars


class FuncCtx(NamedTuple):
	x: int
	y: int
	z: float


def test_func_manual_construction() -> None:
	"""Test manual construction of Func objects."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = x + y
	func = Func((x, y), expr)

	# Test string representation without context
	assert func.to_string() == "(x, y) -> (x + y)"

	# Test string representation with context
	ctx = FuncCtx(x=1, y=2, z=3.14)
	assert func.to_string(ctx) == "(1, 2) -> (x + y) -> 3"


def test_to_func_automatic_extraction() -> None:
	"""Test automatic variable extraction with to_func()."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = x * y
	func = expr.to_func()

	# Should automatically extract (x, y) variables
	assert func.to_string() == "(x, y) -> (x * y)"

	# Test with context
	ctx = FuncCtx(x=3, y=4, z=1.0)
	assert func.to_string(ctx) == "(3, 4) -> (x * y) -> 12"


def test_single_variable_extraction() -> None:
	"""Test extraction with single variable."""
	x = Var[int, FuncCtx]("x")

	expr = x * 2  # x * Const(2)
	func = expr.to_func()

	# Should extract just x
	assert func.to_string() == "x -> (x * 2)"

	ctx = FuncCtx(x=5, y=0, z=0.0)
	assert func.to_string(ctx) == "5 -> (x * 2) -> 10"


def test_complex_expression_extraction() -> None:
	"""Test variable extraction from complex expressions."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")
	z = Var[float, FuncCtx]("z")

	# Complex expression: (x + y) * z
	expr = (x + y) * z
	func = expr.to_func()

	# Should extract all three variables in order
	assert func.to_string() == "(x, y, z) -> ((x + y) * z)"

	ctx = FuncCtx(x=2, y=3, z=1.5)
	expected = "(2, 3, 1.5) -> ((x + y) * z) -> 7.5"
	assert func.to_string(ctx) == expected


def test_constant_only_expression() -> None:
	"""Test expression with only constants."""
	expr = Const("answer", 42)
	func = expr.to_func()

	# No variables to extract
	assert func.to_string() == "() -> answer:42"

	ctx = FuncCtx(x=0, y=0, z=0.0)
	assert func.to_string(ctx) == "() -> answer:42 -> 42"


def test_mixed_variables_and_constants() -> None:
	"""Test extraction with mix of variables and constants."""
	x = Var[int, FuncCtx]("x")

	expr = x + Const("offset", 10)
	func = expr.to_func()

	# Should only extract variable x
	assert func.to_string() == "x -> (x + offset:10)"

	ctx = FuncCtx(x=5, y=0, z=0.0)
	assert func.to_string(ctx) == "5 -> (x + offset:10) -> 15"


def test_variable_order_preservation() -> None:
	"""Test that variable extraction preserves order of first occurrence."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")
	z = Var[float, FuncCtx]("z")

	# Use variables in different order: z, x, y
	expr = z + x + y
	func = expr.to_func()

	# Should preserve order of first occurrence: z, x, y
	assert func.to_string() == "(z, x, y) -> ((z + x) + y)"


def test_duplicate_variables() -> None:
	"""Test that duplicate variable usage doesn't create duplicates in args."""
	x = Var[int, FuncCtx]("x")

	# Use x multiple times
	expr = (x * x) + x
	func = expr.to_func()

	# Should only have x once in args
	assert func.to_string() == "x -> ((x * x) + x)"

	ctx = FuncCtx(x=3, y=0, z=0.0)
	assert func.to_string(ctx) == "3 -> ((x * x) + x) -> 12"  # 3*3 + 3 = 12


def test_nested_expressions() -> None:
	"""Test variable extraction from nested expressions."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	# Nested: (x + (y * 2))
	expr = x + (y * 2)
	func = expr.to_func()

	assert func.to_string() == "(x, y) -> (x + (y * 2))"

	ctx = FuncCtx(x=1, y=3, z=0.0)
	assert func.to_string(ctx) == "(1, 3) -> (x + (y * 2)) -> 7"  # 1 + (3*2) = 7


def test_extract_vars_helper_function() -> None:
	"""Test the _extract_vars helper function directly."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = x + y
	vars_tuple = _extract_vars((), expr)

	assert len(vars_tuple) == 2
	assert vars_tuple[0].name == "x"
	assert vars_tuple[1].name == "y"


def test_extract_vars_with_existing_vars() -> None:
	"""Test _extract_vars when some variables already exist."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")
	z = Var[float, FuncCtx]("z")

	# Start with x already in vars
	existing_vars = (x,)

	expr = x + y + z  # x is already in existing_vars
	vars_tuple = _extract_vars(existing_vars, expr)

	# Should have x, y, z (x was already there, y and z added)
	assert len(vars_tuple) == 3
	assert vars_tuple[0] is x  # Same object
	assert vars_tuple[1].name == "y"
	assert vars_tuple[2].name == "z"


def test_func_types() -> None:
	"""Test that Func maintains proper types."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = x + y
	func = Func((x, y), expr)

	# Type checking
	assert_type(func, Func[int, FuncCtx])
	assert len(func.args) == 2
	assert func.expr == expr


def test_func_with_boolean_expressions() -> None:
	"""Test Func with boolean expressions."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = x > y
	func = expr.to_func()

	assert func.to_string() == "(x, y) -> (x > y)"

	ctx = FuncCtx(x=5, y=3, z=0.0)
	assert func.to_string(ctx) == "(5, 3) -> (x > y) -> True"


def test_func_with_logical_operations() -> None:
	"""Test Func with logical operations."""
	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	expr = (x > 0) & (y < 10)
	func = expr.to_func()

	assert func.to_string() == "(x, y) -> ((x > 0) & (y < 10))"

	ctx = FuncCtx(x=5, y=3, z=0.0)
	assert func.to_string(ctx) == "(5, 3) -> ((x > 0) & (y < 10)) -> True"


def test_empty_args_func() -> None:
	"""Test Func with no arguments (constants only)."""
	expr = Const("pi", 3.14159)
	func = Func((), expr)

	assert func.to_string() == "() -> pi:3.14159"

	ctx = FuncCtx(x=0, y=0, z=0.0)
	assert func.to_string(ctx) == "() -> pi:3.14159 -> 3.14159"


def test_func_integration_with_existing_features() -> None:
	"""Test that Func works with other Mahonia features."""
	from mahonia import Predicate

	x = Var[int, FuncCtx]("x")
	y = Var[int, FuncCtx]("y")

	# Create a predicate and convert to function
	expr = (x + y) > 10
	predicate = Predicate("Sum greater than 10", expr)
	func = expr.to_func()

	assert func.to_string() == "(x, y) -> ((x + y) > 10)"

	ctx_true = FuncCtx(x=7, y=8, z=0.0)
	ctx_false = FuncCtx(x=2, y=3, z=0.0)

	assert "True" in func.to_string(ctx_true)
	assert "False" in func.to_string(ctx_false)

	# Verify predicate still works
	assert predicate.unwrap(ctx_true) is True
	assert predicate.unwrap(ctx_false) is False
