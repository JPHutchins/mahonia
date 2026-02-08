# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

import math
from typing import NamedTuple, SupportsFloat, SupportsIndex, assert_type

import pytest

from mahonia import Const, Failure, Var
from mahonia.python_func import (
	PythonFunc0,
	PythonFunc0Wrapper,
	PythonFunc1,
	PythonFunc1Wrapper,
	PythonFunc2,
	PythonFunc2Wrapper,
	ResultAdd,  # pyright: ignore[reportUnusedImport]
	python_func,
)


class Ctx(NamedTuple):
	x: float
	y: float


x = Var[float, Ctx]("x")
y = Var[float, Ctx]("y")


def div(a: float, b: float) -> float:
	return a / b


def get_pi() -> float:
	return math.pi


def raise_error() -> float:
	raise ValueError("test error")


def my_sqrt(val: float) -> float:
	return math.sqrt(val)


safe_sqrt = python_func(my_sqrt)
safe_div = python_func(div)
get_pi_wrapped = python_func(get_pi)
get_error_wrapped = python_func(raise_error)


class TestPythonFunc0:
	def test_success(self) -> None:
		expr: PythonFunc0[float, Ctx] = get_pi_wrapped()
		assert expr.unwrap(Ctx(x=0, y=0)) == math.pi

	def test_failure(self) -> None:
		expr: PythonFunc0[float, Ctx] = get_error_wrapped()
		result = expr.unwrap(Ctx(x=0, y=0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1
		assert str(result.exceptions[0]) == "test error"

	def test_to_string(self) -> None:
		expr: PythonFunc0[float, Ctx] = get_pi_wrapped()
		assert expr.to_string() == "get_pi()"
		assert expr.to_string(Ctx(x=0, y=0)) == f"get_pi() -> {math.pi}"


class TestPythonFunc1:
	def test_success_with_expr_arg(self) -> None:
		expr = safe_sqrt(x)
		assert expr.unwrap(Ctx(x=4.0, y=0)) == 2.0

	def test_success_with_literal_arg(self) -> None:
		expr: PythonFunc1[float, float, Ctx] = safe_sqrt(9.0)
		assert expr.unwrap(Ctx(x=0, y=0)) == 3.0

	def test_failure(self) -> None:
		expr = safe_sqrt(x)
		result = expr.unwrap(Ctx(x=-1.0, y=0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_failure_propagation_from_arg(self) -> None:
		inner = safe_sqrt(x)
		outer = safe_sqrt(inner)
		result = outer.unwrap(Ctx(x=-1.0, y=0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_to_string(self) -> None:
		expr = safe_sqrt(x)
		assert expr.to_string() == "my_sqrt(x)"
		assert expr.to_string(Ctx(x=4.0, y=0)) == "my_sqrt(x:4.0) -> 2.0"


class TestPythonFunc2:
	def test_success_with_expr_args(self) -> None:
		expr = safe_div(x, y)
		assert expr.unwrap(Ctx(x=10.0, y=2.0)) == 5.0

	def test_success_with_literal_args(self) -> None:
		expr: PythonFunc2[float, float, float, Ctx] = safe_div(10.0, 2.0)
		assert expr.unwrap(Ctx(x=0, y=0)) == 5.0

	def test_success_with_mixed_args(self) -> None:
		expr = safe_div(x, 2.0)
		assert expr.unwrap(Ctx(x=10.0, y=0)) == 5.0

	def test_failure(self) -> None:
		expr = safe_div(x, y)
		result = expr.unwrap(Ctx(x=10.0, y=0.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_error_accumulation(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		result = expr.unwrap(Ctx(x=-1.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 2

	def test_to_string(self) -> None:
		expr = safe_div(x, y)
		assert expr.to_string() == "div(x, y)"
		assert expr.to_string(Ctx(x=10.0, y=2.0)) == "div(x:10.0, y:2.0) -> 5.0"


class TestResultOperators:
	def test_result_add(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == 5.0

	def test_result_add_with_literal(self) -> None:
		expr = safe_sqrt(x) + 1.0
		assert expr.unwrap(Ctx(x=4.0, y=0)) == 3.0

	def test_result_radd_with_literal(self) -> None:
		expr = 1.0 + safe_sqrt(x)
		assert expr.unwrap(Ctx(x=4.0, y=0)) == 3.0

	def test_result_sub(self) -> None:
		expr = safe_sqrt(x) - safe_sqrt(y)
		assert expr.unwrap(Ctx(x=16.0, y=9.0)) == 1.0

	def test_result_mul(self) -> None:
		expr = safe_sqrt(x) * safe_sqrt(y)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == 6.0

	def test_result_div(self) -> None:
		expr = safe_sqrt(x) / safe_sqrt(y)
		assert expr.unwrap(Ctx(x=16.0, y=4.0)) == 2.0

	def test_error_accumulation_through_operators(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		result = expr.unwrap(Ctx(x=-1.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 2

	def test_single_failure_propagation(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		result = expr.unwrap(Ctx(x=-1.0, y=4.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_chained_operations(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) * 2.0
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == 10.0


class TestPartial:
	def test_partial_pythonfunc1(self) -> None:
		expr = safe_sqrt(x)
		partial = expr.partial(Ctx(x=4.0, y=0))
		assert partial.to_string() == "my_sqrt(x:4.0)"

	def test_partial_pythonfunc2(self) -> None:
		expr = safe_div(x, y)
		partial = expr.partial(Ctx(x=10.0, y=2.0))
		assert partial.to_string() == "div(x:10.0, y:2.0)"


class TestFactoryFunction:
	def test_factory_returns_correct_wrapper_arity_0(self) -> None:
		wrapper = python_func(get_pi)
		assert isinstance(wrapper, PythonFunc0Wrapper)

	def test_factory_returns_correct_wrapper_arity_1(self) -> None:
		def identity(val: int) -> int:
			return val

		wrapper = python_func(identity)
		assert isinstance(wrapper, PythonFunc1Wrapper)

	def test_factory_returns_correct_wrapper_arity_2(self) -> None:
		def add(a: int, b: int) -> int:
			return a + b

		wrapper = python_func(add)
		assert isinstance(wrapper, PythonFunc2Wrapper)

	def test_factory_raises_for_unsupported_arity(self) -> None:
		def three_args(a: int, b: int, c: int) -> int:
			return a + b + c

		with pytest.raises(ValueError, match="python_func supports 0-2 args, got 3"):
			python_func(three_args)  # type: ignore[arg-type]


class TestIntegration:
	def test_complex_expression(self) -> None:
		expr = safe_div(safe_sqrt(x) + safe_sqrt(y), 2.0)
		assert expr.unwrap(Ctx(x=4.0, y=16.0)) == 3.0

	def test_pure_expressions_unchanged(self) -> None:
		expr = x + y
		assert expr.unwrap(Ctx(x=1.0, y=2.0)) == 3.0
		assert expr.to_string() == "(x + y)"

	def test_mixing_pure_and_ffi(self) -> None:
		expr = safe_sqrt(x + y)
		assert expr.unwrap(Ctx(x=3.0, y=1.0)) == 2.0

	def test_to_string_shows_failure(self) -> None:
		expr = safe_sqrt(x)
		result_str = expr.to_string(Ctx(x=-1.0, y=0))
		assert "Failure" in result_str


@pytest.mark.mypy_testing
def test_python_func_generic_types() -> None:
	assert_type(safe_sqrt, PythonFunc1Wrapper[float, float])
	assert_type(safe_div, PythonFunc2Wrapper[float, float, float])
	assert_type(get_pi_wrapped, PythonFunc0Wrapper[float])

	sqrt_expr = safe_sqrt(x)
	assert_type(sqrt_expr, PythonFunc1[float, float, Ctx])

	div_expr = safe_div(x, y)
	assert_type(div_expr, PythonFunc2[float, float, float, Ctx])

	pi_expr: PythonFunc0[float, Ctx] = get_pi_wrapped()
	assert_type(pi_expr, PythonFunc0[float, Ctx])

	assert_type(sqrt_expr.unwrap(Ctx(x=4.0, y=0)), float | Failure)
	assert_type(div_expr.unwrap(Ctx(x=10.0, y=2.0)), float | Failure)
	assert_type(pi_expr.unwrap(Ctx(x=0, y=0)), float | Failure)

	add_expr = safe_sqrt(x) + safe_sqrt(y)
	assert_type(add_expr, ResultAdd[float, Ctx])
	assert_type(add_expr.unwrap(Ctx(x=4.0, y=9.0)), float | Failure)


@pytest.mark.mypy_testing
def test_math_sqrt_wrapper_types() -> None:
	stdlib_sqrt = python_func(math.sqrt)
	assert_type(stdlib_sqrt, PythonFunc1Wrapper[SupportsFloat | SupportsIndex, float])

	expr = stdlib_sqrt(x)
	assert_type(expr, PythonFunc1[SupportsFloat | SupportsIndex, float, Ctx])


@pytest.mark.mypy_testing
def test_ffi_pollution_coerces_to_result_types() -> None:
	from mahonia import Add
	from mahonia.python_func import (
		ResultAdd,  # pyright: ignore[reportUnusedImport]
		ResultDiv,
		ResultEq,
		ResultGe,
		ResultGt,
		ResultLe,
		ResultLt,
		ResultMul,
		ResultNe,
		ResultSub,
	)

	pure_add = x + y
	assert_type(pure_add, Add[float, Ctx])

	ffi_left_add = safe_sqrt(x) + y
	assert_type(ffi_left_add, ResultAdd[float, Ctx])

	ffi_left_sub = safe_sqrt(x) - y
	assert_type(ffi_left_sub, ResultSub[float, Ctx])

	ffi_left_mul = safe_sqrt(x) * y
	assert_type(ffi_left_mul, ResultMul[float, Ctx])

	ffi_left_div = safe_sqrt(x) / y
	assert_type(ffi_left_div, ResultDiv[float, Ctx])

	literal_radd = 1.0 + safe_sqrt(x)
	assert_type(literal_radd, ResultAdd[float, Ctx])

	literal_rsub = 1.0 - safe_sqrt(x)
	assert_type(literal_rsub, ResultSub[float, Ctx])

	literal_rmul = 2.0 * safe_sqrt(x)
	assert_type(literal_rmul, ResultMul[float, Ctx])

	literal_rdiv = 1.0 / safe_sqrt(x)
	assert_type(literal_rdiv, ResultDiv[float, Ctx])

	ffi_plus_pure = safe_sqrt(x) + (x + y)
	assert_type(ffi_plus_pure, ResultAdd[float, Ctx])

	ffi_lt = safe_sqrt(x) < 5.0
	assert_type(ffi_lt, ResultLt[float, Ctx])

	ffi_le = safe_sqrt(x) <= 5.0
	assert_type(ffi_le, ResultLe[float, Ctx])

	ffi_gt = safe_sqrt(x) > 1.0
	assert_type(ffi_gt, ResultGt[float, Ctx])

	ffi_ge = safe_sqrt(x) >= 1.0
	assert_type(ffi_ge, ResultGe[float, Ctx])

	ffi_eq = safe_sqrt(x) == 2.0
	assert_type(ffi_eq, ResultEq[float, Ctx])

	ffi_ne = safe_sqrt(x) != 3.0
	assert_type(ffi_ne, ResultNe[float, Ctx])

	ffi_chain = (safe_sqrt(x) + safe_sqrt(y)) * 2.0
	assert_type(ffi_chain, ResultMul[float, Ctx])

	ffi_cmp_chain = (safe_sqrt(x) > 1.0) & (safe_sqrt(y) < 10.0)
	from mahonia import And

	assert_type(ffi_cmp_chain, And[bool, Ctx])

	pure_plus_ffi_add = (x + y) + safe_sqrt(x)
	assert_type(pure_plus_ffi_add, ResultAdd[float, Ctx])

	pure_plus_ffi_sub = (x + y) - safe_sqrt(x)
	assert_type(pure_plus_ffi_sub, ResultSub[float, Ctx])

	pure_plus_ffi_mul = (x + y) * safe_sqrt(x)
	assert_type(pure_plus_ffi_mul, ResultMul[float, Ctx])

	pure_plus_ffi_div = (x + y) / safe_sqrt(x)
	assert_type(pure_plus_ffi_div, ResultDiv[float, Ctx])

	var_plus_ffi = x + safe_sqrt(y)
	assert_type(var_plus_ffi, ResultAdd[float, Ctx])

	var_minus_ffi = x - safe_sqrt(y)
	assert_type(var_minus_ffi, ResultSub[float, Ctx])

	var_times_ffi = x * safe_sqrt(y)
	assert_type(var_times_ffi, ResultMul[float, Ctx])

	var_div_ffi = x / safe_sqrt(y)
	assert_type(var_div_ffi, ResultDiv[float, Ctx])


class TestCompositionNestedFFI:
	def test_composition_ffi_nested_arithmetic(self) -> None:
		inner = safe_sqrt(x) + safe_sqrt(y)
		outer = inner * 2.0
		final = outer - 1.0

		assert final.unwrap(Ctx(x=4.0, y=9.0)) == (2.0 + 3.0) * 2.0 - 1.0
		assert final.to_string() == "(((my_sqrt(x) + my_sqrt(y)) * 2.0) - 1.0)"
		assert (
			final.to_string(Ctx(x=4.0, y=9.0))
			== "(((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0) * 2.0 -> 10.0) - 1.0 -> 9.0)"
		)

	def test_composition_deep_nesting(self) -> None:
		level1 = safe_sqrt(x)
		level2 = safe_sqrt(level1)
		level3 = safe_sqrt(level2)

		assert level3.unwrap(Ctx(x=256.0, y=0)) == 2.0
		assert level3.to_string() == "my_sqrt(my_sqrt(my_sqrt(x)))"
		assert (
			level3.to_string(Ctx(x=256.0, y=0))
			== "my_sqrt(my_sqrt(my_sqrt(x:256.0) -> 16.0) -> 4.0) -> 2.0"
		)

	def test_composition_ffi_inside_pure_inside_ffi(self) -> None:
		inner_sum = x + y
		dividend = safe_sqrt(inner_sum)
		divisor = safe_sqrt(y)
		expr = safe_div(dividend, divisor)

		assert expr.unwrap(Ctx(x=7.0, y=9.0)) == 4.0 / 3.0
		assert expr.to_string() == "div(my_sqrt((x + y)), my_sqrt(y))"

	def test_composition_mixed_operations(self) -> None:
		mixed = (safe_sqrt(x) + y) * safe_sqrt(y)

		assert mixed.unwrap(Ctx(x=4.0, y=9.0)) == (2.0 + 9.0) * 3.0
		assert mixed.to_string() == "((my_sqrt(x) + y) * my_sqrt(y))"

	def test_composition_four_level_chain(self) -> None:
		expr = ((safe_sqrt(x) + safe_sqrt(y)) * 2.0 - 3.0) / 2.0

		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == ((2.0 + 3.0) * 2.0 - 3.0) / 2.0
		assert expr.to_string() == "((((my_sqrt(x) + my_sqrt(y)) * 2.0) - 3.0) / 2.0)"


class TestToStringEvaluated:
	def test_result_add_evaluated(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		assert expr.to_string() == "(my_sqrt(x) + my_sqrt(y))"
		assert (
			expr.to_string(Ctx(x=4.0, y=9.0))
			== "(my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)"
		)

	def test_result_sub_evaluated(self) -> None:
		expr = safe_sqrt(x) - safe_sqrt(y)
		assert expr.to_string() == "(my_sqrt(x) - my_sqrt(y))"
		assert (
			expr.to_string(Ctx(x=16.0, y=4.0))
			== "(my_sqrt(x:16.0) -> 4.0 - my_sqrt(y:4.0) -> 2.0 -> 2.0)"
		)

	def test_result_mul_evaluated(self) -> None:
		expr = safe_sqrt(x) * safe_sqrt(y)
		assert expr.to_string() == "(my_sqrt(x) * my_sqrt(y))"
		assert (
			expr.to_string(Ctx(x=4.0, y=9.0))
			== "(my_sqrt(x:4.0) -> 2.0 * my_sqrt(y:9.0) -> 3.0 -> 6.0)"
		)

	def test_result_div_evaluated(self) -> None:
		expr = safe_sqrt(x) / safe_sqrt(y)
		assert expr.to_string() == "(my_sqrt(x) / my_sqrt(y))"
		assert (
			expr.to_string(Ctx(x=16.0, y=4.0))
			== "(my_sqrt(x:16.0) -> 4.0 / my_sqrt(y:4.0) -> 2.0 -> 2.0)"
		)

	def test_chained_operators_evaluated(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) * 2.0 - 1.0
		ctx = Ctx(x=4.0, y=9.0)
		result_str = expr.to_string(ctx)
		assert "my_sqrt(x:4.0) -> 2.0" in result_str
		assert "my_sqrt(y:9.0) -> 3.0" in result_str
		assert "-> 9.0" in result_str

	def test_pythonfunc2_evaluated(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		assert expr.to_string() == "div(my_sqrt(x), my_sqrt(y))"
		assert (
			expr.to_string(Ctx(x=16.0, y=4.0))
			== "div(my_sqrt(x:16.0) -> 4.0, my_sqrt(y:4.0) -> 2.0) -> 2.0"
		)


class TestFailurePropagationComprehensive:
	def test_failure_in_nested_shows_in_string(self) -> None:
		expr = safe_sqrt(safe_sqrt(x))
		result_str = expr.to_string(Ctx(x=-1.0, y=0))
		assert "Failure" in result_str

	def test_multiple_failures_accumulated_three_terms(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) + safe_sqrt(x)
		result = expr.unwrap(Ctx(x=-1.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 3

	def test_partial_failure_right_branch(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		result = expr.unwrap(Ctx(x=4.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_failure_at_depth_3(self) -> None:
		level1 = safe_sqrt(x)
		level2 = safe_sqrt(level1)
		level3 = safe_sqrt(level2)
		result = level3.unwrap(Ctx(x=-1.0, y=0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_failure_stops_outer_function(self) -> None:
		call_count = [0]

		def counting_sqrt(val: float) -> float:
			call_count[0] += 1
			return math.sqrt(val)

		counting = python_func(counting_sqrt)
		inner = counting(x)
		outer = counting(inner)

		outer.unwrap(Ctx(x=-1.0, y=0))
		assert call_count[0] == 1

	def test_pythonfunc2_both_args_fail(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		result = expr.unwrap(Ctx(x=-1.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 2

	def test_pythonfunc2_first_arg_fails(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		result = expr.unwrap(Ctx(x=-1.0, y=4.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_pythonfunc2_second_arg_fails(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		result = expr.unwrap(Ctx(x=4.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_failure_in_operator_chain(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y) - safe_sqrt(x)
		result = expr.unwrap(Ctx(x=-1.0, y=4.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 2


class TestBindAndEval:
	def test_bind_pythonfunc1(self) -> None:
		expr = safe_sqrt(x)
		bound = expr.bind(Ctx(x=4.0, y=0))
		assert bound.unwrap() == 2.0
		assert "2.0" in str(bound)

	def test_bind_pythonfunc2(self) -> None:
		expr = safe_div(x, y)
		bound = expr.bind(Ctx(x=10.0, y=2.0))
		assert bound.unwrap() == 5.0

	def test_bind_result_operator(self) -> None:
		expr = safe_sqrt(x) + safe_sqrt(y)
		bound = expr.bind(Ctx(x=4.0, y=9.0))
		assert bound.unwrap() == 5.0

	def test_bind_with_failure(self) -> None:
		bound = safe_sqrt(x).bind(Ctx(x=-1.0, y=0))
		result = bound.unwrap()
		assert isinstance(result, Failure)

	def test_eval_returns_const(self) -> None:
		result = safe_sqrt(x).eval(Ctx(x=4.0, y=0))
		assert isinstance(result, Const)
		assert result.value == 2.0

	def test_eval_failure_wrapped_in_const(self) -> None:
		result = safe_sqrt(x).eval(Ctx(x=-1.0, y=0))
		assert isinstance(result, Const)
		assert isinstance(result.value, Failure)

	def test_callable_syntax_equivalent_to_eval(self) -> None:
		expr = safe_sqrt(x)
		ctx = Ctx(x=4.0, y=0)
		assert expr(ctx).value == expr.eval(ctx).value


class TestBooleanComparison:
	def test_ffi_greater_than_literal(self) -> None:
		expr = safe_sqrt(x) > 1.5
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True
		assert expr.to_string() == "(my_sqrt(x) > 1.5)"

	def test_ffi_less_than_literal(self) -> None:
		expr = safe_sqrt(x) < 5.0
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True
		assert expr.to_string() == "(my_sqrt(x) < 5.0)"

	def test_ffi_equality(self) -> None:
		expr = safe_sqrt(x) == 2.0
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True

	def test_ffi_inequality(self) -> None:
		expr = safe_sqrt(x) != 3.0
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True

	def test_ffi_and_ffi(self) -> None:
		valid = (safe_sqrt(x) > 0) & (safe_sqrt(y) > 0)
		assert valid.unwrap(Ctx(x=4.0, y=9.0)) is True
		assert valid.to_string() == "((my_sqrt(x) > 0) & (my_sqrt(y) > 0))"

	def test_ffi_or_ffi(self) -> None:
		either = (safe_sqrt(x) > 10) | (safe_sqrt(y) > 2)
		assert either.unwrap(Ctx(x=4.0, y=9.0)) is True

	def test_ffi_range_check(self) -> None:
		in_range = (safe_sqrt(x) > 1.0) & (safe_sqrt(x) < 5.0)
		assert in_range.unwrap(Ctx(x=4.0, y=0)) is True

	def test_not_ffi_comparison(self) -> None:
		expr = ~(safe_sqrt(x) > 5.0)
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True


class TestRealisticScenarios:
	def test_distance_formula(self) -> None:
		dx_squared = (x - 0) ** 2
		dy_squared = (y - 0) ** 2
		distance = safe_sqrt(dx_squared + dy_squared)

		assert distance.unwrap(Ctx(x=3.0, y=4.0)) == 5.0
		assert "my_sqrt" in distance.to_string()
		assert distance.to_string() == "my_sqrt((((x - 0)^2) + ((y - 0)^2)))"

	def test_quadratic_discriminant(self) -> None:
		discriminant = Const(None, 5.0**2 - 4 * 1.0 * 6.0)
		result = safe_sqrt(discriminant)
		assert result.unwrap(Ctx(x=0, y=0)) == 1.0

	def test_quadratic_discriminant_complex_roots(self) -> None:
		discriminant = Const(None, 1.0**2 - 4 * 1.0 * 1.0)
		result = safe_sqrt(discriminant)
		failure = result.unwrap(Ctx(x=0, y=0))
		assert isinstance(failure, Failure)

	def test_normalized_value_positive(self) -> None:
		magnitude = safe_sqrt(x * x)
		normalized = safe_div(x, magnitude)
		assert normalized.unwrap(Ctx(x=5.0, y=0)) == 1.0

	def test_normalized_value_negative(self) -> None:
		magnitude = safe_sqrt(x * x)
		normalized = safe_div(x, magnitude)
		assert normalized.unwrap(Ctx(x=-5.0, y=0)) == -1.0

	def test_geometric_mean(self) -> None:
		product = x * y
		geo_mean = safe_sqrt(product)

		assert geo_mean.unwrap(Ctx(x=4.0, y=9.0)) == 6.0

	def test_geometric_mean_negative_product_fails(self) -> None:
		product = x * y
		geo_mean = safe_sqrt(product)
		result = geo_mean.unwrap(Ctx(x=-4.0, y=9.0))
		assert isinstance(result, Failure)

	def test_harmonic_mean(self) -> None:
		reciprocal_sum = safe_div(1.0, x) + safe_div(1.0, y)
		harmonic = safe_div(2.0, reciprocal_sum)
		assert harmonic.unwrap(Ctx(x=4.0, y=4.0)) == 4.0


class TestLambdaFunctions:
	def test_lambda_to_string_shows_lambda(self) -> None:
		double: PythonFunc1Wrapper[float, float] = python_func(lambda v: v * 2)
		expr: PythonFunc1[float, float, Ctx] = double(x)
		assert expr.to_string() == "lambda(x)"

	def test_lambda_execution(self) -> None:
		double: PythonFunc1Wrapper[float, float] = python_func(lambda v: v * 2)
		assert double(x).unwrap(Ctx(x=5.0, y=0)) == 10.0

	def test_lambda_binary(self) -> None:
		multiply: PythonFunc2Wrapper[float, float, float] = python_func(lambda a, b: a * b)
		expr: PythonFunc2[float, float, float, Ctx] = multiply(x, y)
		assert expr.unwrap(Ctx(x=3.0, y=4.0)) == 12.0
		assert expr.to_string() == "lambda(x, y)"

	def test_lambda_nullary(self) -> None:
		constant = python_func(lambda: 42)
		expr: PythonFunc0[int, Ctx] = constant()
		assert expr.unwrap(Ctx(x=0, y=0)) == 42
		assert expr.to_string() == "lambda()"


class TestEdgeCases:
	def test_function_returning_zero(self) -> None:
		zero = python_func(lambda: 0)
		expr: PythonFunc0[int, Ctx] = zero()
		assert expr.unwrap(Ctx(x=0, y=0)) == 0

	def test_function_returning_zero_float(self) -> None:
		expr = safe_sqrt(x)
		assert expr.unwrap(Ctx(x=0.0, y=0)) == 0.0

	def test_pythonfunc0_combined_with_pythonfunc1(self) -> None:
		pi: PythonFunc0[float, Ctx] = get_pi_wrapped()
		expr = pi + safe_sqrt(x)
		assert expr.unwrap(Ctx(x=4.0, y=0)) == pytest.approx(math.pi + 2.0)  # pyright: ignore[reportUnknownMemberType]
		assert expr.to_string() == "(get_pi() + my_sqrt(x))"

	def test_reverse_sub(self) -> None:
		rsub = 10.0 - safe_sqrt(x)
		assert rsub.unwrap(Ctx(x=4.0, y=0)) == 8.0
		assert rsub.to_string() == "(10.0 - my_sqrt(x))"

	def test_reverse_mul(self) -> None:
		rmul = 3.0 * safe_sqrt(x)
		assert rmul.unwrap(Ctx(x=4.0, y=0)) == 6.0
		assert rmul.to_string() == "(3.0 * my_sqrt(x))"

	def test_reverse_div(self) -> None:
		rdiv = 8.0 / safe_sqrt(x)
		assert rdiv.unwrap(Ctx(x=4.0, y=0)) == 4.0
		assert rdiv.to_string() == "(8.0 / my_sqrt(x))"

	def test_reverse_operators_with_failure(self) -> None:
		expr = 10.0 - safe_sqrt(x)
		result = expr.unwrap(Ctx(x=-1.0, y=0))
		assert isinstance(result, Failure)

	def test_very_deep_nesting(self) -> None:
		expr = safe_sqrt(x)
		for _ in range(3):
			expr = safe_sqrt(expr)
		assert expr.unwrap(Ctx(x=65536.0, y=0)) == 2.0

	def test_complex_tree_multiple_paths(self) -> None:
		left = safe_sqrt(x) + safe_sqrt(y)
		right = safe_sqrt(x) * safe_sqrt(y)
		expr = safe_div(left, right)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == (2.0 + 3.0) / (2.0 * 3.0)


SQRT_FAIL = "Failure(exceptions=(ValueError('expected a nonnegative input, got {val}'),))"


class XOnly(NamedTuple):
	x: float


class TestDeepNestedMixedSerialization:
	def test_ffi_inside_pure_inside_ffi_evaluated(self) -> None:
		expr = safe_div(safe_sqrt(x + y), safe_sqrt(y))
		assert expr.to_string() == "div(my_sqrt((x + y)), my_sqrt(y))"
		assert expr.to_string(Ctx(x=7.0, y=9.0)) == (
			"div(my_sqrt((x:7.0 + y:9.0 -> 16.0)) -> 4.0,"
			" my_sqrt(y:9.0) -> 3.0) -> 1.3333333333333333"
		)
		assert expr.unwrap(Ctx(x=7.0, y=9.0)) == 4.0 / 3.0

	def test_mixed_operations_evaluated(self) -> None:
		expr = (safe_sqrt(x) + y) * safe_sqrt(y)
		assert expr.to_string() == "((my_sqrt(x) + y) * my_sqrt(y))"
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"((my_sqrt(x:4.0) -> 2.0 + y:9.0 -> 11.0) * my_sqrt(y:9.0) -> 3.0 -> 33.0)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == 33.0

	def test_four_level_chain_evaluated(self) -> None:
		expr = ((safe_sqrt(x) + safe_sqrt(y)) * 2.0 - 3.0) / 2.0
		assert expr.to_string() == "((((my_sqrt(x) + my_sqrt(y)) * 2.0) - 3.0) / 2.0)"
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"((((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)"
			" * 2.0 -> 10.0) - 3.0 -> 7.0) / 2.0 -> 3.5)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == 3.5

	def test_ffi2_wrapping_result_ops(self) -> None:
		expr = safe_div(safe_sqrt(x + y) * 2.0, safe_sqrt(x * y) - 1.0)
		assert expr.to_string() == ("div((my_sqrt((x + y)) * 2.0), (my_sqrt((x * y)) - 1.0))")
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"div((my_sqrt((x:4.0 + y:9.0 -> 13.0)) -> 3.605551275463989"
			" * 2.0 -> 7.211102550927978),"
			" (my_sqrt((x:4.0 * y:9.0 -> 36.0)) -> 6.0 - 1.0 -> 5.0))"
			" -> 1.4422205101855956"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == math.sqrt(13) * 2.0 / (math.sqrt(36) - 1.0)

	def test_all_three_ffi_arities_mixed(self) -> None:
		expr = (safe_sqrt(x) + get_pi_wrapped()) * safe_div(x, y)
		assert expr.to_string() == "((my_sqrt(x) + get_pi()) * div(x, y))"
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"((my_sqrt(x:4.0) -> 2.0 + get_pi() -> 3.141592653589793"
			" -> 5.141592653589793) * div(x:4.0, y:9.0)"
			" -> 0.4444444444444444 -> 2.285152290484352)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == (2.0 + math.pi) * (4.0 / 9.0)

	def test_named_const_with_ffi_deep(self) -> None:
		expr = safe_div(safe_sqrt(x) * Const("Scale", 10.0) + Const("Offset", 3.0), y)
		assert expr.to_string() == ("div(((my_sqrt(x) * Scale:10.0) + Offset:3.0), y)")
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"div(((my_sqrt(x:4.0) -> 2.0 * Scale:10.0 -> 20.0)"
			" + Offset:3.0 -> 23.0), y:9.0) -> 2.5555555555555554"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) == (2.0 * 10.0 + 3.0) / 9.0


class TestBoundExprSerializationComplete:
	def test_bound_pythonfunc1_exact(self) -> None:
		bound = safe_sqrt(x).bind(Ctx(x=4.0, y=0))
		assert bound.to_string() == "my_sqrt(x:4.0) -> 2.0"
		assert str(bound) == "my_sqrt(x:4.0) -> 2.0"
		assert bound.unwrap() == 2.0

	def test_bound_pythonfunc2_exact(self) -> None:
		bound = safe_div(x, y).bind(Ctx(x=10.0, y=2.0))
		assert bound.to_string() == "div(x:10.0, y:2.0) -> 5.0"
		assert bound.unwrap() == 5.0

	def test_bound_result_add_exact(self) -> None:
		bound = (safe_sqrt(x) + safe_sqrt(y)).bind(Ctx(x=4.0, y=9.0))
		assert bound.to_string() == ("(my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)")
		assert bound.unwrap() == 5.0

	def test_bound_deep_nested_exact(self) -> None:
		bound = safe_div(safe_sqrt(x) + safe_sqrt(y), 2.0).bind(Ctx(x=4.0, y=16.0))
		assert bound.to_string() == (
			"div((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:16.0) -> 4.0 -> 6.0), 2.0) -> 3.0"
		)
		assert bound.unwrap() == 3.0

	def test_bound_four_level(self) -> None:
		bound = ((safe_sqrt(x) + safe_sqrt(y)) * safe_div(x, y)).bind(Ctx(x=4.0, y=9.0))
		assert bound.to_string() == (
			"((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)"
			" * div(x:4.0, y:9.0) -> 0.4444444444444444"
			" -> 2.2222222222222223)"
		)
		assert bound.unwrap() == (2.0 + 3.0) * (4.0 / 9.0)

	def test_bound_composed_with_var(self) -> None:
		bound_sqrt = safe_sqrt(x).bind(Ctx(x=4.0, y=0))
		composed = bound_sqrt + y
		assert composed.to_string() == "(my_sqrt(x:4.0) -> 2.0 + y)"
		assert composed.to_string(Ctx(x=0, y=9.0)) == ("(my_sqrt(x:4.0) -> 2.0 + y:9.0 -> 11.0)")
		assert composed.unwrap(Ctx(x=0, y=9.0)) == 11.0

	def test_bound_in_ffi_arg(self) -> None:
		bound_sqrt = safe_sqrt(x).bind(Ctx(x=4.0, y=0))
		expr = safe_div(bound_sqrt, y)
		assert expr.to_string() == "div(my_sqrt(x:4.0) -> 2.0, y)"
		assert expr.to_string(Ctx(x=0, y=2.0)) == ("div(my_sqrt(x:4.0) -> 2.0, y:2.0) -> 1.0")
		assert expr.unwrap(Ctx(x=0, y=2.0)) == 1.0

	def test_bound_failure_exact(self) -> None:
		bound = safe_sqrt(x).bind(Ctx(x=-1.0, y=0))
		assert bound.to_string() == ("my_sqrt(x:-1.0) -> " + SQRT_FAIL.format(val=-1.0))
		assert isinstance(bound.unwrap(), Failure)


class TestPartialSerializationDeep:
	def test_partial_ffi_inside_pure_resolved(self) -> None:
		expr = safe_div(safe_sqrt(x + y), safe_sqrt(x * y))
		partial = expr.partial(Ctx(x=4.0, y=9.0))
		assert partial.to_string() == ("div(my_sqrt((x:4.0 + y:9.0)), my_sqrt((x:4.0 * y:9.0)))")

	def test_partial_one_var_resolved(self) -> None:
		expr = safe_div(safe_sqrt(x + y), safe_sqrt(x))
		partial = expr.partial(XOnly(x=4.0))
		assert partial.to_string() == ("div(my_sqrt((x:4.0 + y)), my_sqrt(x:4.0))")

	def test_partial_result_op_deep(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) * 2.0 - 1.0
		partial = expr.partial(Ctx(x=4.0, y=9.0))
		assert partial.to_string() == ("(((my_sqrt(x:4.0) + my_sqrt(y:9.0)) * 2.0) - 1.0)")

	def test_partial_pythonfunc0_unchanged(self) -> None:
		pi: PythonFunc0[float, Ctx] = get_pi_wrapped()
		expr = pi + safe_sqrt(x)
		partial = expr.partial(Ctx(x=4.0, y=0))
		assert partial.to_string() == "(get_pi() + my_sqrt(x:4.0))"


class TestFailureSerializationExact:
	def test_failure_single_exact(self) -> None:
		expr = safe_sqrt(x)
		assert expr.to_string(Ctx(x=-1.0, y=0)) == (
			"my_sqrt(x:-1.0) -> " + SQRT_FAIL.format(val=-1.0)
		)
		result = expr.unwrap(Ctx(x=-1.0, y=0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 1

	def test_failure_nested_propagation_exact(self) -> None:
		expr = safe_sqrt(safe_sqrt(x))
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string(Ctx(x=-1.0, y=0)) == (f"my_sqrt(my_sqrt(x:-1.0) -> {fail}) -> {fail}")

	def test_failure_accumulated_exact(self) -> None:
		expr = safe_div(safe_sqrt(x), safe_sqrt(y))
		single = SQRT_FAIL.format(val=-1.0)
		double = (
			"Failure(exceptions=("
			"ValueError('expected a nonnegative input, got -1.0'), "
			"ValueError('expected a nonnegative input, got -1.0')))"
		)
		assert expr.to_string(Ctx(x=-1.0, y=-1.0)) == (
			f"div(my_sqrt(x:-1.0) -> {single}, my_sqrt(y:-1.0) -> {single}) -> {double}"
		)
		result = expr.unwrap(Ctx(x=-1.0, y=-1.0))
		assert isinstance(result, Failure)
		assert len(result.exceptions) == 2

	def test_failure_deep_mixed_exact(self) -> None:
		expr = safe_div(safe_sqrt(x + y), safe_sqrt(x))
		fail_4 = SQRT_FAIL.format(val=-4.0)
		fail_1 = SQRT_FAIL.format(val=-1.0)
		double = (
			"Failure(exceptions=("
			"ValueError('expected a nonnegative input, got -4.0'), "
			"ValueError('expected a nonnegative input, got -1.0')))"
		)
		assert expr.to_string(Ctx(x=-1.0, y=-3.0)) == (
			f"div(my_sqrt((x:-1.0 + y:-3.0 -> -4.0)) -> {fail_4},"
			f" my_sqrt(x:-1.0) -> {fail_1}) -> {double}"
		)

	def test_failure_divzero_exact(self) -> None:
		expr = safe_div(safe_sqrt(x), y)
		assert expr.to_string(Ctx(x=4.0, y=0.0)) == (
			"div(my_sqrt(x:4.0) -> 2.0, y:0.0)"
			" -> Failure(exceptions=(ZeroDivisionError('division by zero'),))"
		)

	def test_bound_failure_deep_exact(self) -> None:
		bound = safe_div(safe_sqrt(x + y), safe_sqrt(x)).bind(Ctx(x=-1.0, y=-3.0))
		fail_4 = SQRT_FAIL.format(val=-4.0)
		fail_1 = SQRT_FAIL.format(val=-1.0)
		double = (
			"Failure(exceptions=("
			"ValueError('expected a nonnegative input, got -4.0'), "
			"ValueError('expected a nonnegative input, got -1.0')))"
		)
		assert bound.to_string() == (
			f"div(my_sqrt((x:-1.0 + y:-3.0 -> -4.0)) -> {fail_4},"
			f" my_sqrt(x:-1.0) -> {fail_1}) -> {double}"
		)
		assert isinstance(bound.unwrap(), Failure)


class TestFailureTrickleDown:
	def test_failure_trickles_through_addition(self) -> None:
		expr = safe_sqrt(x) + 10.0
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string() == "(my_sqrt(x) + 10.0)"
		assert expr.to_string(Ctx(x=-1.0, y=0)) == (f"(my_sqrt(x:-1.0) -> {fail} + 10.0 -> {fail})")
		assert isinstance(expr.unwrap(Ctx(x=-1.0, y=0)), Failure)

	def test_failure_trickles_through_multiplication(self) -> None:
		expr = safe_sqrt(x) * y
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string() == "(my_sqrt(x) * y)"
		assert expr.to_string(Ctx(x=-1.0, y=5.0)) == (
			f"(my_sqrt(x:-1.0) -> {fail} * y:5.0 -> {fail})"
		)
		assert isinstance(expr.unwrap(Ctx(x=-1.0, y=5.0)), Failure)

	def test_failure_trickles_through_ffi2(self) -> None:
		expr = safe_div(safe_sqrt(x) + 1.0, y)
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string() == "div((my_sqrt(x) + 1.0), y)"
		assert expr.to_string(Ctx(x=-1.0, y=5.0)) == (
			f"div((my_sqrt(x:-1.0) -> {fail} + 1.0 -> {fail}), y:5.0) -> {fail}"
		)
		assert isinstance(expr.unwrap(Ctx(x=-1.0, y=5.0)), Failure)

	def test_failure_trickles_through_chain(self) -> None:
		expr = (safe_sqrt(x) + 1.0) * 2.0 - 3.0
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string() == "(((my_sqrt(x) + 1.0) * 2.0) - 3.0)"
		assert expr.to_string(Ctx(x=-1.0, y=0)) == (
			f"(((my_sqrt(x:-1.0) -> {fail} + 1.0 -> {fail}) * 2.0 -> {fail}) - 3.0 -> {fail})"
		)
		assert isinstance(expr.unwrap(Ctx(x=-1.0, y=0)), Failure)

	def test_failure_one_branch_trickles_through_mul(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) * safe_div(x, y)
		fail = SQRT_FAIL.format(val=-1.0)
		assert expr.to_string() == ("((my_sqrt(x) + my_sqrt(y)) * div(x, y))")
		assert expr.to_string(Ctx(x=-1.0, y=9.0)) == (
			f"((my_sqrt(x:-1.0) -> {fail}"
			" + my_sqrt(y:9.0) -> 3.0"
			f" -> {fail})"
			" * div(x:-1.0, y:9.0) -> -0.1111111111111111"
			f" -> {fail})"
		)
		assert isinstance(expr.unwrap(Ctx(x=-1.0, y=9.0)), Failure)


class TestComparisonFFISerialization:
	def test_gt_evaluated_exact(self) -> None:
		expr = safe_sqrt(x) > 1.5
		assert expr.to_string() == "(my_sqrt(x) > 1.5)"
		assert expr.to_string(Ctx(x=4.0, y=0)) == ("(my_sqrt(x:4.0) -> 2.0 > 1.5 -> True)")
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True

	def test_and_range_evaluated_exact(self) -> None:
		expr = (safe_sqrt(x) > 1.0) & (safe_sqrt(x) < 5.0)
		assert expr.to_string() == ("((my_sqrt(x) > 1.0) & (my_sqrt(x) < 5.0))")
		assert expr.to_string(Ctx(x=4.0, y=0)) == (
			"((my_sqrt(x:4.0) -> 2.0 > 1.0 -> True)"
			" & (my_sqrt(x:4.0) -> 2.0 < 5.0 -> True) -> True)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True

	def test_not_ffi_comparison_evaluated_exact(self) -> None:
		expr = ~(safe_sqrt(x) > 5.0)
		assert expr.to_string() == "(not (my_sqrt(x) > 5.0))"
		assert expr.to_string(Ctx(x=4.0, y=0)) == (
			"(not (my_sqrt(x:4.0) -> 2.0 > 5.0 -> False) -> True)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=0)) is True

	def test_deep_ffi_comparison_evaluated(self) -> None:
		expr = (safe_sqrt(x) + safe_sqrt(y)) > safe_div(x, y)
		assert expr.to_string() == ("((my_sqrt(x) + my_sqrt(y)) > div(x, y))")
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)"
			" > div(x:4.0, y:9.0) -> 0.4444444444444444 -> True)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) is True

	def test_and_deep_ffi_comparisons(self) -> None:
		expr = ((safe_sqrt(x) + safe_sqrt(y)) > 0) & (safe_div(x, y) < 1.0)
		assert expr.to_string() == ("(((my_sqrt(x) + my_sqrt(y)) > 0) & (div(x, y) < 1.0))")
		assert expr.to_string(Ctx(x=4.0, y=9.0)) == (
			"(((my_sqrt(x:4.0) -> 2.0 + my_sqrt(y:9.0) -> 3.0 -> 5.0)"
			" > 0 -> True)"
			" & (div(x:4.0, y:9.0) -> 0.4444444444444444"
			" < 1.0 -> True) -> True)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=9.0)) is True


class TestLiteralCoercionSerialization:
	def test_literal_in_ffi_arg(self) -> None:
		expr: PythonFunc1[float, float, Ctx] = safe_sqrt(9.0)
		assert expr.to_string() == "my_sqrt(9.0)"
		assert expr.to_string(Ctx(x=0, y=0)) == "my_sqrt(9.0) -> 3.0"
		assert expr.unwrap(Ctx(x=0, y=0)) == 3.0

	def test_literal_in_result_op(self) -> None:
		expr = safe_sqrt(x) + 1.0
		assert expr.to_string() == "(my_sqrt(x) + 1.0)"
		assert expr.to_string(Ctx(x=4.0, y=0)) == ("(my_sqrt(x:4.0) -> 2.0 + 1.0 -> 3.0)")

	def test_reverse_literal_in_result_op(self) -> None:
		expr = 10.0 - safe_sqrt(x)
		assert expr.to_string() == "(10.0 - my_sqrt(x))"
		assert expr.to_string(Ctx(x=4.0, y=0)) == ("(10.0 - my_sqrt(x:4.0) -> 2.0 -> 8.0)")

	def test_named_const_in_ffi_arg(self) -> None:
		expr = safe_div(safe_sqrt(x), Const("TWO", 2.0))
		assert expr.to_string() == "div(my_sqrt(x), TWO:2.0)"
		assert expr.to_string(Ctx(x=16.0, y=0)) == ("div(my_sqrt(x:16.0) -> 4.0, TWO:2.0) -> 2.0")
		assert expr.unwrap(Ctx(x=16.0, y=0)) == 2.0

	def test_mixed_literal_named_const_var_ffi(self) -> None:
		expr = (safe_sqrt(x) * Const("Scale", 10.0) + 3.0) / y
		assert expr.to_string() == ("(((my_sqrt(x) * Scale:10.0) + 3.0) / y)")
		assert expr.to_string(Ctx(x=4.0, y=5.0)) == (
			"(((my_sqrt(x:4.0) -> 2.0 * Scale:10.0 -> 20.0) + 3.0 -> 23.0) / y:5.0 -> 4.6)"
		)
		assert expr.unwrap(Ctx(x=4.0, y=5.0)) == 4.6
