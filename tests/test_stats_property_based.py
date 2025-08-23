# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Property-based tests for statistical operations using hypothesis.

These tests verify mathematical properties and invariants that should hold
for statistical operations, helping catch edge cases and ensure correctness.
"""

import math
import statistics
from typing import NamedTuple

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mahonia import Approximately, PlusMinus, Predicate, Var
from mahonia.stats import Count, Mean, Median, Percentile, Range, SizedIterable, StdDev


class StatsTestCtx(NamedTuple):
	measurements: list[float]
	batch_id: str


# Strategy for generating non-empty lists of floats
@st.composite
def non_empty_float_lists(draw):
	"""Generate non-empty lists of finite floats for statistical operations."""
	return draw(
		st.lists(
			st.floats(
				min_value=-1000.0,
				max_value=1000.0,
				allow_nan=False,
				allow_infinity=False,
			),
			min_size=1,
			max_size=50,
		)
	)


# Strategy for generating contexts with measurement data
@st.composite
def stats_contexts(draw):
	"""Generate test contexts with measurement data."""
	measurements = draw(non_empty_float_lists())
	batch_id = draw(st.text(min_size=1, max_size=10))
	return StatsTestCtx(measurements=measurements, batch_id=batch_id)


# Strategy for generating contexts with at least 2 measurements (needed for stddev)
@st.composite
def multi_measurement_contexts(draw):
	"""Generate contexts with at least 2 measurements for operations requiring variance."""
	measurements = draw(
		st.lists(
			st.floats(
				min_value=-1000.0,
				max_value=1000.0,
				allow_nan=False,
				allow_infinity=False,
			),
			min_size=2,
			max_size=50,
		)
	)
	batch_id = draw(st.text(min_size=1, max_size=10))
	return StatsTestCtx(measurements=measurements, batch_id=batch_id)


class TestStatisticalProperties:
	"""Test fundamental statistical properties and invariants."""

	@given(stats_contexts())
	def test_mean_bounds(self, ctx: StatsTestCtx):
		"""Mean should be between min and max values."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		mean_expr = Mean(measurements)

		result = mean_expr.unwrap(ctx)
		min_val = min(ctx.measurements)
		max_val = max(ctx.measurements)

		assert min_val <= result <= max_val

	@given(stats_contexts())
	def test_median_bounds(self, ctx: StatsTestCtx):
		"""Median should be between min and max values."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		median_expr = Median(measurements)

		result = median_expr.unwrap(ctx)
		min_val = min(ctx.measurements)
		max_val = max(ctx.measurements)

		assert min_val <= result <= max_val

	@given(multi_measurement_contexts())
	def test_stddev_non_negative(self, ctx: StatsTestCtx):
		"""Standard deviation should always be non-negative."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		stddev_expr = StdDev(measurements)

		result = stddev_expr.unwrap(ctx)
		assert result >= 0.0

	@given(stats_contexts())
	def test_range_non_negative(self, ctx: StatsTestCtx):
		"""Range should always be non-negative."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		range_expr = Range(measurements)

		result = range_expr.unwrap(ctx)
		assert result >= 0.0

	@given(stats_contexts())
	def test_count_matches_length(self, ctx: StatsTestCtx):
		"""Count should match the actual length of the iterable."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		count_expr = Count(measurements)

		result = count_expr.unwrap(ctx)
		assert result == len(ctx.measurements)

	@given(
		st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		stats_contexts(),
	)
	def test_percentile_bounds(self, percentile: float, ctx: StatsTestCtx):
		"""Percentile should be between min and max values."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		percentile_expr = Percentile(percentile, measurements)

		result = percentile_expr.unwrap(ctx)
		min_val = min(ctx.measurements)
		max_val = max(ctx.measurements)

		# Handle floating point precision issues
		tolerance = 1e-15 * max(1.0, abs(min_val), abs(max_val))
		assert min_val - tolerance <= result <= max_val + tolerance


class TestStatisticalInvariance:
	"""Test mathematical invariance properties of statistical operations."""

	@given(
		stats_contexts(),
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
	)
	def test_mean_translation_invariance(self, ctx: StatsTestCtx, offset: float):
		"""Mean(X + c) = Mean(X) + c for constant c."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Original mean
		mean_expr = Mean(measurements)
		original_mean = mean_expr.unwrap(ctx)

		# Translated measurements
		translated_measurements = [x + offset for x in ctx.measurements]
		translated_ctx = StatsTestCtx(translated_measurements, ctx.batch_id)
		translated_mean = mean_expr.unwrap(translated_ctx)

		# Should differ by exactly the offset
		assert abs(translated_mean - (original_mean + offset)) < 1e-10

	@given(
		stats_contexts(),
		st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
	)
	def test_mean_scale_invariance(self, ctx: StatsTestCtx, scale: float):
		"""Mean(c * X) = c * Mean(X) for positive constant c."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Original mean
		mean_expr = Mean(measurements)
		original_mean = mean_expr.unwrap(ctx)

		# Scaled measurements
		scaled_measurements = [x * scale for x in ctx.measurements]
		scaled_ctx = StatsTestCtx(scaled_measurements, ctx.batch_id)
		scaled_mean = mean_expr.unwrap(scaled_ctx)

		# Should be scaled by the same factor
		expected = original_mean * scale
		# Use absolute tolerance for values near zero, relative for larger values
		tolerance = max(1e-10, 1e-10 * abs(expected))
		assert abs(scaled_mean - expected) < tolerance

	@given(
		multi_measurement_contexts(),
		st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
	)
	def test_stddev_scale_invariance(self, ctx: StatsTestCtx, scale: float):
		"""StdDev(c * X) = |c| * StdDev(X) for constant c."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Original standard deviation
		stddev_expr = StdDev(measurements)
		original_stddev = stddev_expr.unwrap(ctx)

		# Skip if original stddev is very small to avoid precision issues
		assume(original_stddev > 1e-10)

		# Scaled measurements
		scaled_measurements = [x * scale for x in ctx.measurements]
		scaled_ctx = StatsTestCtx(scaled_measurements, ctx.batch_id)
		scaled_stddev = stddev_expr.unwrap(scaled_ctx)

		# Should be scaled by the absolute value of the scale factor
		expected = original_stddev * abs(scale)
		assert abs(scaled_stddev - expected) < 1e-10 * expected

	@given(
		stats_contexts(),
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
	)
	def test_median_translation_invariance(self, ctx: StatsTestCtx, offset: float):
		"""Median(X + c) = Median(X) + c for constant c."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Original median
		median_expr = Median(measurements)
		original_median = median_expr.unwrap(ctx)

		# Translated measurements
		translated_measurements = [x + offset for x in ctx.measurements]
		translated_ctx = StatsTestCtx(translated_measurements, ctx.batch_id)
		translated_median = median_expr.unwrap(translated_ctx)

		# Should differ by exactly the offset
		assert abs(translated_median - (original_median + offset)) < 1e-10


class TestConstantDataProperties:
	"""Test statistical operations on constant data."""

	@given(
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		st.integers(min_value=1, max_value=20),
	)
	def test_constant_data_mean_equals_value(self, value: float, size: int):
		"""Mean of constant data should equal the constant value."""
		measurements = [value] * size
		ctx = StatsTestCtx(measurements, "CONST")

		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")
		mean_expr = Mean(measurements_var)

		result = mean_expr.unwrap(ctx)
		assert abs(result - value) < 1e-10

	@given(
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		st.integers(min_value=1, max_value=20),
	)
	def test_constant_data_median_equals_value(self, value: float, size: int):
		"""Median of constant data should equal the constant value."""
		measurements = [value] * size
		ctx = StatsTestCtx(measurements, "CONST")

		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")
		median_expr = Median(measurements_var)

		result = median_expr.unwrap(ctx)
		assert abs(result - value) < 1e-10

	@given(
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		st.integers(min_value=2, max_value=20),  # Need at least 2 for stddev
	)
	def test_constant_data_stddev_is_zero(self, value: float, size: int):
		"""Standard deviation of constant data should be zero."""
		measurements = [value] * size
		ctx = StatsTestCtx(measurements, "CONST")

		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")
		stddev_expr = StdDev(measurements_var)

		result = stddev_expr.unwrap(ctx)
		assert abs(result) < 1e-10

	@given(
		st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
		st.integers(min_value=1, max_value=20),
	)
	def test_constant_data_range_is_zero(self, value: float, size: int):
		"""Range of constant data should be zero."""
		measurements = [value] * size
		ctx = StatsTestCtx(measurements, "CONST")

		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")
		range_expr = Range(measurements_var)

		result = range_expr.unwrap(ctx)
		assert abs(result) < 1e-10


class TestSingleElementProperties:
	"""Test statistical operations on single-element data."""

	@given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
	def test_single_element_stats_equal_value(self, value: float):
		"""For single-element data, mean, median should equal the value."""
		ctx = StatsTestCtx([value], "SINGLE")
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		mean_expr = Mean(measurements)
		median_expr = Median(measurements)
		range_expr = Range(measurements)
		count_expr = Count(measurements)

		assert abs(mean_expr.unwrap(ctx) - value) < 1e-10
		assert abs(median_expr.unwrap(ctx) - value) < 1e-10
		assert abs(range_expr.unwrap(ctx)) < 1e-10  # Range should be 0
		assert count_expr.unwrap(ctx) == 1

	@given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
	def test_single_element_percentiles(self, value: float):
		"""For single-element data, any percentile should equal the value."""
		ctx = StatsTestCtx([value], "SINGLE")
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Test various percentiles
		for percentile in [0, 25, 50, 75, 95, 100]:
			percentile_expr = Percentile(percentile, measurements)
			result = percentile_expr.unwrap(ctx)
			assert abs(result - value) < 1e-10


class TestArithmeticProperties:
	"""Test arithmetic operations with statistical functions."""

	@given(stats_contexts())
	def test_mean_arithmetic_consistency(self, ctx: StatsTestCtx):
		"""Test arithmetic operations with mean expressions."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		mean_expr = Mean(measurements)

		# Test mean + constant
		const_val = 10.0
		sum_expr = mean_expr + const_val
		result = sum_expr.unwrap(ctx)
		expected = mean_expr.unwrap(ctx) + const_val
		assert abs(result - expected) < 1e-10

		# Test mean * constant
		mult_expr = mean_expr * 2.0
		mult_result = mult_expr.unwrap(ctx)
		expected_mult = mean_expr.unwrap(ctx) * 2.0
		assert abs(mult_result - expected_mult) < 1e-10

	@given(multi_measurement_contexts())
	def test_statistical_operation_composition(self, ctx: StatsTestCtx):
		"""Test composing different statistical operations."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		mean_expr = Mean(measurements)
		stddev_expr = StdDev(measurements)

		# Coefficient of variation: stddev / mean (if mean != 0)
		mean_val = mean_expr.unwrap(ctx)
		stddev_val = stddev_expr.unwrap(ctx)
		assume(abs(mean_val) > 1e-10)  # Avoid division by near-zero
		assume(abs(stddev_val) > 1e-12 or abs(mean_val) > 1e-6)  # Avoid both being zero

		cv_expr = stddev_expr / mean_expr
		cv_result = cv_expr.unwrap(ctx)

		expected_cv = stddev_val / mean_val
		# Use absolute tolerance for small values, relative for larger values
		tolerance = max(1e-12, 1e-10 * abs(expected_cv))
		assert abs(cv_result - expected_cv) < tolerance


class TestStatisticalComparisons:
	"""Test comparison operations with statistical functions."""

	@given(stats_contexts())
	def test_mean_with_tolerance_checking(self, ctx: StatsTestCtx):
		"""Test using statistical operations with tolerance specifications."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		mean_expr = Mean(measurements)

		# Create a tolerance around the actual mean
		actual_mean = mean_expr.unwrap(ctx)
		tolerance = 1.0
		spec = PlusMinus("Spec", actual_mean, tolerance)

		# Should be approximately equal
		approx_expr = Approximately(mean_expr, spec)
		assert approx_expr.unwrap(ctx) is True

		# Test with tighter tolerance
		tight_spec = PlusMinus("Tight", actual_mean, 0.001)
		tight_approx = Approximately(mean_expr, tight_spec)
		assert tight_approx.unwrap(ctx) is True

	@given(multi_measurement_contexts())
	def test_statistical_predicate_composition(self, ctx: StatsTestCtx):
		"""Test complex predicates using multiple statistical operations."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		mean_expr = Mean(measurements)
		stddev_expr = StdDev(measurements)
		count_expr = Count(measurements)

		# Create a predicate: mean > 0 AND stddev < 1000 AND count >= 1
		predicate = Predicate(
			"Quality Check", (mean_expr > -1000.0) & (stddev_expr < 1000.0) & (count_expr >= 1)
		)

		# This should generally be true for our test data
		result = predicate.unwrap(ctx)
		assert isinstance(result, bool)


class TestStringRepresentationProperties:
	"""Test string representation properties of statistical operations."""

	@given(stats_contexts())
	def test_string_contains_operation_name(self, ctx: StatsTestCtx):
		"""String representation should contain the operation name."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		operations = [
			(Mean(measurements), "mean"),
			(Median(measurements), "median"),
			(Range(measurements), "range"),
			(Count(measurements), "count"),
		]

		# Only test StdDev if we have enough data
		if len(ctx.measurements) >= 2:
			operations.append((StdDev(measurements), "stddev"))

		for expr, op_name in operations:
			str_repr = expr.to_string()
			assert op_name in str_repr.lower()

			# With context, should show evaluation
			eval_str = expr.to_string(ctx)
			assert op_name in eval_str.lower()
			assert "->" in eval_str

	@given(
		stats_contexts(),
		st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
	)
	def test_percentile_string_contains_value(self, ctx: StatsTestCtx, percentile: float):
		"""Percentile string representation should contain the percentile value."""
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")
		percentile_expr = Percentile(percentile, measurements)

		str_repr = percentile_expr.to_string()
		assert "percentile" in str_repr.lower()
		assert str(percentile) in str_repr


class TestEdgeCases:
	"""Test edge cases and boundary conditions."""

	def test_empty_list_handling(self):
		"""Test that empty lists are handled appropriately."""
		empty_ctx = StatsTestCtx([], "EMPTY")
		measurements = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# Count should work with empty data
		count_expr = Count(measurements)
		assert count_expr.unwrap(empty_ctx) == 0

		# Statistical operations should raise appropriate errors
		mean_expr = Mean(measurements)
		with pytest.raises(statistics.StatisticsError):
			mean_expr.unwrap(empty_ctx)

	@given(
		st.lists(
			st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
			min_size=1,
			max_size=3,
		)
	)
	def test_extreme_values_handling(self, measurements: list[float]):
		"""Test handling of extreme but finite values."""
		ctx = StatsTestCtx(measurements, "EXTREME")
		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# All basic operations should work with extreme values
		mean_expr = Mean(measurements_var)
		median_expr = Median(measurements_var)
		range_expr = Range(measurements_var)
		count_expr = Count(measurements_var)

		# Should not raise exceptions
		mean_result = mean_expr.unwrap(ctx)
		median_result = median_expr.unwrap(ctx)
		range_result = range_expr.unwrap(ctx)
		count_result = count_expr.unwrap(ctx)

		# Results should be finite
		assert math.isfinite(mean_result)
		assert math.isfinite(median_result)
		assert math.isfinite(range_result)
		assert count_result == len(measurements)

	@given(non_empty_float_lists())
	@settings(suppress_health_check=[HealthCheck.filter_too_much])
	def test_ordering_invariance(self, measurements: list[float]):
		"""Test that statistical results are invariant to ordering."""
		# Only test if we have more than one unique value
		unique_vals = list(set(measurements))
		assume(len(unique_vals) > 1)

		original_ctx = StatsTestCtx(measurements, "ORIGINAL")
		shuffled_ctx = StatsTestCtx(measurements[::-1], "SHUFFLED")  # Simple reverse

		measurements_var = Var[SizedIterable[float], StatsTestCtx]("measurements")

		# These operations should be order-invariant
		mean_op = Mean(measurements_var)
		median_op = Median(measurements_var)
		range_op = Range(measurements_var)
		count_op = Count(measurements_var)

		# Test each operation
		mean_original = mean_op.unwrap(original_ctx)
		mean_shuffled = mean_op.unwrap(shuffled_ctx)
		assert abs(mean_original - mean_shuffled) < 1e-10 * max(1.0, abs(mean_original))

		median_original = median_op.unwrap(original_ctx)
		median_shuffled = median_op.unwrap(shuffled_ctx)
		assert abs(median_original - median_shuffled) < 1e-10 * max(1.0, abs(median_original))

		range_original = range_op.unwrap(original_ctx)
		range_shuffled = range_op.unwrap(shuffled_ctx)
		assert abs(range_original - range_shuffled) < 1e-10 * max(1.0, abs(range_original))

		assert count_op.unwrap(original_ctx) == count_op.unwrap(shuffled_ctx)

		# Only test StdDev if we have enough data
		if len(measurements) >= 2:
			stddev_op = StdDev(measurements_var)
			stddev_original = stddev_op.unwrap(original_ctx)
			stddev_shuffled = stddev_op.unwrap(shuffled_ctx)
			assert abs(stddev_original - stddev_shuffled) < 1e-10 * max(1.0, abs(stddev_original))
