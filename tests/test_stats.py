# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test statistical operations."""

import statistics
from typing import NamedTuple, assert_type

import pytest

from mahonia import Approximately, Const, PlusMinus, Predicate, Var
from mahonia.stats import Count, Mean, Median, Percentile, Range, SizedIterable, StdDev


class BatchData(NamedTuple):
	measurements: list[float]
	part_count: int
	batch_id: str


def test_mean() -> None:
	"""Test mean calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	mean_expr = Mean(measurements)
	mean = mean_expr.unwrap(ctx)
	assert_type(mean, float)
	assert mean_expr.unwrap(ctx) == 3.0
	assert mean_expr.to_string() == "mean(measurements)"

	# Test string with context
	result_str = mean_expr.to_string(ctx)
	assert "mean(" in result_str
	assert "-> 3.0" in result_str


def test_stddev() -> None:
	"""Test standard deviation calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	stddev_expr = StdDev(measurements)
	result = stddev_expr.unwrap(ctx)
	assert abs(result - 1.5811388300841898) < 0.0001  # Known stddev for this sequence
	assert stddev_expr.to_string() == "stddev(measurements)"


def test_median() -> None:
	"""Test median calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	median_expr = Median(measurements)
	assert median_expr.unwrap(ctx) == 3.0
	assert median_expr.to_string() == "median(measurements)"


def test_percentile() -> None:
	"""Test percentile calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	p95_expr = Percentile(95, measurements)
	result = p95_expr.unwrap(ctx)
	assert result == 4.8  # 95th percentile of [1,2,3,4,5]
	assert p95_expr.to_string() == "percentile:95(measurements)"


def test_range() -> None:
	"""Test range calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	range_expr = Range(measurements)
	assert range_expr.unwrap(ctx) == 4.0  # 5.0 - 1.0
	assert range_expr.to_string() == "range(measurements)"


def test_count() -> None:
	"""Test count calculation."""
	ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="B123")
	measurements_list = Var[SizedIterable[float], BatchData]("measurements")

	count_expr = Count(measurements_list)
	assert count_expr.unwrap(ctx) == 5
	assert count_expr.to_string() == "count(measurements)"


def test_mean_with_tolerance() -> None:
	"""Test using mean with tolerance checking."""
	ctx = BatchData(measurements=[4.95, 5.05, 4.98, 5.02, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	# Calculate batch average
	batch_avg = Mean(measurements)

	# Define specification
	spec = PlusMinus("Voltage Spec", 5.0, 0.1)

	# Check if batch average meets spec
	quality_check = Approximately(batch_avg, spec)

	assert quality_check.unwrap(ctx) is True
	result_str = quality_check.to_string(ctx)
	assert "≈" in result_str
	assert "True" in result_str


def test_combined_statistical_predicate() -> None:
	"""Test combining multiple statistical measures in a predicate."""
	ctx = BatchData(measurements=[4.95, 5.05, 4.98, 5.02, 5.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	# Statistical measures
	batch_avg = Mean(measurements)
	batch_std = StdDev(measurements)

	# Specifications
	spec = PlusMinus("Voltage Spec", 5.0, 0.1)
	max_variation = Const("Max StdDev", 0.05)

	# Combined quality check
	quality_check = Predicate("Batch Quality", (batch_avg == spec) & (batch_std < max_variation))

	result = quality_check.unwrap(ctx)
	assert isinstance(result, bool)

	result_str = quality_check.to_string(ctx)
	assert "Batch Quality:" in result_str
	assert "mean(" in result_str
	assert "stddev(" in result_str


def test_statistical_arithmetic() -> None:
	"""Test that statistical operations support arithmetic."""
	ctx = BatchData(measurements=[2.0, 4.0, 6.0, 8.0, 10.0], part_count=5, batch_id="B123")
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	batch_avg = Mean(measurements)  # Should be 6.0
	batch_range = Range(measurements)  # Should be 8.0

	# Arithmetic with statistical operations
	ratio_expr = batch_range / batch_avg  # 8.0 / 6.0 = 1.333...

	result = ratio_expr.unwrap(ctx)
	expected = 8.0 / 6.0
	assert abs(result - expected) < 0.0001


def test_empty_iterable_handling() -> None:
	"""Test handling of empty iterables."""
	ctx = BatchData(measurements=[], part_count=0, batch_id="B123")
	measurements_list = Var[SizedIterable[float], BatchData]("measurements")

	count_expr = Count(measurements_list)
	assert count_expr.unwrap(ctx) == 0

	# Statistical functions should raise appropriate errors for empty data
	measurements = Var[SizedIterable[float], BatchData]("measurements")
	mean_expr = Mean(measurements)
	with pytest.raises(statistics.StatisticsError):
		mean_expr.unwrap(ctx)


def test_process_control_scenario() -> None:
	"""Test a typical process control scenario."""
	# Simulate voltage measurements from a batch of parts
	ctx = BatchData(
		measurements=[4.98, 5.02, 4.99, 5.01, 4.97, 5.03, 4.99, 5.00],
		part_count=8,
		batch_id="LOT456",
	)

	measurements = Var[SizedIterable[float], BatchData]("measurements")

	# Process control metrics
	batch_avg = Mean(measurements)
	process_variation = StdDev(measurements)
	measurements_list = Var[SizedIterable[float], BatchData]("measurements")
	part_count = Count(measurements_list)

	# Specifications
	voltage_spec = PlusMinus("Voltage Target", 5.0, 0.05)
	max_std = Const("Process Limit", 0.03)
	min_parts = Const("Min Parts", 5)

	# Combined process control check
	process_ok = Predicate(
		"Process Control",
		(batch_avg == voltage_spec) & (process_variation < max_std) & (part_count >= min_parts),
	)

	result = process_ok.unwrap(ctx)
	assert isinstance(result, bool)

	# Verify individual components
	assert batch_avg.unwrap(ctx) == pytest.approx(4.99875, abs=0.0001)
	assert part_count.unwrap(ctx) == 8

	result_str = process_ok.to_string(ctx)
	assert "Process Control:" in result_str


@pytest.mark.mypy_testing
def test_statistical_generic_types() -> None:
	"""Test that statistical operations have correct generic types."""
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	# Test Mean
	mean_expr = Mean(measurements)
	assert_type(mean_expr, Mean[float, BatchData])
	assert_type(mean_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test StdDev
	stddev_expr = StdDev(measurements)
	assert_type(stddev_expr, StdDev[float, BatchData])
	assert_type(stddev_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test Median
	median_expr = Median(measurements)
	assert_type(median_expr, Median[float, BatchData])
	assert_type(median_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test Percentile
	p95_expr = Percentile(95, measurements)
	assert_type(p95_expr, Percentile[BatchData])
	assert_type(p95_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test Range
	range_expr = Range(measurements)
	assert_type(range_expr, Range[BatchData])
	assert_type(range_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test Count
	measurements_list = Var[SizedIterable[float], BatchData]("measurements")
	count_expr = Count(measurements_list)
	assert_type(count_expr, Count[BatchData])
	assert_type(count_expr.unwrap(BatchData([1.0, 2.0], 2, "B123")), int)

	# Test arithmetic with statistical operations
	mean_plus_one = mean_expr + 1.0
	assert_type(mean_plus_one.unwrap(BatchData([1.0, 2.0], 2, "B123")), float)

	# Test comparison with tolerance
	spec = PlusMinus("Spec", 1.5, 0.1)
	quality_check = mean_expr == spec
	assert (
		quality_check.to_string(BatchData([1.0, 2.0], 2, "B123"))
		== "(mean(measurements:2[1.0,2.0] -> 1.5) ≈ Spec:1.5 ± 0.1 -> True)"
	)


def test_iterable_string_representation() -> None:
	"""Test the new string representation format for iterables."""
	measurements = Var[SizedIterable[float], BatchData]("measurements")

	# Test single element
	single_ctx = BatchData(measurements=[5.0], part_count=1, batch_id="SINGLE")
	mean_expr = Mean(measurements)
	assert mean_expr.to_string(single_ctx).startswith("mean(measurements:1[5.0]")

	# Test two elements (should show all)
	two_ctx = BatchData(measurements=[3.0, 7.0], part_count=2, batch_id="TWO")
	assert mean_expr.to_string(two_ctx).startswith("mean(measurements:2[3.0,7.0]")

	# Test more than two elements (should show first..last)
	many_ctx = BatchData(measurements=[1.0, 2.0, 3.0, 4.0, 5.0], part_count=5, batch_id="MANY")
	assert mean_expr.to_string(many_ctx).startswith("mean(measurements:5[1.0,..5.0]")

	# Test with different statistical operations
	stddev_expr = StdDev(measurements)
	assert stddev_expr.to_string(many_ctx).startswith("stddev(measurements:5[1.0,..5.0]")

	median_expr = Median(measurements)
	assert median_expr.to_string(many_ctx).startswith("median(measurements:5[1.0,..5.0]")

	range_expr = Range(measurements)
	assert range_expr.to_string(many_ctx).startswith("range(measurements:5[1.0,..5.0]")

	# Test with Percentile (different to_string implementation)
	p95_expr = Percentile(95, measurements)
	assert p95_expr.to_string(many_ctx).startswith("percentile:95(measurements:5[1.0,..5.0]")

	# Test Count with empty list (doesn't fail like statistical operations)
	measurements_list = Var[SizedIterable[float], BatchData]("measurements")
	count_expr = Count(measurements_list)
	empty_ctx = BatchData(measurements=[], part_count=0, batch_id="EMPTY")
	assert count_expr.to_string(empty_ctx).startswith("count(measurements:0[]")

	# Test without context (should use original format)
	assert mean_expr.to_string() == "mean(measurements)"
	assert p95_expr.to_string() == "percentile:95(measurements)"
