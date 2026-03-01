# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import NamedTuple, assert_type

import pytest

from mahonia import Failure, PlusMinus, Result, Var
from mahonia.python_func import (
	PythonFunc1,
	PythonFunc1Wrapper,
	ResultApproximately,
	python_func,
)


class Reading(NamedTuple):
	cal_path: str
	raw_counts: float


def read_gain(path: str) -> float:
	return float(Path(path).read_text().strip())


def validate_adc(raw: float) -> float:
	if not (0 <= raw <= 4095):
		raise ValueError(f"ADC reading out of range [0, 4095]: {raw}")
	return raw


safe_gain = python_func(read_gain)
safe_validate = python_func(validate_adc)

cal_path = Var[str, Reading]("cal_path")
raw_counts = Var[float, Reading]("raw_counts")

calibrated = safe_validate(raw_counts) * safe_gain(cal_path)

target = PlusMinus("target_voltage", 3.3, 0.1)
in_spec = calibrated == target


@pytest.fixture
def cal_file(tmp_path: Path) -> str:
	f = tmp_path / "cal.txt"
	f.write_text("0.000825\n")
	return str(f)


def test_specification_is_the_program() -> None:
	"""to_string() without context IS the program specification."""
	assert calibrated.to_string() == "(validate_adc(raw_counts) * read_gain(cal_path))"
	assert (
		in_spec.to_string()
		== "((validate_adc(raw_counts) * read_gain(cal_path)) ≈ target_voltage:3.3 ± 0.1)"
	)


def test_audit_trail_shows_every_intermediate(cal_file: str) -> None:
	"""to_string(ctx) shows every intermediate value — the core value of mahonia."""
	ctx = Reading(cal_path=cal_file, raw_counts=4000.0)

	cal_trace = calibrated.to_string(ctx)
	assert "validate_adc(raw_counts:4000.0) -> 4000.0" in cal_trace
	assert "read_gain(cal_path:" in cal_trace
	assert "-> 0.000825" in cal_trace
	assert "-> 3.3" in cal_trace

	spec_trace = in_spec.to_string(ctx)
	assert "≈" in spec_trace
	assert "target_voltage:3.3 ± 0.1" in spec_trace
	assert "-> True" in spec_trace


def test_failure_audit_trail(cal_file: str) -> None:
	"""Validation failure trace shows WHERE the pipeline broke."""
	ctx = Reading(cal_path=cal_file, raw_counts=-1.0)

	trace = calibrated.to_string(ctx)
	assert "validate_adc(raw_counts:-1.0) -> Failure" in trace
	assert "ADC reading out of range" in trace
	assert "read_gain(cal_path:" in trace
	assert "-> 0.000825" in trace


def test_io_failure_audit_trail() -> None:
	"""Missing cal file produces a Failure trace showing which FFI failed."""
	ctx = Reading(cal_path="/nonexistent/cal.txt", raw_counts=4000.0)

	trace = calibrated.to_string(ctx)
	assert "validate_adc(raw_counts:4000.0) -> 4000.0" in trace
	assert "read_gain(cal_path:/nonexistent/cal.txt) -> Failure" in trace

	result = calibrated.unwrap(ctx)
	assert isinstance(result, Failure)
	assert isinstance(result.exceptions[0], (FileNotFoundError, OSError))


def test_error_accumulation() -> None:
	"""Both validation AND IO fail — two Failures accumulated in one trace."""
	ctx = Reading(cal_path="/nonexistent/cal.txt", raw_counts=-1.0)

	result = calibrated.unwrap(ctx)
	assert isinstance(result, Failure)
	assert len(result.exceptions) == 2

	trace = calibrated.to_string(ctx)
	assert "validate_adc(raw_counts:-1.0) -> Failure" in trace
	assert "read_gain(cal_path:/nonexistent/cal.txt) -> Failure" in trace


def test_tolerance_pass_and_fail(cal_file: str) -> None:
	"""ResultApproximately: within ±0.1V passes, outside fails."""
	ctx_pass = Reading(cal_path=cal_file, raw_counts=4000.0)
	assert in_spec.unwrap(ctx_pass) is True

	ctx_edge = Reading(cal_path=cal_file, raw_counts=3879.0)
	assert in_spec.unwrap(ctx_edge) is True

	ctx_fail = Reading(cal_path=cal_file, raw_counts=2000.0)
	assert in_spec.unwrap(ctx_fail) is False

	assert "≈" in in_spec.to_string()
	assert "target_voltage" in in_spec.to_string()
	assert "± 0.1" in in_spec.to_string()


def test_tolerance_with_failure(cal_file: str) -> None:
	"""Failure propagates through ResultApproximately — no crash, just Failure."""
	ctx = Reading(cal_path=cal_file, raw_counts=-1.0)

	result = in_spec.unwrap(ctx)
	assert isinstance(result, Failure)
	assert "ADC reading out of range" in str(result.exceptions[0])

	trace = in_spec.to_string(ctx)
	assert "≈" in trace
	assert "Failure" in trace


def test_reusable_across_readings(cal_file: str) -> None:
	"""Same expression tree, three different readings, three different traces."""
	readings = [
		Reading(cal_path=cal_file, raw_counts=4000.0),
		Reading(cal_path=cal_file, raw_counts=2000.0),
		Reading(cal_path=cal_file, raw_counts=0.0),
	]

	results = [calibrated.unwrap(r) for r in readings]
	assert results[0] == pytest.approx(3.3)  # pyright: ignore[reportUnknownMemberType]
	assert results[1] == pytest.approx(1.65)  # pyright: ignore[reportUnknownMemberType]
	assert results[2] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]

	traces = [calibrated.to_string(r) for r in readings]
	assert "-> 3.3" in traces[0]
	assert "-> 1.65" in traces[1]
	assert "-> 0.0" in traces[2]


def test_compliance_snapshot(cal_file: str) -> None:
	"""bind() creates a frozen compliance record with the full audit trail."""
	ctx = Reading(cal_path=cal_file, raw_counts=4000.0)

	record = calibrated.bind(ctx)
	assert record.unwrap() == pytest.approx(3.3)  # pyright: ignore[reportUnknownMemberType]
	assert record.ctx == ctx
	assert "3.3" in str(record)


def test_ffi_virality_types() -> None:
	"""FFI contamination propagates through the entire expression tree."""
	assert_type(safe_gain, PythonFunc1Wrapper[str, float])
	assert_type(safe_validate, PythonFunc1Wrapper[float, float])
	assert_type(safe_gain(cal_path), PythonFunc1[str, float, Reading])
	assert_type(calibrated, Result[float, Reading, float])
	assert_type(in_spec, ResultApproximately[float, Reading])  # type: ignore[assert-type]
