# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Test enhanced error messages for variable evaluation."""

from typing import NamedTuple

import pytest

from mahonia import EvalError, Var


class ExampleContext(NamedTuple):
	voltage: float
	temperature: float
	current: float


def test_variable_typo_suggestions() -> None:
	"""Test that typos in variable names provide helpful suggestions."""
	ctx = ExampleContext(voltage=5.0, temperature=25.0, current=1.2)

	# Test with a typo that should suggest 'voltage'
	voltage_typo = Var[float, ExampleContext]("voltag")  # Missing 'e'

	with pytest.raises(EvalError, match=r"Variable 'voltag' not found.*did you mean 'voltage'"):
		voltage_typo.unwrap(ctx)

	# Test with a typo that should suggest 'temperature'
	temp_typo = Var[float, ExampleContext]("temperatur")  # Missing 'e'

	with pytest.raises(
		EvalError, match=r"Variable 'temperatur' not found.*did you mean 'temperature'"
	):
		temp_typo.unwrap(ctx)


def test_variable_no_close_matches() -> None:
	"""Test variable not found with no close matches."""
	ctx = ExampleContext(voltage=5.0, temperature=25.0, current=1.2)

	# Test with completely wrong name
	wrong_name = Var[float, ExampleContext]("foobar")

	with pytest.raises(EvalError, match=r"Variable 'foobar' not found in context$"):
		wrong_name.unwrap(ctx)


def test_multiple_suggestions() -> None:
	"""Test that multiple close matches are suggested."""

	class MultiFieldContext(NamedTuple):
		current_voltage: float
		current_temp: float
		current_flow: float

	ctx = MultiFieldContext(current_voltage=5.0, current_temp=25.0, current_flow=1.2)

	# 'current' should match multiple fields
	typo = Var[float, MultiFieldContext]("curren")  # Missing 't'

	with pytest.raises(EvalError) as exc_info:
		typo.unwrap(ctx)

	error_msg = str(exc_info.value)
	assert "Variable 'curren' not found" in error_msg
	assert "did you mean" in error_msg
	# Should suggest at least one of the current_* fields
	assert any(field in error_msg for field in ["current_voltage", "current_temp", "current_flow"])
