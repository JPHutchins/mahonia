import asyncio
import csv
from typing import Any, Final, NamedTuple

import cyclopts
from rich.console import Console

import mahonia
from manufacturing.types import Testable, TestStep, cached_context

app = cyclopts.App()
console = Console()


class Power(NamedTuple):
	voltage: float
	current: float


class Temperature(NamedTuple):
	temp_readings: list[float]
	temp: float


@cached_context
async def step0_check_power() -> Power:
	await asyncio.sleep(2.000)
	return Power(voltage=4.2, current=0.42)


def step1_power() -> TestStep[Power]:
	voltage = mahonia.Var[float, Power]("voltage")
	current = mahonia.Var[float, Power]("current")
	power = voltage * current

	tests = (
		(voltage > 3.0) & (voltage < 5.0),
		(current > 0.1) & (current < 1.0),
		mahonia.Predicate("Power under limit", power < 5.0),
		mahonia.Predicate("Power over minimum", power > 0.5),
	)

	return TestStep(step0_check_power().context_func, tests)


def step2_resistance() -> TestStep[Power]:
	voltage = mahonia.Var[float, Power]("voltage")
	current = mahonia.Var[float, Power]("current")
	resistance = voltage / current
	target_resistance = mahonia.PlusMinus("R_target", 10.0, 1.0)

	tests = (
		mahonia.Predicate("Resistance check", resistance == target_resistance),
		resistance > 5.0,
	)

	return TestStep(step0_check_power().context_func, tests)


def step3_temperature() -> TestStep[Temperature]:
	temps = mahonia.Var[mahonia.SizedIterable[float], Temperature]("temp_readings")
	temp = mahonia.Var[float, Temperature]("temp")
	avg_temp = mahonia.Const("avg_temp", 25.0)
	in_range = (temp > 20.0) & (temp < 30.0)
	exceeds_avg = temp > avg_temp

	tests = (
		mahonia.AllExpr(in_range.map(temps)),
		mahonia.Predicate("Some temps exceed average", mahonia.AnyExpr(exceeds_avg.map(temps))),
	)

	async def func() -> Temperature:
		await asyncio.sleep(0.5)
		return Temperature(temp_readings=[22.0, 24.0, 26.0], temp=25.0)

	return TestStep(func, tests)


STEPS: Final = (
	step0_check_power,
	step1_power,
	step2_resistance,
	step3_temperature,
)


class PassContext(NamedTuple):
	tests: mahonia.SizedIterable[mahonia.BoundExpr[Any, Any, Any]]

	@classmethod
	def from_cases(
		cls,
		tests: mahonia.SizedIterable[Testable],
		ctx: mahonia.ContextProtocol,
	) -> "PassContext":
		return cls(tests=tuple(t.bind(ctx) for t in tests))


@app.default
async def run() -> None:
	tests: Final = mahonia.Var[
		mahonia.SizedIterable[mahonia.BoundExpr[Any, Any, Any]], PassContext
	]("tests")
	pass_expr: Final = mahonia.FoldLExpr(mahonia.And, tests)

	for step in STEPS:
		console.print(f"Test step: {step.__name__}")
		for test in step().tests:
			print(f"  Test: {test.to_string()}")

	console.print()
	console.print("----------------------")
	console.print()

	class ResultContext(NamedTuple):
		results: list[mahonia.BoundExpr[Any, Any, Any]]

	results: Final = mahonia.Var[
		mahonia.SizedIterable[mahonia.BoundExpr[Any, Any, Any]], ResultContext
	]("results")

	result_context = ResultContext(results=[])

	with open("test_results.csv", "w", newline="") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Step", "Pass/Fail", "Context", "Tests"])

		for step in STEPS:
			ctx = await step().context_func()
			console.print(f"Test step: {step.__name__} {ctx=}")

			if len(step().tests) == 0:
				console.print("  No tests defined, skipping.")
				continue
			pass_bound = pass_expr.bind(PassContext.from_cases(step().tests, ctx))
			result_context.results.append(pass_bound)

			pass_fail = "PASS" if pass_bound.unwrap() else "FAIL"
			writer.writerow(
				(
					step.__name__,
					pass_fail,
					str(ctx),
					str(pass_bound),
				)
			)
			for test in step().tests:
				print(f"  Test: {test.to_string(ctx)}")

	result_expr = mahonia.FoldLExpr(mahonia.And, results)

	console.print()
	console.print(f"Overall result: {'PASS' if result_expr.unwrap(result_context) else 'FAIL'}")


@app.command
async def list_tests() -> None: ...


def main() -> None:
	app()
