# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Final

from camas import Parallel, Sequential, Task

UV: Final = "uv run --all-packages"

format = Task(f"{UV} ruff format .")
format_check = Task(f"{UV} ruff format --check .")
lint = Task(f"{UV} ruff check .")
lint_fix = Task(f"{UV} ruff check --fix .")
fix = Sequential(lint_fix, format)
mypy = Task(f"{UV} mypy .")
pyright = Task(f"{UV} pyright src examples tests")
typecheck = Parallel(mypy, pyright)
test = Task(f"{UV} pytest --doctest-modules")
coverage = Task(f"{UV} pytest --doctest-modules --cov --cov-report=xml --cov-report=term-missing")

all = Sequential(fix, Parallel(typecheck, test))
check = Parallel(format_check, lint, typecheck, test)

matrix = Sequential(
	Task("uv sync --all-packages"),
	check,
	env={"UV_PROJECT_ENVIRONMENT": ".venv-{PY}", "UV_PYTHON": "{PY}"},
	matrix={
		"PY": tuple(
			stripped
			for line in (Path(__file__).parent / ".python-version").read_text().splitlines()
			if (stripped := line.strip()) and not stripped.startswith("#")
		)
	},
)
