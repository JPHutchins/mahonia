# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

from pathlib import Path

from camas import Parallel, Sequential, Task

format = Task("uv run ruff format .")
format_check = Task("uv run ruff format --check .")
lint = Task("uv run ruff check .")
lint_fix = Task("uv run ruff check --fix .")
fix = Sequential(lint_fix, format)
mypy = Task("uv run mypy .")
pyright = Task("uv run pyright src examples tests")
typecheck = Parallel(mypy, pyright)
test = Task("uv run pytest --doctest-modules")
coverage = Task("uv run pytest --doctest-modules --cov --cov-report=xml --cov-report=term-missing")

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
