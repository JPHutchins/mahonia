# Claude Instructions for Mahonia

Mahonia is a domain specific language (DSL) for defining, evaluating, saving, and serializing binary expressions within a Python interpreter.

## Project Structure
- `src/mahonia/` - Main source code
  - `__init__.py` - Core expression types, operators, `BinaryOperationOverloads`
  - `python_func.py` - Python function FFI with `ResultBinaryOperationOverloads`, `ResultExpr` protocol
  - `latex.py` - LaTeX conversion functionality
  - `stats.py` - Statistics functionality
- `tests/` - Test files
- `plans/` - Design documents
- Uses Python 3.12+
- Uses uv for package management
- Uses ruff for linting and formatting
- Uses mypy and pyright for type checking
- Uses pytest for testing

## Development Commands

### Setup
```bash
source .venv/bin/activate  # Activate virtual environment
```

### Testing
```bash
uv run task test  # Run all tests including doctests
uv run pytest     # Run pytest directly
```

### Linting and Formatting
```bash
uv run task format  # Format code with ruff
uv run task lint    # Run ruff check, mypy, and pyright
uv run ruff format .     # Format directly
uv run ruff check .      # Lint directly
uv run mypy .            # Type check directly
```

### All checks
```bash
uv run task all    # Run format, lint, and test
```

## Key Features
- Binary expression evaluation with context binding
- LaTeX mathematical notation support
- Immutable BoundExpr objects for repeated evaluation
- Serialization for logging and debugging
- Type-safe variable definitions
- Statistics expressions
- Python function FFI with railroad-oriented error handling (`Failure` accumulation)
- Compile-time type virality: FFI contamination propagates bidirectionally through `ResultExpr` protocol

## Important Files
- `tests/latex_examples.md` - pytest-generated latex examples
- `pyproject.toml` - Project configuration
- `plans/python_func_ffi.md` - FFI design document

## Notes
- Uses tab indentation (configured in ruff)
- Line length limit of 100 characters
- Requires Python 3.12+
- No external runtime dependencies (dependencies are dev-only)
- Use FP principles
- Do not create named variables that are used only 1x (prefer function composition)
- Minimal python doc strings are OK, but inline comments are almost never desired
- Do not import redundant type utilities like Type, Union, etc.
- Prefer strong immutable types like NamedTuple, and use type unions to create sum types