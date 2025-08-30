# Claude Instructions for Mahonia

Mahonia is a domain specific language (DSL) for defining, evaluating, saving, and serializing binary expressions within a Python interpreter.

## Project Structure
- `src/mahonia/` - Main source code
  - `latex.py` - LaTeX conversion functionality  
  - `stats.py` - Statistics functionality
- `tests/` - Test files
- Uses Python 3.12+
- Uses hatch for project management
- Uses ruff for linting and formatting
- Uses mypy for type checking
- Uses pytest for testing

## Development Commands

### Setup
```bash
source .venv/bin/activate  # Activate virtual environment
```

### Testing
```bash
hatch run test  # Run all tests including doctests
pytest  # Run pytest directly
```

### Linting and Formatting
```bash
hatch run format  # Format code with ruff
hatch run lint    # Run ruff check and mypy
ruff format .     # Format directly
ruff check .      # Lint directly  
mypy .           # Type check directly
```

### All checks
```bash
hatch run all    # Run format, lint, and test
```

## Key Features
- Binary expression evaluation with context binding
- LaTeX mathematical notation support
- Immutable BoundExpr objects for repeated evaluation
- Serialization for logging and debugging
- Type-safe variable definitions
- Statistics expressions

## Important Files
- `tests/latex_examples.md` - pytest-generated latex examples
- `pyproject.toml` - Project configuration
- LaTeX functionality is in `src/mahonia/latex.py`

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