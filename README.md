# Mahonia

Mahonia is a domain specific language (DSL) for defining, evaluating, saving, and
serializing binary expressions within a Python interpreter.

## Motivation

Say that you are writing an application that conducts some assembly line testing
at a manufacturing facility. While the primary requirement is to flag those
units that do not meet expectations, a secondary requirement is to record _what,
why, and how_ a test has failed.

For this example, we will use Mahonia's `Predicate` type.

First, we have to define what we are measuring - the "context".
```python
from typing import NamedTuple

from mahonia import Approximately, PlusMinus, Predicate, Var

class MyContext(NamedTuple):
	voltage: float
```

Next, for each "variable" of the context, we declare a matching Mahonia `Var` type.
```python
voltage = Var[float, MyContext]("voltage")
```

Now we can write an expression. This expression defines a named predicate that
will evaluate to `True` if the evaluated voltage is within 0.05 of 5.0.
```python
expr = Predicate(
	"Voltage OK",
	Approximately(
		voltage, PlusMinus("Nominal", 5.0, plus_minus=0.05)
	)
)
```

Then we'll take the measurement and bind it:
```python
from my_app import get_voltage

voltage_check = expr.bind(MyContext(voltage=get_voltage()))
```

This creates an immutable expression that Mahonia calls a `BoundExpr`. We can
evaluate it as many times as we like:
```python
voltage_check.unwrap() # True
voltage_check.unwrap() # True
```

We can inspect the evaluation context:
```python
print(voltage_check.ctx) # MyContext(voltage=5.03)
```

As well as serialize it for the logs:
```python
str(voltage_check) # or voltage_check.to_string()
# If it was success, for example:
# Voltage OK: True (voltage:5.03 ≈ Nominal:5.0 ± 0.05 -> True)
# Or a fail:
# Voltage OK: False (voltage:4.90 ≈ Nominal:5.0 ± 0.05 -> False)
```
## Develop

Contributions are welcome!

> [!IMPORTANT]
> ### First time setup
>
> - Install [uv](https://github.com/astral-sh/uv)
> - Initialize the environment:
>   ```
>   uv sync --locked --all-extras --dev
>   ```

### Formatting
```
uv run hatch run format
```
> [!NOTE]
> VSCode is setup to do this for you on save - feel free to add more editors.

### Linting
```
uv run hatch run lint
```
> [!NOTE]
> VSCode is setup to to run these LSPs in the background - feel free to add more
> editors.

### Tests
```
uv run hatch run tests
```

### All
```
uv run hatch run all
```
