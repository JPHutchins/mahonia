# Mahonia

Mahonia is a domain specific language (DSL) for defining, evaluating, saving, and
serializing binary expressions within a Python interpreter.

[![Test](https://github.com/JPHutchins/mahonia/actions/workflows/test.yaml/badge.svg)](https://github.com/JPHutchins/mahonia/actions/workflows/test.yaml)
[![Publish Python 🐍 distribution 📦 to PyPI](https://github.com/JPHutchins/mahonia/actions/workflows/publish.yaml/badge.svg)](https://github.com/JPHutchins/mahonia/actions/workflows/publish.yaml)

## Motivation

Say that you are writing an application that conducts some assembly line testing
at a manufacturing facility. While the primary requirement is to flag those
units that do not meet expectations, a secondary requirement is to record _what,
why, and how_ a test has failed.

First, define what is being measured - the "context".
```python
from typing import NamedTuple

from mahonia import Approximately, PlusMinus, Predicate, Var

class Measurements(NamedTuple):
	voltage: float
```

Next, for each "variable" of the context, we declare a matching Mahonia `Var` type.
```python
voltage = Var[float, Measurements]("voltage")
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
print(voltage_check.ctx) # Measurements(voltage=5.03)
```

As well as serialize it for the logs:
```python
str(voltage_check) # or voltage_check.to_string()
# If it was success, for example:
# Voltage OK: True (voltage:5.03 ≈ Nominal:5.0 ± 0.05 -> True)
# Or a fail:
# Voltage OK: False (voltage:4.90 ≈ Nominal:5.0 ± 0.05 -> False)
```

## LaTeX Support

Mahonia expressions can be converted to LaTeX mathematical notation:

```python
from mahonia.latex import latex

# Convert expressions to LaTeX
latex(voltage)  # 'voltage'
latex(expr.expr)  # 'voltage \\approx Nominal \\pm 0.05'
```

For comprehensive examples of LaTeX output for all supported expression types, see [tests/latex_examples.md](tests/latex_examples.md).
