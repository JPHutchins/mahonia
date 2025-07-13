# Bix

Bix is a domain specific language (DSL) for defining, evaluating, saving, and
serializing (Bi)nary E(x)pressions within a Python interpreter.

## Syntax

Bix overloads Python operators:

- `+`: `Add`
- `-`: `Sub`
- `*`: `Mul`
- `/`: `Div` (floats only)
- `==`: `Eq`
- `!=`: `Neq`
- `<`: `Lt`
- `<=`: `Lte`
- `>`: `Gt`
- `>=`: `Gte`
- `&`: `And` (logical)
- `|`: `Or` (logical)
- `~`: `Not` (logical)

So, the expression `(x + y) > 0` would create an instance of the following type:
```python
Gt(
	Add(
		Var,
		Var,
	),
	Const
)
```

## For Example

```python
from typing import NamedTuple, assert_type
import binexpr as bix

class Ctx(NamedTuple):
	x: int
	y: int

x = bix.Var[int, Ctx]("x")
y = bix.Var[int, Ctx]("y")
MAX = bix.Const("Max", 10)

# Create an expression that adds two variables
sum_expr = x + y
assert_type(sum_expr, bix.Add[int, Ctx])

# Serialize the expression to a string
print(sum_expr.to_string())  # (x + y)

# Evaluate the expression with concrete values
sum_result = sum_expr(Ctx(x=1, y=2))
assert_type(sum_result, bix.Const[int])

# Serialize the evaluated expression to a string
print(sum_expr.to_string(Ctx(x=1, y=2)))
# (x:1 + y:2 -> 3)

# Access the value of the result
sum_value = sum_result.value
assert_type(sum_value, int)
print(sum_value)  # 3

# Shorthand for getting the value
sum_value = sum_expr.unwrap(Ctx(x=1, y=2))
assert_type(sum_value, int)
print(sum_value)  # 3

# Bind an expression to a context
bound_expr = sum_expr.bind(Ctx(x=1, y=2))
assert_type(bound_expr, bix.BoundExpr[int, Ctx])
print(bound_expr)  # (x:1 + y:2 -> 3)
print(bound_expr.unwrap())  # 3

# Combine expressions
zero = x + y - x - y
assert_type(zero, bix.Sub[int, Ctx])
print(zero.to_string())  # (((x + y) - x) - y)
print(zero.to_string(Ctx(x=1, y=2)))
# (((x:1 + y:2 -> 3) - x:1 -> 2) - y:2 -> 0)
print(zero.unwrap(Ctx(x=1, y=2)))  # 0

# Compose expressions
four = sum_expr + x
assert_type(four, bix.Add[int, Ctx])
print(four.to_string())  # ((x + y) + x)
print(four.to_string(Ctx(x=1, y=2)))
# ((x:1 + y:2 -> 3) + x:1 -> 4)
print(four.unwrap(Ctx(x=1, y=2)))  # 4

# Create a predicate
is_valid = (x > 0) & (y < MAX)
print(is_valid.to_string())  # ((x > 0) & (y < Max:10))
print(is_valid.to_string(Ctx(x=1, y=2)))
# ((x:1 > 0 -> True) & (y:2 < Max:10 -> True) -> True)
print(is_valid.unwrap(Ctx(x=1, y=2)))  # True

# Define the predicate
cate = bix.Predicate("x is pos, y less than Max", is_valid)
assert_type(cate, bix.Predicate[Ctx])
print(cate.to_string())
# x is pos, y less than Max: ((x > 0) & (y < Max:10))
print(cate.to_string(Ctx(x=1, y=2)))
# x is pos, y less than Max: True ((x:1 > 0 -> True) & (y:2 < Max:10 -> True) -> True)
print(cate.unwrap(Ctx(x=1, y=2)))  # True
```

### Motivation

Say that you are writing an application that conducts some assembly line testing
at a manufacturing facility. While the primary requirement is to flag those
units that do not meet expectations, a secondary requirement is to record _what,
why, and how_ a test has failed.

For this example, we will use Bix's `Predicate` type.

First, we have to define what we are measuring - the "context".
```python
from typing import NamedTuple

import binexpr as bix

class MyContext(NamedTuple):
	voltage: float
```

Next, for each "variable" of the context, we declare a matching Bix `Var` type.
```python
voltage = bix.Var[float, MyContext]("voltage")
```

Now we can write an expression. This expression defines a named predicate that
will evaluate to `True` if the evaluated voltage is within 0.05 of 5.0.
```python
expr = bix.Predicate(
	"Voltage OK",
	bix.Approximately(
		voltage, bix.PlusMinus("Nominal", 5.0, plus_minus=0.05)
	)
)
```

Then we'll take the measurement and bind it:
```python
from my_app import get_voltage

voltage_check = expr.bind(MyContext(voltage=get_voltage()))
```

This creates an immutable expression that Bix calls a `BoundExpr`. We can
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
>   uv sync
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
