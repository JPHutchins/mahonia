from typing import TypeVar, Generic, Protocol, runtime_checkable, overload, Union
from dataclasses import dataclass

L = TypeVar("L")
R = TypeVar("R")
O = TypeVar("O")
C = TypeVar("C")

@runtime_checkable
class Expr(Protocol[O, C]):
    def eval(self, ctx: C) -> "Const[O, C]": ...
    def to_string(self, ctx: C | None = None) -> str: ...
    def __call__(self, ctx: C) -> "Const[O, C]":
        return self.eval(ctx)

class ExprBase(Generic[O, C]):
    # Arithmetic
    def __add__(self, other: Union["ExprBase", O]) -> "Add":
        if isinstance(other, ExprBase):
            return Add(self, other)
        else:
            return Add(self, Const(other))

    def __sub__(self, other: Union["ExprBase", O]) -> "Sub":
        if isinstance(other, ExprBase):
            return Sub(self, other)
        else:
            return Sub(self, Const(other))

    def __mul__(self, other: Union["ExprBase", O]) -> "Mul":
        if isinstance(other, ExprBase):
            return Mul(self, other)
        else:
            return Mul(self, Const(other))

    def __truediv__(self, other: Union["ExprBase", O]) -> "Div":
        if isinstance(other, ExprBase):
            return Div(self, other)
        else:
            return Div(self, Const(other))

    # Bitwise/logical
    def __and__(self, other: Union["ExprBase", O]) -> "And":
        if isinstance(other, ExprBase):
            return And(self, other)
        else:
            return And(self, Const(other))

    def __or__(self, other: Union["ExprBase", O]) -> "Or":
        if isinstance(other, ExprBase):
            return Or(self, other)
        else:
            return Or(self, Const(other))

    # Comparison
    def __eq__(self, other: object) -> "Eq":
        if isinstance(other, ExprBase):
            return Eq(self, other)
        else:
            return Eq(self, Const(other))

    def __ne__(self, other: object) -> "Ne":
        if isinstance(other, ExprBase):
            return Ne(self, other)
        else:
            return Ne(self, Const(other))

    def __lt__(self, other: Union["ExprBase", O]) -> "Lt":
        if isinstance(other, ExprBase):
            return Lt(self, other)
        else:
            return Lt(self, Const(other))

    def __le__(self, other: Union["ExprBase", O]) -> "Le":
        if isinstance(other, ExprBase):
            return Le(self, other)
        else:
            return Le(self, Const(other))

    def __gt__(self, other: Union["ExprBase", O]) -> "Gt":
        if isinstance(other, ExprBase):
            return Gt(self, other)
        else:
            return Gt(self, Const(other))

    def __ge__(self, other: Union["ExprBase", O]) -> "Ge":
        if isinstance(other, ExprBase):
            return Ge(self, other)
        else:
            return Ge(self, Const(other))

    # Unary
    def __invert__(self) -> "Not":
        return Not(self)

# ---- Core Expressions ----

@dataclass(frozen=True)
class Const(ExprBase, Generic[O, C], Expr[O, C]):
    value: O
    def eval(self, ctx: C) -> "Const[O, C]":
        return self
    def to_string(self, ctx: C | None = None) -> str:
        return str(self.value)

@dataclass(frozen=True)
class Var(ExprBase, Generic[O, C], Expr[O, C]):
    name: str
    def eval(self, ctx: C) -> Const[O, C]:
        return Const(getattr(ctx, self.name))
    def to_string(self, ctx: C | None = None) -> str:
        if ctx is not None:
            try:
                value = getattr(ctx, self.name)
                return f"{self.name}:{value}"
            except AttributeError:
                pass
        return self.name

# ---- Binary Operations ----

@dataclass(frozen=True)
class Add(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        lval = self.left.eval(ctx).value
        rval = self.right.eval(ctx).value
        return Const(lval + rval)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} + {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Sub(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        lval = self.left.eval(ctx).value
        rval = self.right.eval(ctx).value
        return Const(lval - rval)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} - {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Mul(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        lval = self.left.eval(ctx).value
        rval = self.right.eval(ctx).value
        return Const(lval * rval)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} * {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Div(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        lval = self.left.eval(ctx).value
        rval = self.right.eval(ctx).value
        return Const(lval / rval)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} / {self.right.to_string(ctx)})"

# ---- Boolean Operations ----

@dataclass(frozen=True)
class Eq(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value == self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} == {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Ne(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value != self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} != {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Lt(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value < self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} < {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Le(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value <= self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} <= {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Gt(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value > self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} > {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Ge(ExprBase, Generic[L, R, C], Expr[bool, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[bool, C]:
        return Const(self.left.eval(ctx).value >= self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} >= {self.right.to_string(ctx)})"

# ---- Logical Operations ----

@dataclass(frozen=True)
class And(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        return Const(self.left.eval(ctx).value and self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} and {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Or(ExprBase, Generic[L, R, O, C], Expr[O, C]):
    left: ExprBase
    right: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        return Const(self.left.eval(ctx).value or self.right.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"({self.left.to_string(ctx)} or {self.right.to_string(ctx)})"

@dataclass(frozen=True)
class Not(ExprBase, Generic[O, C], Expr[O, C]):
    expr: ExprBase
    def eval(self, ctx: C) -> Const[O, C]:
        return Const(not self.expr.eval(ctx).value)
    def to_string(self, ctx: C | None = None) -> str:
        return f"(not {self.expr.to_string(ctx)})"

# ---- Example Context ----

@dataclass(frozen=True)
class MyContext:
    x: int
    y: float
    name: str

# ---- Usage Example ----

x = Var[int, MyContext]("x")
y = Var[float, MyContext]("y")
name = Var[str, MyContext]("name")

ctx = MyContext(x=5, y=10.0, name="example")

# Arithmetic
sum_expr = x + y  # Add[int, float, float, MyContext]
mul_expr = x * 5  # Mul[int, int, int, MyContext]

# Boolean
eq_expr = x == 5
lt_expr = x < y

# Logical
and_expr = eq_expr & lt_expr
or_expr = eq_expr | lt_expr

print(sum_expr.to_string(ctx), "->", sum_expr.eval(ctx))
print(mul_expr.to_string(ctx), "->", mul_expr.eval(ctx))
print(eq_expr.to_string(ctx), "->", eq_expr.eval(ctx))
print(lt_expr.to_string(ctx), "->", lt_expr.eval(ctx))
print(and_expr.to_string(ctx), "->", and_expr.eval(ctx))
print(or_expr.to_string(ctx), "->", or_expr.eval(ctx))