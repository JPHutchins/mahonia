from typing import (
    TypeVar,
    Final,
    Protocol,
    runtime_checkable,
    overload,
    Generic,
    ClassVar,
)
from dataclasses import dataclass

T = TypeVar("T")
"""The type of the expression's value."""

T_co = TypeVar("T_co", covariant=True)
"""The covariant type of the expression's value."""

S = TypeVar("S")
"""The type of the expression's context."""

S_contra = TypeVar("S_contra", contravariant=True)
"""The contravariant type of the expression's context."""


@runtime_checkable
class Eval(Protocol[T, S]):
    def eval(self, ctx: S) -> "Const[T, S]": ...


@runtime_checkable
class ToString(Protocol[S_contra]):
    def to_string(self, ctx: S_contra | None = None) -> str: ...


class Expr(Generic[T, S]):
    def eval(self, ctx: S) -> "Const[T, S]":
        raise NotImplementedError()

    def to_string(self, ctx: S | None = None) -> str:
        raise NotImplementedError()

    def __call__(self, ctx: S) -> "Const[T, S]":
        return self.eval(ctx)


class BoolExpr(Expr[bool, S], Generic[S]):
    def eval(self, ctx: S) -> "Const[bool, S]":
        raise NotImplementedError()

    def to_string(self, ctx: S | None = None) -> str:
        return super().to_string(ctx)

    def __call__(self, ctx: S) -> "Const[bool, S]":
        return self.eval(ctx)


class UnaryOperationOverloads(Expr[bool, S]):
    def __invert__(self) -> "Not[S]":
        return Not(self)


class BooleanBinaryOperationOverloads(BoolExpr[S], Generic[S]):
    def __and__(self, other: BoolExpr[S]) -> "And[S]":
        return And(self, other)

    def __or__(self, other: BoolExpr[S]) -> "Or[S]":
        return Or(self, other)

    @overload  # type: ignore[override]
    def __eq__(self, other: bool) -> "Eq[bool, S]": ...

    @overload
    def __eq__(self, other: BoolExpr[S]) -> "Eq[bool, S]": ...

    def __eq__(self, other: BoolExpr[S] | bool) -> "Eq[bool, S]":  # type: ignore[override]
        if isinstance(other, BoolExpr):
            return Eq(self, other)
        else:
            return Eq(self, Const[bool, S](other))


class BinaryOperationOverloads(Expr[T, S]):
    @overload  # type: ignore[override]
    def __eq__(self, other: Expr[T, S]) -> "Eq[T, S]": ...

    @overload  # type: ignore[override]
    def __eq__(self, other: bool) -> "Eq[bool, S]": ...

    def __eq__(self, other: Expr[T, S] | bool) -> "Eq[T, S]":
        if isinstance(other, Expr):
            return Eq(self, other)
        else:
            return Eq(self, Const[bool, S](other))

    @overload  # type: ignore[override]
    def __ne__(self, other: Expr[T, S]) -> "Ne[T, S]": ...

    @overload  # type: ignore[override]
    def __ne__(self, other: T) -> "Ne[T, S]": ...

    def __ne__(self, other: Expr[T, S] | T) -> "Ne[T, S]":  # type: ignore[override]
        if isinstance(other, Expr):
            return Ne(self, other)
        else:
            return Ne(self, Const[T, S](other))

    @overload
    def __lt__(self, other: T) -> "Lt[T, S]": ...

    @overload
    def __lt__(self, other: Expr[T, S]) -> "Lt[T, S]": ...

    def __lt__(self, other: Expr[T, S] | T) -> "Lt[T, S]":
        if isinstance(other, Expr):
            return Lt(self, other)
        else:
            return Lt(self, Const[T, S](other))

    @overload
    def __le__(self, other: T) -> "Le[T, S]": ...

    @overload
    def __le__(self, other: Expr[T, S]) -> "Le[T, S]": ...

    def __le__(self, other: Expr[T, S] | T) -> "Le[T, S]":
        if isinstance(other, Expr):
            return Le(self, other)
        else:
            return Le(self, Const[T, S](other))

    @overload
    def __gt__(self, other: T) -> "Gt[T, S]": ...

    @overload
    def __gt__(self, other: Expr[T, S]) -> "Gt[T, S]": ...

    def __gt__(self, other: Expr[T, S] | T) -> "Gt[T, S]":
        if isinstance(other, Expr):
            return Gt(self, other)
        else:
            return Gt(self, Const[T, S](other))

    @overload
    def __ge__(self, other: T) -> "Ge[T, S]": ...

    @overload
    def __ge__(self, other: Expr[T, S]) -> "Ge[T, S]": ...

    def __ge__(self, other: Expr[T, S] | T) -> "Ge[T, S]":
        if isinstance(other, Expr):
            return Ge(self, other)
        else:
            return Ge(self, Const[T, S](other))

    @overload
    def __add__(self, other: T) -> "Add[T, S]": ...

    @overload
    def __add__(self, other: Expr[T, S]) -> "Add[T, S]": ...

    def __add__(self, other: Expr[T, S] | T) -> "Add[T, S]":
        if isinstance(other, Expr):
            return Add(self, other)
        else:
            return Add(self, Const[T, S](other))

    @overload
    def __sub__(self, other: T) -> "Sub[T, S]": ...

    @overload
    def __sub__(self, other: Expr[T, S]) -> "Sub[T, S]": ...

    def __sub__(self, other: Expr[T, S] | T) -> "Sub[T, S]":
        if isinstance(other, Expr):
            return Sub(self, other)
        else:
            return Sub(self, Const[T, S](other))

    @overload
    def __mul__(self, other: T) -> "Mul[T, S]": ...

    @overload
    def __mul__(self, other: Expr[T, S]) -> "Mul[T, S]": ...

    def __mul__(self, other: Expr[T, S] | T) -> "Mul[T, S]":
        if isinstance(other, Expr):
            return Mul(self, other)
        else:
            return Mul(self, Const[T, S](other))

    @overload
    def __truediv__(self, other: T) -> "Div[T, S]": ...

    @overload
    def __truediv__(self, other: Expr[T, S]) -> "Div[T, S]": ...

    def __truediv__(self, other: Expr[T, S] | T) -> "Div[T, S]":
        if isinstance(other, Expr):
            return Div(self, other)
        else:
            return Div(self, Const[T, S](other))


@dataclass(frozen=True, eq=False)
class Const(BinaryOperationOverloads[T, S], Expr[T, S]):
    value: T

    def eval(self, ctx: S) -> "Const[T, S]":
        return self

    def to_string(self, ctx: S | None = None) -> str:
        return str(self.value)


@dataclass(frozen=True, eq=False)
class Var(BinaryOperationOverloads[T, S], Expr[T, S]):
    name: str

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(getattr(ctx, self.name))

    def to_string(self, ctx: S | None = None) -> str:
        if ctx is None:
            return self.name
        else:
            return f"{self.name}:{self.eval(ctx).value}"


@dataclass(frozen=True, eq=False)
class BinaryOpEval(Expr[T, S], Generic[T, S]):
    left: Expr[T, S]
    right: Expr[T, S]


class BinaryOpToString(ToString[S], BinaryOpEval[T, S], Generic[T, S]):
    op: ClassVar[str] = " ? "
    template: ClassVar[str] = "({left}{op}{right})"
    template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

    def to_string(self, ctx: S | None = None) -> str:
        left: Final = self.left.to_string(ctx)
        right: Final = self.right.to_string(ctx)
        if ctx is None:
            return self.template.format(left=left, op=self.op, right=right)
        else:
            return self.template_eval.format(
                left=left, op=self.op, right=right, out=self.eval(ctx).value
            )


@dataclass(frozen=True, eq=False)
class UnaryOpEval(Eval[bool, S]):
    left: Expr[bool, S]


class UnaryOpToString(ToString[S], UnaryOpEval[S]):
    op: ClassVar[str] = " ? "
    template: ClassVar[str] = "{op}{left}"
    template_eval: ClassVar[str] = "{op}{left} -> {out}"

    def to_string(self, ctx: S | None = None) -> str:
        left: Final = self.left.to_string(ctx)
        if ctx is None:
            return self.template.format(op=self.op, left=left)
        else:
            return self.template_eval.format(
                op=self.op, left=left, out=self.eval(ctx).value
            )


class And(BinaryOpToString[bool, S], BooleanBinaryOperationOverloads[S], Generic[S]):
    op: ClassVar[str] = " and "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value and self.right.eval(ctx).value)


class Or(BinaryOpToString[bool, S], BooleanBinaryOperationOverloads[S], Generic[S]):
    op: ClassVar[str] = " or "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value or self.right.eval(ctx).value)


class Not(UnaryOpToString[S], UnaryOperationOverloads[S]):
    op: ClassVar[str] = " not "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(not self.left.eval(ctx).value)


class Add(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " + "

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(self.left.eval(ctx).value + self.right.eval(ctx).value)


class Eq(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
    Generic[T, S],
):
    op: ClassVar[str] = " = "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value == self.right.eval(ctx).value)


class Ne(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " != "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value != self.right.eval(ctx).value)


class Lt(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " < "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value < self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Le(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " <= "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value <= self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Gt(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " > "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value > self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Ge(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " >= "

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const(self.left.eval(ctx).value >= self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Sub(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " - "

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(self.left.eval(ctx).value - self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Mul(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " * "

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(self.left.eval(ctx).value * self.right.eval(ctx).value)


@dataclass(frozen=True, eq=False)
class Div(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " / "

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(self.left.eval(ctx).value / self.right.eval(ctx).value)


@dataclass(frozen=True)
class Ctx:
    x: int
    y: int
    name: str


# Example usage
x = Var[int, Ctx]("x")
y = Var[int, Ctx]("y")
name = Var[str, Ctx]("name")

ctx = Ctx(x=5, y=10, name="example")

predicate = ((x == Const[int, Ctx](5)) & (y == Const[int, Ctx](10))) | (
    name == Const[str, Ctx]("test")
)


print("\nstart here\n")

TEST: Final = Const[str, Ctx]("test")

p = name == TEST
print(p.to_string())  # ( name == test )
print(p.to_string(ctx))  # ( name == test )
print(p.eval(ctx))  # True

EXAMPLE: Final = Const[str, Ctx]("example")

# test == operator
p = name == EXAMPLE
print(p(ctx))  # True
print(p.to_string())  # ( name == test )
print(p.to_string(ctx))  # ( name == test )


result = predicate(ctx)
print(result)
print("here")
print(predicate.to_string())
print(predicate.to_string(ctx))

p2 = (x + 5) == Const[int, Ctx](15)
print(p2(ctx))
print(p2.to_string())
print(p2.to_string(ctx))

p3 = (x + Const[int, Ctx](5)) == (y - Const[int, Ctx](5))
print(p3(ctx))
print(p3.to_string())
print(p3.to_string(ctx))

# divsision and multiplication
p4 = (x * Const[int, Ctx](5)) == (y / Const[int, Ctx](2))
print(p4(ctx))
print(p4.to_string())
print(p4.to_string(ctx))


p5 = Const[int, Ctx](10) + Const[int, Ctx](10)
print(p5(ctx))
print(p5.to_string())
print(p5.to_string(ctx))

p6 = Const[int, Ctx](10) == Const[int, Ctx](10)
print(p6(ctx))
print(p6.to_string())
print(p6.to_string(ctx))
