from dataclasses import dataclass
from typing import Any, ClassVar, Final, Generic, Protocol, TypeVar, overload, runtime_checkable

T = TypeVar("T")
"""The type of the expression's value."""

T_co = TypeVar("T_co", covariant=True)
"""The covariant type of the expression's value."""

S = TypeVar("S")
"""The type of the expression's context."""

S_contra = TypeVar("S_contra", contravariant=True)
"""The contravariant type of the expression's context."""


@runtime_checkable
class Eval(Protocol[T, S_contra]):
    def eval(self, ctx: S_contra) -> "Const[T]": ...


@runtime_checkable
class ToString(Protocol[S_contra]):
    def to_string(self, ctx: S_contra | None = None) -> str: ...


class Expr(Generic[T, S]):
    def eval(self, ctx: S) -> "Const[T]":
        raise NotImplementedError()

    def to_string(self, ctx: S | None = None) -> str:
        raise NotImplementedError()

    def __call__(self, ctx: S) -> "Const[T]":
        return self.eval(ctx)


class BoolExpr(Expr[bool, S]):
    def eval(self, ctx: S) -> "Const[bool]":
        raise NotImplementedError()

    def to_string(self, ctx: S | None = None) -> str:
        return super().to_string(ctx)

    def __call__(self, ctx: S) -> "Const[bool]":
        return self.eval(ctx)


class UnaryOperationOverloads(Expr[bool, S]):
    def __invert__(self) -> "Not[S]":
        return Not(self)


class BooleanBinaryOperationOverloads(BoolExpr[S]):
    def __and__(self, other: BoolExpr[S]) -> "And[S]":
        return And(self, other)

    def __or__(self, other: BoolExpr[S]) -> "Or[S]":
        return Or(self, other)

    def __invert__(self) -> "Not[S]":
        return Not(self)


class BinaryOperationOverloads(Expr[T, S]):
    @overload  # type: ignore[override]
    def __eq__(self, other: Expr[T, S]) -> "Eq[T, S]": ...

    @overload  # type: ignore[override]
    def __eq__(self, other: T) -> "Eq[T, S]": ...

    def __eq__(self, other: Expr[T, S] | T) -> "Eq[T, S]":  # type: ignore[misc]
        if isinstance(other, Expr):
            return Eq(self, other)
        else:
            return Eq(self, Const(None, other))

    @overload  # type: ignore[override]
    def __ne__(self, other: Expr[T, S]) -> "Ne[T, S]": ...

    @overload  # type: ignore[override]
    def __ne__(self, other: T) -> "Ne[T, S]": ...

    def __ne__(self, other: Expr[T, S] | T) -> "Ne[T, S]":  # type: ignore[override]
        if isinstance(other, Expr):
            return Ne(self, other)
        else:
            return Ne(self, Const[T](None, other))

    @overload
    def __lt__(self, other: T) -> "Lt[T, S]": ...

    @overload
    def __lt__(self, other: Expr[T, S]) -> "Lt[T, S]": ...

    def __lt__(self, other: Expr[T, S] | T) -> "Lt[T, S]":
        if isinstance(other, Expr):
            return Lt(self, other)
        else:
            return Lt(self, Const[T](None, other))

    @overload
    def __le__(self, other: T) -> "Le[T, S]": ...

    @overload
    def __le__(self, other: Expr[T, S]) -> "Le[T, S]": ...

    def __le__(self, other: Expr[T, S] | T) -> "Le[T, S]":
        if isinstance(other, Expr):
            return Le(self, other)
        else:
            return Le(self, Const[T](None, other))

    @overload
    def __gt__(self, other: T) -> "Gt[T, S]": ...

    @overload
    def __gt__(self, other: Expr[T, S]) -> "Gt[T, S]": ...

    def __gt__(self, other: Expr[T, S] | T) -> "Gt[T, S]":
        if isinstance(other, Expr):
            return Gt(self, other)
        else:
            return Gt(self, Const[T](None, other))

    @overload
    def __ge__(self, other: T) -> "Ge[T, S]": ...

    @overload
    def __ge__(self, other: Expr[T, S]) -> "Ge[T, S]": ...

    def __ge__(self, other: Expr[T, S] | T) -> "Ge[T, S]":
        if isinstance(other, Expr):
            return Ge(self, other)
        else:
            return Ge(self, Const[T](None, other))

    @overload
    def __add__(self, other: T) -> "Add[T, S]": ...

    @overload
    def __add__(self, other: Expr[T, S]) -> "Add[T, S]": ...

    def __add__(self, other: Expr[T, S] | T) -> "Add[T, S]":
        if isinstance(other, Expr):
            return Add(self, other)
        else:
            return Add(self, Const[T](None, other))

    @overload
    def __sub__(self, other: T) -> "Sub[T, S]": ...

    @overload
    def __sub__(self, other: Expr[T, S]) -> "Sub[T, S]": ...

    def __sub__(self, other: Expr[T, S] | T) -> "Sub[T, S]":
        if isinstance(other, Expr):
            return Sub(self, other)
        else:
            return Sub(self, Const[T](None, other))

    @overload
    def __mul__(self, other: T) -> "Mul[T, S]": ...

    @overload
    def __mul__(self, other: Expr[T, S]) -> "Mul[T, S]": ...

    def __mul__(self, other: Expr[T, S] | T) -> "Mul[T, S]":
        if isinstance(other, Expr):
            return Mul(self, other)
        else:
            return Mul(self, Const[T](None, other))

    @overload
    def __truediv__(self, other: float) -> "Div[float, S]": ...

    @overload
    def __truediv__(self, other: Expr[float, S]) -> "Div[float, S]": ...

    def __truediv__(self, other: Expr[float, S] | float) -> "Div[float, S]":
        if isinstance(other, Expr):
            return Div(self, other)
        else:
            return Div(self, Const[float](None, other))


@dataclass(frozen=True, eq=False)
class Const(BinaryOperationOverloads[T, Any], BooleanBinaryOperationOverloads[Any]):
    name: str | None
    value: T

    def eval(self, ctx: Any) -> "Const[T]":
        return self

    def to_string(self, ctx: Any | None = None) -> str:
        return f"{self.name}:{self.value}" if self.name else str(self.value)


@dataclass(frozen=True, eq=False)
class Var(BinaryOperationOverloads[T, S], BooleanBinaryOperationOverloads[S]):
    name: str

    def eval(self, ctx: S) -> Const[T]:
        return Const(self.name, getattr(ctx, self.name))

    def to_string(self, ctx: S | None = None) -> str:
        if ctx is None:
            return self.name
        else:
            return f"{self.name}:{self.eval(ctx).value}"


@dataclass(frozen=True, eq=False)
class UnaryOpEval(Eval[bool, S]):
    left: Expr[bool, S]


class UnaryOpToString(ToString[S], UnaryOpEval[S]):
    op: ClassVar[str] = "not "
    template: ClassVar[str] = "({op}{left})"
    template_eval: ClassVar[str] = "({op}{left} -> {out})"

    def to_string(self, ctx: S | None = None) -> str:
        left: Final = self.left.to_string(ctx)
        if ctx is None:
            return self.template.format(op=self.op, left=left)
        else:
            return self.template_eval.format(op=self.op, left=left, out=self.eval(ctx).value)


class Not(UnaryOpToString[S], UnaryOperationOverloads[S]):
    op: ClassVar[str] = "not "

    def eval(self, ctx: S) -> Const[bool]:
        return Const(None, not self.left.eval(ctx).value)


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


class And(BinaryOpToString[bool, S], BooleanBinaryOperationOverloads[S], Generic[S]):
    op: ClassVar[str] = " and "

    def eval(self, ctx: S) -> Const[bool]:
        return Const(None, self.left.eval(ctx).value and self.right.eval(ctx).value)


class Or(BinaryOpToString[bool, S], BooleanBinaryOperationOverloads[S], Generic[S]):
    op: ClassVar[str] = " or "

    def eval(self, ctx: S) -> Const[bool]:
        return Const(None, self.left.eval(ctx).value or self.right.eval(ctx).value)


class Eq(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " == "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value == self.right.eval(ctx).value)


class Ne(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " != "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value != self.right.eval(ctx).value)


type SupportsLessThan = int | float


class Lt(
    BinaryOpToString[SupportsLessThan, S],
    BinaryOperationOverloads[SupportsLessThan, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " < "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value < self.right.eval(ctx).value)


class Le(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " <= "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value <= self.right.eval(ctx).value)


class Gt(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " > "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value > self.right.eval(ctx).value)


class Ge(
    BinaryOpToString[T, S],
    BinaryOperationOverloads[T, S],
    BooleanBinaryOperationOverloads[S],
):
    op: ClassVar[str] = " >= "

    def eval(self, ctx: S) -> Const[bool]:  # type: ignore[override]
        return Const(None, self.left.eval(ctx).value >= self.right.eval(ctx).value)


class Add(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " + "

    def eval(self, ctx: S) -> Const[T]:
        return Const(None, self.left.eval(ctx).value + self.right.eval(ctx).value)


class Sub(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " - "

    def eval(self, ctx: S) -> Const[T]:
        return Const(None, self.left.eval(ctx).value - self.right.eval(ctx).value)


class Mul(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " * "

    def eval(self, ctx: S) -> Const[T]:
        return Const(None, self.left.eval(ctx).value * self.right.eval(ctx).value)


class Div(BinaryOpToString[T, S], BinaryOperationOverloads[T, S]):
    op: ClassVar[str] = " / "

    def eval(self, ctx: S) -> Const[T]:
        return Const(None, self.left.eval(ctx).value / self.right.eval(ctx).value)
