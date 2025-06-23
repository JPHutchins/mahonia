from typing import TypeVar, Protocol, NamedTuple, Generic, Final, overload, runtime_checkable

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class Expr(Protocol[T, S]):
    def eval(self, ctx: S) -> "Expr[T, S]": ...

    def to_string(self, ctx: S | None = None) -> str: ...

    def __and__(self, other: "Expr[T, S]") -> "And[T, S]": ...

    def __or__(self, other: "Expr[T, S]") -> "Or[T, S]": ...

    def __invert__(self) -> "Not[T, S]": ...

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]": ...  # type: ignore[override]


class And(NamedTuple, Generic[T, S]):
    exprs: tuple[Expr[T, S], ...]

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const[bool, S](all(e.eval(ctx) for e in self.exprs))

    def to_string(self, ctx: S | None = None) -> str:
        if len(self.exprs) == 1:
            return self.exprs[0].to_string(ctx)
        expr_strs = [e.to_string(ctx) for e in self.exprs]
        return f"( {' and '.join(expr_strs)} )"

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)


class Or(NamedTuple, Generic[T, S]):
    exprs: tuple[Expr[T, S], ...]

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const[bool, S](any(e.eval(ctx) for e in self.exprs))

    def to_string(self, ctx: S | None = None) -> str:
        if len(self.exprs) == 1:
            return self.exprs[0].to_string(ctx)
        expr_strs = [e.to_string(ctx) for e in self.exprs]
        return f"( {' or '.join(expr_strs)} )"

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)


class Not(NamedTuple, Generic[T, S]):
    expr: Expr[T, S]

    def eval(self, ctx: S) -> "Const[bool, S]":
        return Const[bool, S](not self.expr.eval(ctx))

    def to_string(self, ctx: S | None = None) -> str:
        return f"( not {self.expr.to_string(ctx)} )"

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)


class Var(NamedTuple, Generic[T, S]):
    name: str

    def eval(self, ctx: S) -> "Const[T, S]":
        return Const(getattr(ctx, self.name))

    def to_string(self, ctx: S | None = None) -> str:
        if ctx is not None:
            try:
                value = getattr(ctx, self.name)
                return f"{self.name}: {value}"
            except AttributeError:
                pass
        return self.name

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)
    
    @overload
    def __add__(self, other: T) -> "Add[T, S]": ...

    @overload
    def __add__(self, other: Expr[T, S]) -> "Add[T, S]": ...
    
    def __add__(self, other: Expr[T, S] | T) -> "Add[T, S]":
        if isinstance(other, Expr):
            return Add(self, other)
        else:
            return Add(self, Const[T, S](other))


class Const(NamedTuple, Generic[T, S]):
    value: T

    def eval(self, ctx: S) -> "Const[T, S]":
        return self

    def to_string(self, ctx: S | None = None) -> str:
        return str(self.value)

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)
    
    def __add__(self: Expr[T, S], other: Expr[T, S]) -> "Add[T, S]":
        return Add(self, other)
    

class Add(NamedTuple, Generic[T, S]):
    left: Expr[T, S]
    right: Expr[T, S]

    def eval(self, ctx: S) -> Const[T, S]:
        return Const[T, S](self.left.eval(ctx).value + self.right.eval(ctx).value)
    
    def to_string(self, ctx: S | None = None) -> str:
        left_str = self.left.to_string(ctx)
        right_str = self.right.to_string(ctx)
        return f"( {left_str} + {right_str} )"
    
    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)


class Eq(NamedTuple, Generic[T, S]):
    left: Expr[T, S]
    right: Expr[T, S]

    def eval(self, ctx: S) -> Const[bool, S]:
        return Const[bool, S](self.left.eval(ctx).value == self.right.eval(ctx).value)

    def to_string(self, ctx: S | None = None) -> str:
        left_str = self.left.to_string(ctx)
        right_str = self.right.to_string(ctx)
        return f"( {left_str} = {right_str} )"

    def __and__(self: Expr[T, S], other: Expr[T, S]) -> "And[T, S]":
        return And((self, other))

    def __or__(self: Expr[T, S], other: Expr[T, S]) -> "Or[T, S]":
        return Or((self, other))

    def __invert__(self: Expr[T, S]) -> "Not[T, S]":
        return Not(self)

    def __eq__(self, other: "Expr[T, S]") -> "Eq[T, S]":  # type: ignore[override]
        return Eq(self, other)



class Ctx(NamedTuple):
    x: int
    y: int
    name: str


x = Var[int, Ctx]("x")
y = Var[int, Ctx]("y")
name = Var[str, Ctx]("name")

# Build complex predicates
predicate = (
    (x == Const[int, Ctx](5))
    | (y == Const[int, Ctx](10))
    | (name == Const[str, Ctx]("test"))
)

# Evaluate
ctx = Ctx(x=5, y=10, name="example")
result = predicate.eval(ctx)  # True
print(result)

# String representations
print(predicate.to_string())
print(predicate.to_string(ctx))

TEST: Final = Const[str, Ctx]("test")

p = name == TEST
print(p.eval(ctx))  # True
print(p.to_string())  # ( name == test )
print(p.to_string(ctx))  # ( name == test )

EXAMPLE: Final = Const[str, Ctx]("example")

# test == operator
p = name == EXAMPLE
print(p.eval(ctx))  # True
print(p.to_string())  # ( name == test )
print(p.to_string(ctx))  # ( name == test )

# test addition
p2 = (((x + 5) == Const[int, Ctx](15)) and Const[int, Ctx](10))
print(p2.eval(ctx))  # 15
print(p2.to_string())  # ( x + y )
print(p2.to_string(ctx))  # ( x + y