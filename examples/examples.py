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

p7 = (x * Const[int, Ctx](2)) + (y / Const[int, Ctx](2)) == Const[int, Ctx](20)
print(p7(ctx))
print(p7.to_string())
print(p7.to_string(ctx))

p8 = Const[int, Ctx](10) / Const[int, Ctx](5)
print(p8(ctx))
print(p8.to_string())
print(p8.to_string(ctx))
