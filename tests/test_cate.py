from dataclasses import dataclass
from cate import Var, Const, eq

def test_cate() -> None:
    # Example usage
    @dataclass
    class Ctx:
        x: int
        y: int
        name: str

    # Create predicates using the DSL
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    name = Var[str, Ctx]("name")

    # Build complex predicates
    predicate = (eq(x, Const(5)) & eq(y, Const(10))) | eq(name, Const("test"))

    # Evaluate
    ctx = Ctx(x=5, y=10, name="example")
    result = predicate.eval(ctx)  # True

    # String representations
    print(predicate.to_string())  # ((x = 5) ∧ (y = 10)) ∨ (name = test)
    print(predicate.to_string(ctx))  # ((x=5 = 5) ∧ (y=10 = 10)) ∨ (name=example = test)s