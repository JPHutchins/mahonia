from dataclasses import dataclass
from typing import Final, assert_type

import pytest

from cate import Add, Approximately, Const, Eq, Not, Percent, PlusMinus, Var, between


@dataclass(frozen=True)
class Ctx:
    x: int
    y: int
    name: str
    flag: bool = True
    custom: object = object()
    f: float = 1.5


ctx: Final = Ctx(x=5, y=10, name="example")


def test_const_name_eval_and_str() -> None:
    c = Const("Forty-two", 42)
    assert c.value == 42
    assert c.eval(ctx).value == 42
    assert c.to_string() == "Forty-two:42"
    assert c.to_string(ctx) == "Forty-two:42"


def test_const_eval_and_str() -> None:
    c = Const(None, 100)
    assert c.value == 100
    assert c.eval(ctx).value == 100
    assert c.to_string() == "100"
    assert c.to_string(ctx) == "100"


@pytest.mark.mypy_testing
def test_const_generic_type() -> None:
    c_int = Const(None, 1)
    assert_type(c_int, Const[int])
    assert_type(c_int.value, int)

    c_str = Const(None, "test")
    assert_type(c_str, Const[str])
    assert_type(c_str.value, str)

    c_float = Const(None, 1.5)
    assert_type(c_float, Const[float])
    assert_type(c_float.value, float)

    c_complex = Const(None, 1 + 2j)
    assert_type(c_complex, Const[complex])
    assert_type(c_complex.value, complex)

    c_bool = Const(None, True)
    assert_type(c_bool, Const[bool])
    assert_type(c_bool.value, bool)

    class CustomType:
        pass

    c_custom = Const(None, CustomType())
    assert_type(c_custom, Const[CustomType])
    assert_type(c_custom.value, CustomType)


@pytest.mark.mypy_testing
def test_var_generic_type() -> None:
    v_int = Var[int, Ctx]("x")
    assert_type(v_int, Var[int, Ctx])
    assert_type(v_int.eval(ctx).value, int)

    v_str = Var[str, Ctx]("name")
    assert_type(v_str, Var[str, Ctx])
    assert_type(v_str.eval(ctx).value, str)

    v_float = Var[float, Ctx]("y")
    assert_type(v_float, Var[float, Ctx])
    assert_type(v_float.eval(ctx).value, float)

    v_bool = Var[bool, Ctx]("flag")
    assert_type(v_bool, Var[bool, Ctx])
    assert_type(v_bool.eval(ctx).value, bool)

    class CustomType:
        pass

    v_custom = Var[CustomType, Ctx]("custom")
    assert_type(v_custom, Var[CustomType, Ctx])
    assert_type(v_custom.eval(ctx).value, CustomType)


@pytest.mark.mypy_testing
def test_eq_generic_type() -> None:
    v_int = Var[int, Ctx]("x")
    c_int = Const("Five", 5)
    eq_expr = v_int == c_int
    assert_type(eq_expr, Eq[int, Ctx])
    assert_type(eq_expr.eval(ctx).value, bool)

    v_str = Var[str, Ctx]("name")
    c_str = Const("Example", "example")
    eq_expr_str = v_str == c_str
    assert_type(eq_expr_str, Eq[str, Ctx])
    assert_type(eq_expr_str.eval(ctx).value, bool)


@pytest.mark.mypy_testing
def test_add_generic_type() -> None:
    v_int = Var[int, Ctx]("x")
    c_int = Const("Five", 5)
    add_expr = v_int + c_int
    assert_type(add_expr, Add[int, Ctx])
    assert_type(add_expr.eval(ctx).value, int)

    v_float = Var[float, Ctx]("y")
    c_float = Const("Two", 2.0)
    add_expr_float = v_float + c_float
    assert_type(add_expr_float, Add[float, Ctx])
    assert_type(add_expr_float.eval(ctx).value, float)


def test_var_eval_and_str() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    name = Var[str, Ctx]("name")
    assert x.eval(ctx).value == 5
    assert y.eval(ctx).value == 10
    assert name.eval(ctx).value == "example"
    assert x.to_string() == "x"
    assert x.to_string(ctx) == "x:5"
    assert name.to_string(ctx) == "name:example"


def test_add_sub_mul_div() -> None:
    x = Var[float, Ctx]("x")
    y = Var[float, Ctx]("y")
    c5 = Const("Five", 5.0)
    c2 = Const("Two", 2.0)
    assert (x + y).eval(ctx).value == 15
    assert (x + c5).eval(ctx).value == 10
    assert (y - x).eval(ctx).value == 5
    assert (x * c2).eval(ctx).value == 10
    assert (y / c2).eval(ctx).value == 5
    assert (x + y).to_string() == "(x + y)"
    assert (x + c5).to_string(ctx) == "(x:5 + Five:5.0 -> 10.0)"


def test_add() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    assert (x + y).eval(ctx).value == 15
    assert (x + 5).eval(ctx).value == 10
    assert (x + y).to_string() == "(x + y)"
    assert (x + 5).to_string(ctx) == "(x:5 + 5 -> 10)"


def test_comparisons() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    c5 = Const("Five", 5)
    c10 = Const("Ten", 10)
    assert (x == c5).eval(ctx).value is True
    assert (y == c5).eval(ctx).value is False
    assert (x != c5).eval(ctx).value is False
    assert (y != c5).eval(ctx).value is True
    assert (x < y).eval(ctx).value is True
    assert (y > x).eval(ctx).value is True
    assert (x <= c5).eval(ctx).value is True
    assert (y >= c10).eval(ctx).value is True
    assert (x == c5).to_string() == "(x == Five:5)"
    assert (x == c5).to_string(ctx) == "(x:5 == Five:5 -> True)"


def test_logical_ops() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    c5 = Const("Five", 5)
    c10 = Const("Ten", 10)
    pred = ((x == c5) & (y == c10)) | (x != c5)
    assert pred.eval(ctx).value is True
    assert pred.to_string() == "(((x == Five:5) and (y == Ten:10)) or (x != Five:5))"  # noqa: E501
    # Evaluate with context
    assert (
        pred.to_string(ctx)
        == "(((x:5 == Five:5 -> True) and (y:10 == Ten:10 -> True) -> True) or (x:5 != Five:5 -> False) -> True)"  # noqa: E501
    )


def test_not() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("name", 5)
    expr = ~(x == c5)
    assert expr.eval(ctx).value is False
    assert expr.to_string() == "(not (x == name:5))"
    assert expr.to_string(ctx) == "(not (x:5 == name:5 -> True) -> False)"


def test_nested_arithmetic() -> None:
    x = Var[float, Ctx]("x")
    y = Var[float, Ctx]("y")
    expr = (x * Const("name", 2.0)) + (y / Const("name", 2.0))
    assert expr.eval(ctx).value == 15
    assert expr.to_string() == "((x * name:2.0) + (y / name:2.0))"
    assert expr.to_string(ctx) == "((x:5 * name:2.0 -> 10.0) + (y:10 / name:2.0 -> 5.0) -> 15.0)"


def test_constants_only() -> None:
    c10 = Const("name", 10.0)
    c5 = Const("name", 5.0)
    assert (c10 + c5).eval(None).value == 15
    assert (c10 / c5).eval(None).value == 2
    assert (c10 == c10).eval(None).value is True
    assert (c10 == c5).eval(None).value is False
    assert (c10 == 10.0).eval(None).value is True
    assert (c10 != 5.0).eval(None).value is True
    assert (c10 > c5).eval(None).value is True
    assert (c10 + c5 * 50).eval(None).value == 260.0
    assert (c10 + c5 * 50).to_string() == "(name:10.0 + (name:5.0 * 50))"
    assert (c10 + c5 * 50).to_string(ctx) == "(name:10.0 + (name:5.0 * 50 -> 250.0) -> 260.0)"


def test_chained_arithmetic() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    expr = x + y * 2 - 3
    assert expr.eval(ctx).value == 5 + 10 * 2 - 3


def test_const_to_string_edge_cases() -> None:
    c_none = Const(None, None)
    assert c_none.to_string() == "None"
    assert c_none.eval(ctx).value is None


def test_bool_logic() -> None:
    flag = Var[bool, Ctx]("flag")
    c_true = Const("True", True)
    c_false = Const("False", False)
    assert (flag & c_true).eval(ctx).value is True
    assert (flag | c_false).eval(ctx).value is True
    assert (~flag).eval(ctx).value is False


def test_const_vs_python_literal() -> None:
    c10 = Const("name", 10.0)
    expr = c10 == 10.0
    assert expr.eval(None).value is True
    assert (c10 != 5.0).eval(None).value is True
    assert (c10 > 5.0).eval(None).value is True
    assert (c10 < 20.0).eval(None).value is True


def test_var_add_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x + c5).eval(ctx).value == 10
    assert (x + 2).eval(ctx).value == 7


def test_const_add_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 + x).eval(ctx).value == 10
    assert (c5 + 2).eval(ctx).value == 7


def test_var_sub_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x - c5).eval(ctx).value == 0
    assert (x - 2).eval(ctx).value == 3


def test_const_sub_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 - x).eval(ctx).value == 0
    assert (c5 - 2).eval(ctx).value == 3


def test_var_mul_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x * c5).eval(ctx).value == 25
    assert (x * 2).eval(ctx).value == 10


def test_const_mul_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 * x).eval(ctx).value == 25
    assert (c5 * 2).eval(ctx).value == 10


def test_var_truediv_const_and_literal() -> None:
    y = Var[float, Ctx]("y")
    c5 = Const("Five", 5.0)
    assert (y / c5).eval(ctx).value == 2
    assert (y / 2.0).eval(ctx).value == 5


def test_const_truediv_var_and_literal() -> None:
    y = Var[float, Ctx]("y")
    c10 = Const("Ten", 10)
    assert (c10 / y).eval(ctx).value == 1
    assert (c10 / 2).eval(ctx).value == 5


def test_var_eq_const() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x == c5).eval(ctx).value is True


def test_const_eq_var() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 == x).eval(ctx).value is True


def test_var_ne_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x != c5).eval(ctx).value is False
    assert (x != 7).eval(ctx).value is True


def test_const_ne_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 != x).eval(ctx).value is False
    assert (c5 != 7).eval(ctx).value is True


def test_var_lt_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c10 = Const("Ten", 10)
    assert (x < c10).eval(ctx).value is True
    assert (x < 3).eval(ctx).value is False


def test_const_lt_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c3 = Const("Three", 3)
    assert (c3 < x).eval(ctx).value is True
    assert (c3 < 2).eval(ctx).value is False


def test_var_le_const_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (x <= c5).eval(ctx).value is True
    assert (x <= 4).eval(ctx).value is False


def test_const_le_var_and_literal() -> None:
    x = Var[int, Ctx]("x")
    c5 = Const("Five", 5)
    assert (c5 <= x).eval(ctx).value is True
    assert (c5 <= 4).eval(ctx).value is False


def test_var_gt_const_and_literal() -> None:
    y = Var[int, Ctx]("y")
    c5 = Const("Five", 5)
    assert (y > c5).eval(ctx).value is True
    assert (y > 20).eval(ctx).value is False


def test_const_gt_var_and_literal() -> None:
    y = Var[int, Ctx]("y")
    c20 = Const("Twenty", 20)
    assert (c20 > y).eval(ctx).value is True
    assert (c20 > 30).eval(ctx).value is False


def test_var_ge_const_and_literal() -> None:
    y = Var[int, Ctx]("y")
    c10 = Const("Ten", 10)
    assert (y >= c10).eval(ctx).value is True
    assert (y >= 20).eval(ctx).value is False


def test_const_ge_var_and_literal() -> None:
    y = Var[int, Ctx]("y")
    c10 = Const("Ten", 10)
    assert (c10 >= y).eval(ctx).value is True
    assert (c10 >= 20).eval(ctx).value is False


def test_var_in_range() -> None:
    min = Const("min", 0)
    max = Const("max", 10)
    x = Var[int, Ctx]("x")

    expr = (min <= x) & (x <= max)
    assert expr.eval(ctx).value is True
    assert expr.to_string() == "((min:0 <= x) and (x <= max:10))"
    assert expr.to_string(ctx) == "((min:0 <= x:5 -> True) and (x:5 <= max:10 -> True) -> True)"

    expr = (max >= x) & (x >= min)
    assert expr.eval(ctx).value is True
    assert expr.to_string() == "((max:10 >= x) and (x >= min:0))"
    assert expr.to_string(ctx) == "((max:10 >= x:5 -> True) and (x:5 >= min:0 -> True) -> True)"

    expr = (min < x) & (x < max)
    assert expr.eval(ctx).value is True
    assert expr.to_string() == "((min:0 < x) and (x < max:10))"
    assert expr.to_string(ctx) == "((min:0 < x:5 -> True) and (x:5 < max:10 -> True) -> True)"

    expr = (max > x) & (x > min)
    assert expr.eval(ctx).value is True
    assert expr.to_string() == "((max:10 > x) and (x > min:0))"
    assert expr.to_string(ctx) == "((max:10 > x:5 -> True) and (x:5 > min:0 -> True) -> True)"


def test_deeply_nested_all_operations() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")
    f = Var[float, Ctx]("f")
    c5 = Const("Five", 5)
    c10 = Const("Ten", 10)
    c2 = Const("Two", 2)
    c3 = Const("Three", 3)
    c7 = Const("Seven", 7.0)
    c13 = Const("Thirteen", 13.0)

    expr = ~(
        (((x + c5) * (y - c2)) == c3)
        & ((x >= c5) | (y < c10))
        & ((x != 7) & (y <= 20))
        & ((x > 0) & (y > 0))
        | ((f / c7 + c7) < c13)
    )

    assert_type(expr, Not[Ctx])

    result = expr.eval(ctx).value
    assert isinstance(result, bool)

    s = expr.to_string()
    print(s)
    assert (
        s
        == "(not (((((((x + Five:5) * (y - Two:2)) == Three:3) and ((x >= Five:5) or (y < Ten:10))) and ((x != 7) and (y <= 20))) and ((x > 0) and (y > 0))) or (((f / Seven:7.0) + Seven:7.0) < Thirteen:13.0)))"
    )

    s_ctx = expr.to_string(ctx)
    print(s_ctx)
    assert (
        s_ctx
        == "(not (((((((x:5 + Five:5 -> 10) * (y:10 - Two:2 -> 8) -> 80) == Three:3 -> False) and ((x:5 >= Five:5 -> True) or (y:10 < Ten:10 -> False) -> True) -> False) and ((x:5 != 7 -> True) and (y:10 <= 20 -> True) -> True) -> False) and ((x:5 > 0 -> True) and (y:10 > 0 -> True) -> True) -> False) or (((f:1.5 / Seven:7.0 -> 0.21428571428571427) + Seven:7.0 -> 7.214285714285714) < Thirteen:13.0 -> True) -> True) -> False)"  # noqa: E501
    )

    assert result is False

    # expression is immutable, so multiple evaluations should yield the same result
    for _ in range(5):
        assert expr.eval(ctx).value is False
        assert expr.to_string() == s
        assert expr.to_string(ctx) == s_ctx
        assert expr.to_string(ctx) == s_ctx


def test_between() -> None:
    x = Var[int, Ctx]("x")

    expr = between(x, 5, 10)
    assert expr.eval(ctx).value is False
    assert expr.to_string() == "((Low:5 < x) and (x < High:10))"
    assert expr.to_string(ctx) == "((Low:5 < x:5 -> False) and (x:5 < High:10 -> True) -> False)"

    expr = between(x, 0, 10)
    assert expr.eval(ctx).value is True
    assert expr.to_string() == "((Low:0 < x) and (x < High:10))"
    assert expr.to_string(ctx) == "((Low:0 < x:5 -> True) and (x:5 < High:10 -> True) -> True)"

    expr = (Const("Low", 5) < x) & (x < Const("High", 10))
    print()
    print(expr.to_string())
    print()
    print(expr.to_string(ctx))


def test_manual_within() -> None:
    f = Var[float, Ctx]("f")

    expr = (f - Const("Target", 1.5)) < Const("Tolerance", 0.01)
    print()
    print(expr.to_string())
    print()
    print(expr.to_string(ctx))


def test_approximately() -> None:
    x = Var[float, Ctx]("x")

    FIVE = PlusMinus("Five", 4.9, 0.1)
    assert_type(FIVE, PlusMinus[float])

    expr = Approximately(x, FIVE)
    assert_type(expr, Approximately[float, Ctx])
    print()
    print(expr.to_string())
    print(expr.to_string(ctx))

    FIVE_ = Percent("Five", 5.0, 1.0)
    assert_type(FIVE_, Percent[float])

    expr = Approximately(x, FIVE_)
    assert_type(expr, Approximately[float, Ctx])

    print()
    print(expr.to_string())
    print(expr.to_string(ctx))


def test_composition() -> None:
    x = Var[int, Ctx]("x")
    y = Var[int, Ctx]("y")

    x_plus_y = x + y
    SUM = PlusMinus("Sum", 15, 0.1)

    expr = Approximately(x_plus_y, SUM)
    assert_type(expr, Approximately[int, Ctx])

    print()
    print(expr.to_string())
    print(expr.to_string(ctx))

    expr = Approximately(x * y, Percent("Product", 50.0, 5.0))
    assert_type(expr, Approximately[int, Ctx])

    print()
    print(expr.to_string())
    print(expr.to_string(ctx))
