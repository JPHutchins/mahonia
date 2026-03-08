# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from textwrap import dedent
from typing import Final, NamedTuple

from mahonia import Const, Func, Match, Predicate, Var
from mahonia.pretty import pretty
from mahonia.python_func import python_func
from mahonia.unary import Clamp


class Ctx(NamedTuple):
	x: int
	y: int
	f: float


ctx = Ctx(x=5, y=10, f=1.5)

x_ctx = Var[int, Ctx]("x")
y_ctx = Var[int, Ctx]("y")
f_ctx = Var[float, Ctx]("f")

deeply_nested: Final = ~(
	(((x_ctx + Const("Five", 5)) * (y_ctx - Const("Two", 2))) == Const("Three", 3))
	& ((x_ctx >= Const("Five", 5)) | (y_ctx < Const("Ten", 10)))
	& ((x_ctx != 7) & (y_ctx <= 20))
	& ((x_ctx > 0) & (y_ctx > 0))
	| ((f_ctx / Const("Seven", 7.0) + Const("Seven", 7.0)) < Const("Thirteen", 13.0))
)


class FizzBuzzCtx(NamedTuple):
	n: int


n = Var[int, FizzBuzzCtx]("n")

fizzbuzz: Final = Match(
	((n % 15) == 0, Const("fb", "FizzBuzz")),
	((n % 3) == 0, Const("f", "Fizz")),
	((n % 5) == 0, Const("b", "Buzz")),
	default=n,
)


def test_leaf_unchanged() -> None:
	assert pretty(x_ctx) == "x"
	assert pretty(x_ctx, ctx) == "x:5"
	assert pretty(Const("Five", 5)) == "Five:5"


def test_flat_when_fits() -> None:
	assert pretty(x_ctx + y_ctx) == "(x + y)"
	assert pretty(x_ctx + y_ctx, ctx) == "(x:5 + y:10 -> 15)"


def test_binary_breaks_with_context() -> None:
	assert pretty((x_ctx + y_ctx) * (x_ctx - y_ctx), ctx, width=30) == dedent("""\
		(
		  (x:5 + y:10 -> 15)
		  * (x:5 - y:10 -> -5)
		  -> -75
		)""")


def test_binary_breaks_without_context() -> None:
	c5 = Const("Five", 5)
	expr = (x_ctx + c5) * (y_ctx - c5) + c5
	assert pretty(expr, width=39) == dedent("""\
		(
		  ((x + Five:5) * (y - Five:5))
		  + Five:5
		)""")


def test_not_breaks() -> None:
	expr = ~(x_ctx > y_ctx)
	assert pretty(expr, ctx, width=30) == dedent("""\
		(not
		  (x:5 > y:10 -> False)
		  -> True
		)""")


def test_width_controls_breaking() -> None:
	expr = (x_ctx + y_ctx) * (x_ctx - y_ctx)
	assert pretty(expr, ctx, width=200) == expr.to_string(ctx)
	assert "\n" in pretty(expr, ctx, width=30)


def test_deeply_nested_no_context() -> None:
	assert pretty(deeply_nested) == dedent("""\
		(not
		  (
		    (
		      (
		        (
		          (((x + Five:5) * (y - Two:2)) == Three:3)
		          & ((x >= Five:5) | (y < Ten:10))
		        )
		        & ((x != 7) & (y <= 20))
		      )
		      & ((x > 0) & (y > 0))
		    )
		    | (((f / Seven:7.0) + Seven:7.0) < Thirteen:13.0)
		  )
		)""")


def test_deeply_nested_with_context() -> None:
	assert pretty(deeply_nested, ctx) == dedent("""\
		(not
		  (
		    (
		      (
		        (
		          (
		            ((x:5 + Five:5 -> 10) * (y:10 - Two:2 -> 8) -> 80)
		            == Three:3
		            -> False
		          )
		          & ((x:5 >= Five:5 -> True) | (y:10 < Ten:10 -> False) -> True)
		          -> False
		        )
		        & ((x:5 != 7 -> True) & (y:10 <= 20 -> True) -> True)
		        -> False
		      )
		      & ((x:5 > 0 -> True) & (y:10 > 0 -> True) -> True)
		      -> False
		    )
		    | (
		      (
		        (f:1.5 / Seven:7.0 -> 0.21428571428571427)
		        + Seven:7.0
		        -> 7.214285714285714
		      )
		      < Thirteen:13.0
		      -> True
		    )
		    -> True
		  )
		  -> False
		)""")


def test_result_division() -> None:
	x = Var[float, Ctx]("x")
	y = Var[float, Ctx]("y")
	assert pretty(x / y, width=200) == "(x / y)"


def test_pure_transparent() -> None:
	from mahonia import Pure

	assert pretty(Pure(x_ctx)) == "x"
	assert pretty(Pure(x_ctx), ctx) == "x:5"


def test_bound_expr() -> None:
	bound = ((x_ctx + y_ctx) * (x_ctx - y_ctx)).bind(ctx)
	assert pretty(bound, width=30) == dedent("""\
		(
		  (x:5 + y:10 -> 15)
		  * (x:5 - y:10 -> -5)
		  -> -75
		)""")


def test_predicate_no_context() -> None:
	pred = Predicate("check", (x_ctx + y_ctx) * (x_ctx - y_ctx) > Const("Limit", 50))
	assert pretty(pred, width=30) == dedent("""\
		check: (
		  ((x + y) * (x - y))
		  > Limit:50
		)""")


def test_predicate_with_context() -> None:
	pred = Predicate("check", (x_ctx + y_ctx) * (x_ctx - y_ctx) > Const("Limit", 50))
	assert pretty(pred, ctx, width=30) == dedent("""\
		check: False (
		  (
		    (x:5 + y:10 -> 15)
		    * (x:5 - y:10 -> -5)
		    -> -75
		  )
		  > Limit:50
		  -> False
		)""")


def _double(x: float) -> float:
	return x * 2


def _safe_sqrt(x: float) -> float:
	if x < 0:
		raise ValueError(f"expected non-negative, got {x}")
	return float(x**0.5)


double: Final = python_func(_double)
safe_sqrt: Final = python_func(_safe_sqrt)


def test_python_func() -> None:
	x_f = Var[float, Ctx]("x")
	expr = double(x_f)
	assert pretty(expr, Ctx(x=4, y=0, f=0.0), width=12) == dedent("""\
		_double(
		  x:4
		) -> 8""")
	assert pretty(expr, width=9) == dedent("""\
		_double(
		  x
		)""")


class FloatCtx(NamedTuple):
	x: float
	y: float


def _add(a: float, b: float) -> float:
	return a + b


add: Final = python_func(_add)


def test_python_func_multi_arg() -> None:
	x_f = Var[float, FloatCtx]("x")
	y_f = Var[float, FloatCtx]("y")
	expr = add(x_f, y_f)
	assert pretty(expr, FloatCtx(x=1.0, y=2.0), width=15) == dedent("""\
		_add(
		  x:1.0,
		  y:2.0
		) -> 3.0""")
	assert pretty(expr, width=9) == dedent("""\
		_add(
		  x,
		  y
		)""")


def test_python_func_failure() -> None:
	x_f = Var[float, FloatCtx]("x")
	y_f = Var[float, FloatCtx]("y")
	expr = safe_sqrt(x_f) + safe_sqrt(y_f)
	fail = "Failure(exceptions=(ValueError('expected non-negative, got -1.0'),))"
	assert pretty(expr, FloatCtx(x=-1.0, y=4.0), width=40) == dedent(f"""\
		(
		  _safe_sqrt(
		    x:-1.0
		  ) -> {fail}
		  + _safe_sqrt(y:4.0) -> 2.0
		  -> {fail}
		)""")


def test_func() -> None:
	func = (x_ctx + y_ctx).to_func()
	assert pretty(func, width=10) == dedent("""\
		(x, y) ->
		  (x + y)""")


def test_clamp() -> None:
	clamp_expr = Clamp(0, 100)(x_ctx + y_ctx)
	assert pretty(clamp_expr, width=20) == dedent("""\
		(clamp 0 100
		  (x + y)
		)""")
	assert pretty(clamp_expr, ctx, width=20) == dedent("""\
		(clamp 0 100
		  (x:5 + y:10 -> 15)
		  -> 15
		)""")


def test_fizzbuzz_no_context() -> None:
	assert pretty(fizzbuzz) == dedent("""\
		(match
		  ((n % 15) == 0 -> fb:FizzBuzz)
		  ((n % 3) == 0 -> f:Fizz)
		  ((n % 5) == 0 -> b:Buzz)
		  else n
		)""")


def test_fizzbuzz_with_context() -> None:
	assert pretty(fizzbuzz, FizzBuzzCtx(n=15)) == dedent("""\
		(match
		  ((n:15 % 15 -> 0) == 0 -> True -> fb:FizzBuzz)
		  ((n:15 % 3 -> 0) == 0 -> True -> f:Fizz)
		  ((n:15 % 5 -> 0) == 0 -> True -> b:Buzz)
		  else n:15
		  -> FizzBuzz
		)""")

	assert pretty(fizzbuzz, FizzBuzzCtx(n=7)) == dedent("""\
		(match
		  ((n:7 % 15 -> 7) == 0 -> False -> fb:FizzBuzz)
		  ((n:7 % 3 -> 1) == 0 -> False -> f:Fizz)
		  ((n:7 % 5 -> 2) == 0 -> False -> b:Buzz)
		  else n:7
		  -> 7
		)""")


def test_func_single_arg() -> None:
	func = x_ctx.to_func()
	assert pretty(func, width=3) == dedent("""\
		x ->
		  x""")


def test_func_zero_args() -> None:
	func = Func((), Const("pi", 3.14))
	assert pretty(func, width=5) == dedent("""\
		() ->
		  pi:3.14""")


def test_func_with_context() -> None:
	func = (x_ctx + y_ctx).to_func()
	assert pretty(func, ctx, width=20) == dedent("""\
		(x, y) ->
		  (x:5 + y:10 -> 15)
		  -> 15""")


def test_fallback_returns_flat() -> None:
	assert pretty(Const("LongName", 12345), width=5) == "LongName:12345"
