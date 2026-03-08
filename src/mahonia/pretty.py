# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import Any, Final

from mahonia import (
	Abs,
	BinaryOp,
	BoundExpr,
	Expr,
	Func,
	Neg,
	Not,
	Pure,
	Result,
)
from mahonia.match import MatchExpr
from mahonia.predicate import Predicate
from mahonia.python_func import PythonFuncBase
from mahonia.types import ContextProtocol
from mahonia.unary import ClampExpr


def pretty[S: ContextProtocol](
	expr: Expr[Any, S, Any] | Func[Any, S] | Predicate[S],
	ctx: S | None = None,
	*,
	width: int = 80,
	indent: str = "  ",
) -> str:
	"""Pretty-print a mahonia expression tree with width-based line breaking.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> class Ctx(NamedTuple):
	...     x: int
	...     y: int
	>>> x, y = Var[int, Ctx]("x"), Var[int, Ctx]("y")
	>>> pretty(x + y)
	'(x + y)'
	>>> pretty(x + y, Ctx(x=1, y=2))
	'(x:1 + y:2 -> 3)'
	>>> pretty((x + y) * (x - y), Ctx(x=3, y=1), width=30)
	'(\\n  (x:3 + y:1 -> 4)\\n  * (x:3 - y:1 -> 2)\\n  -> 8\\n)'
	"""
	return _fmt(expr, ctx, 0, width, indent)


def _fmt(
	expr: Expr[Any, Any, Any] | Func[Any, Any] | Predicate[Any],
	ctx: Any | None,
	depth: int,
	width: int,
	indent: str,
) -> str:
	flat: Final = expr.to_string(ctx)
	available: Final = width - depth * len(indent)
	if len(flat) <= available:
		return flat

	pad: Final = indent * (depth + 1)
	pad0: Final = indent * depth

	match expr:
		case Pure(inner=inner):
			return _fmt(inner, ctx, depth, width, indent)

		case BoundExpr(expr=inner, ctx=bound_ctx):  # pyright: ignore[reportUnknownVariableType]
			return _fmt(inner, bound_ctx, depth, width, indent)  # pyright: ignore[reportUnknownArgumentType]

		case Result(inner=op_cls, left=left, right=right):
			left_s = _fmt(left, ctx, depth + 1, width, indent)
			right_s = _fmt(right, ctx, depth + 1, width, indent)
			op = op_cls.op.lstrip()
			if ctx is None:
				return f"(\n{pad}{left_s}\n{pad}{op}{right_s}\n{pad0})"
			result = expr.unwrap(ctx)
			return f"(\n{pad}{left_s}\n{pad}{op}{right_s}\n{pad}-> {result}\n{pad0})"

		case BinaryOp(left=left, right=right):
			left_s = _fmt(left, ctx, depth + 1, width, indent)
			right_s = _fmt(right, ctx, depth + 1, width, indent)
			op = expr.op.lstrip()
			if ctx is None:
				return f"(\n{pad}{left_s}\n{pad}{op}{right_s}\n{pad0})"
			result = expr.eval(ctx).value
			return f"(\n{pad}{left_s}\n{pad}{op}{right_s}\n{pad}-> {result}\n{pad0})"

		case Predicate(name=name, expr=inner):
			inner_s = _fmt(inner, ctx, depth, width, indent)
			if ctx is None:
				return f"{name}: {inner_s}" if name else inner_s
			result = expr.unwrap(ctx)
			return f"{name}: {result} {inner_s}" if name else f"{result} {inner_s}"

		case MatchExpr(branches=branches, default=default):

			def strip_outer_parens(s: str) -> str:
				return s[1:-1] if s.startswith("(") and s.endswith(")") else s

			branch_lines = "\n".join(
				f"{pad}({strip_outer_parens(_fmt(cond, ctx, depth + 1, width, indent))} -> {val.to_string()})"
				for cond, val in branches
			)
			if ctx is None:
				return f"(match\n{branch_lines}\n{pad}else {default.to_string()}\n{pad0})"
			result = expr.eval(ctx).value
			return f"(match\n{branch_lines}\n{pad}else {default.to_string(ctx)}\n{pad}-> {result}\n{pad0})"

		case PythonFuncBase(args=args):
			args_parts = [_fmt(a, ctx, depth + 1, width, indent) for a in args]
			args_str = f",\n{pad}".join(args_parts)
			if ctx is None:
				return f"{expr.name}(\n{pad}{args_str}\n{pad0})"
			result = expr.unwrap(ctx)
			return f"{expr.name}(\n{pad}{args_str}\n{pad0}) -> {result}"

		case Func(args=args, expr=body):
			body_s = _fmt(body, ctx, depth + 1, width, indent)
			if len(args) == 0:
				args_s = "()"
			elif len(args) == 1:
				args_s = args[0].to_string()
			else:
				args_s = "(" + ", ".join(a.to_string() for a in args) + ")"
			if ctx is None:
				return f"{args_s} ->\n{pad}{body_s}"
			result = body.unwrap(ctx)
			return f"{args_s} ->\n{pad}{body_s}\n{pad}-> {result}"

		case ClampExpr(lo=lo, hi=hi, value=value):
			val_s = _fmt(value, ctx, depth + 1, width, indent)
			if ctx is None:
				return f"(clamp {lo} {hi}\n{pad}{val_s}\n{pad0})"
			result = expr.eval(ctx).value
			return f"(clamp {lo} {hi}\n{pad}{val_s}\n{pad}-> {result}\n{pad0})"

		case Abs(left=left) | Neg(left=left) | Not(left=left):
			left_s = _fmt(left, ctx, depth + 1, width, indent)
			unary_op: str = expr.op.rstrip()
			if ctx is None:
				return f"({unary_op}\n{pad}{left_s}\n{pad0})"
			result = expr.eval(ctx).value
			return f"({unary_op}\n{pad}{left_s}\n{pad}-> {result}\n{pad0})"

		case _:
			return flat
