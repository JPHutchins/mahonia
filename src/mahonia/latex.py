# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""LaTeX serialization for mahonia expressions.

This module provides functionality to convert mahonia expressions to LaTeX
mathematical notation.

>>> from typing import NamedTuple
>>> class Ctx(NamedTuple):
...     x: int
...     y: int
>>> from mahonia import Var, Const
>>> x = Var[int, Ctx]("x")
>>> y = Var[int, Ctx]("y")
>>> expr = x + y * 2
>>> latex(expr)
'x + y \\\\cdot 2'
>>> latex(x > 5)
'x > 5'
"""

from enum import Flag, auto
from typing import Any, Final, Generic, NamedTuple, assert_never

from mahonia import (
	Add,
	And,
	Approximately,
	Const,
	Div,
	Eq,
	Expr,
	Ge,
	Gt,
	Le,
	Lt,
	Mul,
	Ne,
	Not,
	Or,
	Percent,
	PlusMinus,
	Pow,
	Predicate,
	S,
	Sub,
	Var,
)
from mahonia.stats import Count, Mean, Median, Percentile, Range, StdDev

type BinaryOpExpr = (
	Eq[Any, Any]
	| Ne[Any, Any]
	| Lt[Any, Any]
	| Le[Any, Any]
	| Gt[Any, Any]
	| Ge[Any, Any]
	| And[Any, Any]
	| Or[Any, Any]
)


class Show(Flag):
	"""Display options for latex evaluation.

	Assuming that x is 2 and y is 3:
	- (none): `x + y`
	- VALUES: `x:2 + y:3`
	- WORK: `(x + y \\rightarrow 5)`
	- VALUES | WORK: `(x:2 + y:3 \\rightarrow 5)`

	Examples:
	>>> from typing import NamedTuple
	>>> class TestCtx(NamedTuple):
	...     x: int
	...     y: int
	>>> x = Var[int, TestCtx]("x")
	>>> y = Var[int, TestCtx]("y")
	>>> test_ctx = TestCtx(x=2, y=3)
	>>> expr = x + y
	>>> latex(expr, LatexCtx(test_ctx, Show.VALUES))
	'(x:2 + y:3 \\\\rightarrow 5)'
	>>> latex(expr, LatexCtx(test_ctx, Show.WORK))
	'(x + y \\\\rightarrow 5)'
	>>> latex(expr, LatexCtx(test_ctx, Show.VALUES | Show.WORK))
	'(x:2 + y:3 \\\\rightarrow 5)'
	"""

	VALUES = auto()
	"""Add values to variables: `name:<val>."""
	WORK = auto()
	"""Show the evaluated result of the expression."""


class LatexCtx(NamedTuple, Generic[S]):
	ctx: S
	show: Show = Show.VALUES | Show.WORK


def latex(expr: Expr[Any, S], ctx: LatexCtx[S] | None = None) -> str:
	"""Convert a mahonia expression to LaTeX mathematical notation.

	Examples:
	>>> from typing import NamedTuple
	>>> class TestCtx(NamedTuple):
	...     x: float
	...     y: float
	>>> x = Var[float, TestCtx]("x")
	>>> y = Var[float, TestCtx]("y")
	>>> latex(x + y)
	'x + y'
	>>> latex(x * y)
	'x \\\\cdot y'
	>>> latex(x / y)
	'\\\\frac{x}{y}'
	>>> latex(x**2 + y**2)
	'x^2 + y^2'
	>>> # With context - default shows values and work
	>>> test_ctx = TestCtx(x=2.0, y=3.0)
	>>> latex(x + y, LatexCtx(test_ctx))
	'(x:2.0 + y:3.0 \\\\rightarrow 5.0)'
	>>> # Show only values
	>>> latex(x + y, LatexCtx(test_ctx, Show.VALUES))
	'(x:2.0 + y:3.0 \\\\rightarrow 5.0)'
	>>> # Show only work
	>>> latex(x + y, LatexCtx(test_ctx, Show.WORK))
	'(x + y \\\\rightarrow 5.0)'
	>>> # Show nothing (structure only)
	>>> latex(x + y, LatexCtx(test_ctx, Show(0)))
	'(x + y \\\\rightarrow 5.0)'
	>>> # Boolean expressions
	>>> latex(x > y, LatexCtx(test_ctx, Show.VALUES | Show.WORK))
	'(x:2.0 > y:3.0 \\\\rightarrow \\\\text{False})'
	>>> # Complex expressions
	>>> latex((x + y) / 2, LatexCtx(test_ctx, Show.VALUES))
	'(\\\\frac{x:2.0 + y:3.0}{2} \\\\rightarrow 2.5)'
	"""
	match ctx:
		case None:
			return _latex_expr_structure(expr)
		case LatexCtx(ctx_data, show) if show & Show.WORK:
			structure = _latex_expr_structure(expr, ctx)
			return (
				structure
				if "\\rightarrow" in structure
				else _format_with_result(structure, expr.eval(ctx_data))
			)
		case LatexCtx(ctx_data, _):
			return _format_with_result(_latex_expr_structure(expr, ctx), expr.eval(ctx_data))
		case _:
			assert_never(ctx)


def _latex_value(value: Any) -> str:
	"""Convert a value to LaTeX format."""
	if isinstance(value, bool):
		return f"\\text{{{str(value)}}}"
	return str(value)


def _format_with_result(structure: str, result: Const[Any]) -> str:
	"""Format structure with result arrow notation."""
	result_latex = (
		_latex_expr_structure(result)
		if isinstance(result, (PlusMinus, Percent))
		else _latex_value(result.value)
	)
	return f"({structure} \\rightarrow {result_latex})"


def _latex_expr_structure(expr: Expr[Any, Any], latex_ctx: LatexCtx[Any] | None = None) -> str:
	"""Convert expression structure to LaTeX, optionally showing variable values."""
	match expr:
		case Var(name=name):
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.VALUES:
					evaluated = expr.eval(ctx)
					return f"{_latex_var(name)}:{evaluated.value}"
				case _:
					return _latex_var(name)
		case PlusMinus(name=name, value=value, plus_minus=pm):
			return f"{_latex_var(name) if name else str(value)} \\pm {pm}"
		case Percent(name=name, value=value, percent=pct):
			return f"{_latex_var(name) if name else str(value)} \\pm {pct}\\%"
		case Const(name=name, value=value) if name:
			match latex_ctx:
				case LatexCtx(_, show) if show & Show.VALUES:
					return f"{_latex_var(name)}:{value}"
				case _:
					return _latex_var(name)
		case Const(value=value):
			return _latex_value(value)
		case Add():
			left = _latex_expr_structure(expr.left, latex_ctx)
			right = _latex_expr_structure(expr.right, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({left} + {right} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"{left} + {right}"
		case Sub():
			left = _latex_expr_structure(expr.left, latex_ctx)
			right = _latex_expr_structure(expr.right, latex_ctx)
			right_formatted = f"({right})" if _needs_parentheses(expr.right, expr) else right
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({left} - {right_formatted} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"{left} - {right_formatted}"
		case Mul():
			left = _latex_expr_structure(expr.left, latex_ctx)
			right = _latex_expr_structure(expr.right, latex_ctx)
			left_formatted = f"({left})" if _needs_parentheses(expr.left, expr) else left
			right_formatted = f"({right})" if _needs_parentheses(expr.right, expr) else right
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({left_formatted} \\cdot {right_formatted} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"{left_formatted} \\cdot {right_formatted}"
		case Div():
			left = _latex_expr_structure(expr.left, latex_ctx)
			right = _latex_expr_structure(expr.right, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\frac{{{left}}}{{{right}}} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\frac{{{left}}}{{{right}}}"
		case Pow():
			base = _latex_expr_structure(expr.left, latex_ctx)
			power = _latex_expr_structure(expr.right, latex_ctx)
			formatted_base = f"({base})" if _needs_parentheses(expr.left, expr) else base
			power_formatted = (
				f"{formatted_base}^{{{power}}}"
				if len(power) > 1 or not power.isdigit()
				else f"{formatted_base}^{power}"
			)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({power_formatted} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return power_formatted
		case Eq() | Ne() | Lt() | Le() | Gt() | Ge() | And() | Or() as binary_op:
			left = _latex_expr_structure(binary_op.left, latex_ctx)
			right = _latex_expr_structure(binary_op.right, latex_ctx)
			expr_str = f"{left} {LATEX_OP[type(binary_op)]} {right}"
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({expr_str} \\rightarrow {_latex_value(binary_op.eval(ctx).value)})"
				case _:
					return expr_str
		case Not():
			operand = _latex_expr_structure(expr.left, latex_ctx)
			operand_formatted = f"({operand})" if _needs_parentheses(expr.left, expr) else operand
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\neg {operand_formatted} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\neg {operand_formatted}"
		case Approximately():
			left = _latex_expr_structure(expr.left, latex_ctx)
			right = _latex_expr_structure(expr.right, latex_ctx)
			expr_str = f"{left} \\approx {right}"
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({expr_str} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return expr_str
		case Predicate(name=name, expr=pred_expr):
			expr_latex = _latex_expr_structure(pred_expr, latex_ctx)
			pred_str = f"\\text{{{name}}}: {expr_latex}" if name else expr_latex
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({pred_str} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return pred_str
		case Mean():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\bar{{{operand_formatted}}} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\bar{{{operand_formatted}}}"
		case StdDev():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\sigma_{{{operand_formatted}}} \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\sigma_{{{operand_formatted}}}"
		case Median():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\text{{median}}({operand_formatted}) \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\text{{median}}({operand_formatted})"
		case Percentile():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			percentile_str = (
				f"P_{{{int(expr.percentile) if expr.percentile.is_integer() else expr.percentile}}}"
			)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"({percentile_str}({operand_formatted}) \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"{percentile_str}({operand_formatted})"
		case Range():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return f"(\\text{{range}}({operand_formatted}) \\rightarrow {_latex_value(expr.eval(ctx).value)})"
				case _:
					return f"\\text{{range}}({operand_formatted})"
		case Count():
			operand_formatted = _format_statistical_operand(expr, latex_ctx)
			match latex_ctx:
				case LatexCtx(ctx, show) if show & Show.WORK:
					return (
						f"(|{operand_formatted}| \\rightarrow {_latex_value(expr.eval(ctx).value)})"
					)
				case _:
					return f"|{operand_formatted}|"
		case _:
			return f"\\text{{Unknown: {type(expr).__name__}}}"


LATEX_OP: Final[dict[type[BinaryOpExpr], str]] = {
	Eq: "=",
	Ne: "\\neq",
	Lt: "<",
	Le: "\\leq",
	Gt: ">",
	Ge: "\\geq",
	And: "\\land",
	Or: "\\lor",
}


def _latex_var(name: str) -> str:
	"""Convert a variable name to LaTeX format.

	Handles subscripts and Greek letters.
	"""
	# Handle subscripts (e.g., x_1 -> x_1, x_max -> x_{max})
	if "_" in name:
		parts = name.split("_", 1)
		base, subscript = parts
		return f"{base}_{subscript}" if len(subscript) == 1 else f"{base}_{{{subscript}}}"

	# Common Greek letters
	return {
		"alpha": "\\alpha",
		"beta": "\\beta",
		"gamma": "\\gamma",
		"delta": "\\delta",
		"epsilon": "\\epsilon",
		"zeta": "\\zeta",
		"eta": "\\eta",
		"theta": "\\theta",
		"iota": "\\iota",
		"kappa": "\\kappa",
		"lambda": "\\lambda",
		"mu": "\\mu",
		"nu": "\\nu",
		"xi": "\\xi",
		"pi": "\\pi",
		"rho": "\\rho",
		"sigma": "\\sigma",
		"tau": "\\tau",
		"upsilon": "\\upsilon",
		"phi": "\\phi",
		"chi": "\\chi",
		"psi": "\\psi",
		"omega": "\\omega",
	}.get(name.lower(), name)


PRECEDENCE: Final[dict[type[Expr[Any, Any]], int]] = {
	Or: 1,
	And: 2,
	Not: 3,
	Eq: 4,
	Ne: 4,
	Lt: 4,
	Le: 4,
	Gt: 4,
	Ge: 4,
	Approximately: 4,
	Add: 5,
	Sub: 5,
	Mul: 6,
	Div: 6,
	Pow: 7,
	Mean: 8,
	StdDev: 8,
	Median: 8,
	Percentile: 8,
	Range: 8,
	Count: 8,
}


def _needs_parentheses(operand: Expr[Any, Any], parent: Expr[Any, Any]) -> bool:
	"""Determine if an operand needs parentheses in the context of its parent operation."""

	if isinstance(
		operand, (Var, Const, PlusMinus, Percent, Mean, StdDev, Median, Percentile, Range, Count)
	):
		return False

	if PRECEDENCE[type(operand)] < PRECEDENCE[type(parent)]:
		return True

	# Special cases for subtraction and division (right-associative concerns)
	if isinstance(parent, Sub) and operand is parent.right:
		if isinstance(operand, (Add, Sub)):
			return True

	if isinstance(parent, Div) and operand is parent.right:
		if isinstance(operand, (Mul, Div)):
			return True

	return False


def _format_statistical_operand(stat_expr: Expr[Any, Any], latex_ctx: LatexCtx[Any] | None) -> str:
	"""Format statistical operand for LaTeX, handling iterable display.

	This leverages the existing to_string method from the statistical operations
	to get proper iterable formatting, then converts to LaTeX-friendly format.
	"""
	match latex_ctx:
		case LatexCtx(ctx, show) if show & Show.VALUES:
			# Use the existing to_string formatting which handles iterable display nicely
			string_repr = stat_expr.to_string(ctx)

			# Extract the operand part from the string representation
			# E.g., "mean(measurements:5[1.0,..5.0] -> 3.0)" -> "measurements:5[1.0,..5.0]"
			if " -> " in string_repr:
				operand_part = string_repr.split(" -> ")[0]
			else:
				operand_part = string_repr

			# Remove the operation name and parentheses
			# E.g., "mean(measurements:5[1.0,..5.0])" -> "measurements:5[1.0,..5.0]"
			# For Percentile: "percentile:95(measurements:5[1.0,..5.0])" -> "measurements:5[1.0,..5.0]"

			# Find the first opening parenthesis and extract what's inside
			paren_start = operand_part.find("(")
			if paren_start != -1:
				# Find the matching closing parenthesis
				paren_end = operand_part.rfind(")")
				if paren_end != -1 and paren_end > paren_start:
					variable_part = operand_part[paren_start + 1 : paren_end]
				else:
					variable_part = operand_part[paren_start + 1 :]
			else:
				variable_part = operand_part

			# Convert variable names to LaTeX format
			# E.g., "measurements:5[1.0,..5.0]" -> "measurements:5[1.0,..5.0]"
			if ":" in variable_part:
				var_name, rest = variable_part.split(":", 1)
				latex_var_name = _latex_var(var_name)
				return f"{latex_var_name}:{rest}"
			else:
				return _latex_var(variable_part)
		case _:
			# Without context or values, just show the variable name
			if hasattr(stat_expr, "left"):
				left_expr = getattr(stat_expr, "left")
				if hasattr(left_expr, "name"):
					return _latex_var(getattr(left_expr, "name"))
			return "data"
