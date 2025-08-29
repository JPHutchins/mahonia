# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""LaTeX serialization for mahonia expressions.

This module provides functionality to convert mahonia expressions to LaTeX
mathematical notation for use in documents, papers, or presentation materials.

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

from typing import Any, Final

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
	Sub,
	Var,
)

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


def latex(expr: Expr[Any, Any]) -> str:
	"""Convert a mahonia expression to LaTeX mathematical notation.

	Examples:
	>>> from typing import NamedTuple
	>>> class Ctx(NamedTuple):
	...     x: float
	...     y: float
	>>> x = Var[float, Ctx]("x")
	>>> y = Var[float, Ctx]("y")
	>>> latex(x + y)
	'x + y'
	>>> latex(x * y)
	'x \\\\cdot y'
	>>> latex(x / y)
	'\\\\frac{x}{y}'
	>>> latex(x**2 + y**2)
	'x^2 + y^2'
	"""
	match expr:
		case Var(name=name):
			return _latex_var(name)
		case PlusMinus(name=name, value=value, plus_minus=pm):
			name_str = _latex_var(name) if name else str(value)
			return f"{name_str} \\pm {pm}"
		case Percent(name=name, value=value, percent=pct):
			name_str = _latex_var(name) if name else str(value)
			return f"{name_str} \\pm {pct}\\%"
		case Const(name=name, value=value) if name:
			return _latex_var(name)
		case Const(value=value):
			return str(value)
		case Add():
			return f"{latex(expr.left)} + {latex(expr.right)}"
		case Sub():
			right = latex(expr.right)
			return f"{latex(expr.left)} - {f'({right})' if _needs_parentheses(expr.right, expr) else right}"
		case Mul():
			left = latex(expr.left)
			right = latex(expr.right)
			return f"{f'({left})' if _needs_parentheses(expr.left, expr) else left} \\cdot {f'({right})' if _needs_parentheses(expr.right, expr) else right}"
		case Div():
			return f"\\frac{{{latex(expr.left)}}}{{{latex(expr.right)}}}"
		case Pow():
			base = latex(expr.left)
			power = latex(expr.right)
			formatted_base = f"({base})" if _needs_parentheses(expr.left, expr) else base
			return (
				f"{formatted_base}^{{{power}}}"
				if len(power) > 1 or not power.isdigit()
				else f"{formatted_base}^{power}"
			)
		case Eq() | Ne() | Lt() | Le() | Gt() | Ge() | And() | Or() as binary_op:
			return f"{latex(binary_op.left)} {LATEX_OP[type(binary_op)]} {latex(binary_op.right)}"
		case Not():
			operand = latex(expr.left)
			return f"\\neg {f'({operand})' if _needs_parentheses(expr.left, expr) else operand}"
		case Approximately():
			return f"{latex(expr.left)} \\approx {latex(expr.right)}"
		case Predicate(name=name, expr=pred_expr):
			expr_latex = latex(pred_expr)
			return f"\\text{{{name}}}: {expr_latex}" if name else expr_latex
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
}


def _needs_parentheses(operand: Expr[Any, Any], parent: Expr[Any, Any]) -> bool:
	"""Determine if an operand needs parentheses in the context of its parent operation."""

	if isinstance(operand, (Var, Const, PlusMinus, Percent)):
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
