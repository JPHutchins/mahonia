# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from mahonia import Expr, Var


def extract_vars(
	vars: tuple["Var[Any, Any]", ...], expr: "Expr[Any, Any, Any]"
) -> tuple["Var[Any, Any]", ...]:
	"""Extract all unique variables from an expression, preserving order.

	Note: pyright ignores on pattern match cases below are necessary because pattern
	matching on generic dataclasses loses type parameter information. This is a known
	pyright limitation with structural pattern matching on Generic types.
	"""
	from mahonia import (
		AllExpr,
		AnyExpr,
		BinaryOp,
		Contains,
		FilterExpr,
		FoldLExpr,
		MapExpr,
		MaxExpr,
		MinExpr,
		UnaryOpEval,
		Var,
	)

	match expr:
		case Var() as v:
			if id(v) not in (id(var) for var in vars):
				vars += (v,)
		case BinaryOp(left, right):
			vars = extract_vars(vars, left)
			vars = extract_vars(vars, right)
		case UnaryOpEval(left=left):  # pyright: ignore[reportUnnecessaryComparison, reportUnknownVariableType]
			vars = extract_vars(vars, left)  # pyright: ignore[reportUnknownArgumentType]
		case Contains(element=element, container=container):  # pyright: ignore[reportUnknownVariableType]
			vars = extract_vars(vars, element)  # pyright: ignore[reportUnknownArgumentType]
			vars = extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case _ if hasattr(expr, "branches") and hasattr(expr, "default"):
			for condition, value in expr.branches:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
				vars = extract_vars(vars, condition)  # pyright: ignore[reportUnknownArgumentType]
				vars = extract_vars(vars, value)  # pyright: ignore[reportUnknownArgumentType]
			vars = extract_vars(vars, expr.default)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
		case (
			AnyExpr(container)
			| AllExpr(container)
			| MinExpr(container)
			| MaxExpr(container)
			| FoldLExpr(container=container)  # pyright: ignore[reportUnknownVariableType]
		):
			vars = extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case MapExpr(func=func, container=container):  # pyright: ignore[reportUnknownVariableType]
			for arg in func.args:
				if isinstance(arg, Var) and id(arg) not in (id(var) for var in vars):
					vars += (arg,)
			vars = extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case FilterExpr(predicate=predicate, container=container):  # pyright: ignore[reportUnknownVariableType]
			for arg in predicate.args:
				if isinstance(arg, Var) and id(arg) not in (id(var) for var in vars):
					vars += (arg,)
			vars = extract_vars(vars, container)  # pyright: ignore[reportUnknownArgumentType]
		case _:
			pass
	return vars
