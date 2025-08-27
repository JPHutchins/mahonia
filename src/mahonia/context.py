# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""Context classes that automatically generate Var instances for type-safe field access.

This module provides the Context base class that eliminates the need for manual
Var creation by automatically generating appropriately typed Var instances at
class creation time.

Example:
    >>> from typing import NamedTuple
    >>> class MyContext(Context):
    ...     my_name: str
    ...     my_other_var: int
    >>> my_name, my_other_var = MyContext.vars
    >>> # my_name is now a properly typed Var[str, MyContext]
    >>> # my_other_var is now a properly typed Var[int, MyContext]
"""

from __future__ import annotations

from typing import Any, NamedTuple, get_type_hints

from . import Var


class ContextMeta(type):
	"""Metaclass for Context classes that generates the vars attribute."""

	def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
		# Create the class first
		new_class = super().__new__(cls, name, bases, namespace)

		# Skip processing for the base Context class itself
		if name == "Context" and not bases:
			return new_class

		# Get type hints for the class (direct annotations only, no inheritance)
		# Try get_type_hints first, but fall back to __annotations__ if that fails
		# This handles cases where @dataclass hasn't run yet
		try:
			type_hints = get_type_hints(new_class)
		except (NameError, AttributeError):
			# Fall back to direct annotations only
			type_hints = getattr(new_class, "__annotations__", {})
			if not type_hints:
				return new_class

		# Filter out non-field annotations:
		# - private attributes (starting with _)
		# - methods and properties
		# - the vars attribute itself
		field_hints = {}
		for field_name, field_type in type_hints.items():
			if not field_name.startswith("_") and field_name != "vars":
				# Check if it's actually a callable (method/property)
				field_attr = getattr(new_class, field_name, None)
				if field_attr is not None and callable(field_attr):
					continue  # Skip callable attributes
				field_hints[field_name] = field_type

		if not field_hints:
			# Create empty vars for classes with no fields
			new_class.vars = NamedTuple("Vars", [])()
			return new_class

		# Create Var instances for each field
		var_instances = {
			field_name: Var[field_type, new_class](field_name)
			for field_name, field_type in field_hints.items()
		}

		# Create a NamedTuple class for the vars
		VarsClass = NamedTuple("Vars", list(var_instances.items()))

		# Create the vars instance
		vars_instance = VarsClass(**var_instances)

		# Set the vars attribute on the class
		new_class.vars = vars_instance

		# IMPORTANT: Create proper type annotations for mypy
		# The key insight: we need to set up the class so mypy sees the right types
		# without needing complex plugin logic

		# Add proper __annotations__ for the vars attribute
		if not hasattr(new_class, "__annotations__"):
			new_class.__annotations__ = {}

		# Create a type annotation for vars that mypy can understand
		# We'll construct the tuple type signature
		if field_hints:
			from typing import get_type_hints

			# Try to create a proper type annotation
			# This is a simplified approach - in a full implementation,
			# we might need to construct the exact Tuple[Var[...], ...] type
			new_class.__annotations__["vars"] = "NamedTuple"  # Simplified for now

		return new_class


class Context(metaclass=ContextMeta):
	"""Base class for contexts that automatically generate Var instances.

	This class uses a metaclass to automatically create a 'vars' class attribute
	containing properly typed Var instances for each annotated field.

	Example:
	    >>> from dataclasses import dataclass
	    >>> @dataclass
	    ... class MyCtx(Context):
	    ...     x: int
	    ...     y: str
	    >>> x, y = MyCtx.vars
	    >>> # x is Var[int, MyCtx], y is Var[str, MyCtx]
	    >>> isinstance(x, Var)
	    True
	    >>> x.name
	    'x'
	    >>> ctx = MyCtx(x=42, y="hello")
	    >>> x.unwrap(ctx)
	    42
	    >>> y.unwrap(ctx)
	    'hello'
	"""

	# This will be set by the metaclass for subclasses
	vars: NamedTuple = NamedTuple("EmptyVars", [])()
	__slots__ = ()
