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

from mahonia import Var

EmptyVars = NamedTuple("EmptyVars", [])


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
			new_class.vars = EmptyVars()  # type: ignore
			return new_class

		# Create Var instances for each field
		var_instances: dict[str, Any] = {  # type: ignore[var-annotated]
			field_name: Var(field_name) for field_name, field_type in field_hints.items()
		}

		# Create a NamedTuple class for the vars with proper field types
		var_fields = []
		for field_name, field_type in field_hints.items():
			var_fields.append((field_name, type(var_instances[field_name])))

		# Create the NamedTuple class
		Vars = NamedTuple(f"{new_class.__name__}Vars", var_fields)  # type: ignore[misc]

		# Try to enhance the NamedTuple with better type annotations for IDEs
		# This is the key insight: manually set __annotations__ to be more specific
		enhanced_annotations = {}
		for field_name, field_type in field_hints.items():
			# Try to create the specific Var[field_type, new_class] type for annotations
			try:
				# This creates the actual Var[int, Context] type for the annotation
				specific_var_type = Var[field_type, new_class]  # type: ignore[valid-type]
				enhanced_annotations[field_name] = specific_var_type
			except Exception:
				# Fallback to generic Var if the specific type creation fails
				# This could happen if field_type is not a proper type or other issues
				# we can at least indicate that this field contains a Var
				enhanced_annotations[field_name] = Var

		# Set the enhanced annotations on the Vars
		Vars.__annotations__ = enhanced_annotations

		# Create the vars instance
		vars_instance = Vars(**var_instances)

		# Set the vars attribute on the class
		new_class.Vars = Vars  # type: ignore[attr-defined]
		new_class.vars = vars_instance  # type: ignore[attr-defined]

		# IMPORTANT: Create proper type annotations for IDE support
		# Add proper __annotations__ for the vars attribute that IDEs can understand
		if not hasattr(new_class, "__annotations__"):
			new_class.__annotations__ = {}

		# Try to create better type information for IDEs
		# The key insight: Pylance might need the actual NamedTuple type, not generic annotations
		if field_hints:
			# Since we created Vars above, use that for the annotation
			# This gives IDEs the actual NamedTuple class type
			new_class.__annotations__["vars"] = Vars
		else:
			# Empty tuple for classes with no fields
			new_class.__annotations__["vars"] = type(EmptyVars())

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
	# The type annotation here helps IDEs understand the general shape
	vars: Any = EmptyVars()
	__slots__ = ()
