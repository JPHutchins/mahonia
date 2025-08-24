# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

"""MyPy plugin for mahonia Context classes.

This plugin provides static type inference for Context classes, ensuring that
the automatically generated 'vars' attribute has the correct type information.

Based on patterns from the Pydantic mypy plugin approach.
"""

from typing import Callable, Optional
from typing import Type as TypingType

from mypy.nodes import (
	MDEF,
	AssignmentStmt,
	NameExpr,
	SymbolTableNode,
	TypeInfo,
)
from mypy.nodes import (
	Var as MypyVar,
)
from mypy.plugin import ClassDefContext, Plugin
from mypy.types import Instance, TupleType, Type


class MahoniaPlugin(Plugin):
	"""MyPy plugin for mahonia Context classes."""

	def get_class_decorator_hook(
		self, fullname: str
	) -> Optional[Callable[[ClassDefContext], None]]:
		"""Hook for class decorators - dataclass decorator."""
		if fullname == "dataclasses.dataclass":
			return self._dataclass_class_hook
		return None

	def _dataclass_class_hook(self, ctx: ClassDefContext) -> None:
		"""Process dataclass decorators on Context subclasses."""
		# Check if this class inherits from Context
		if self._ctx_inherits_from_context(ctx):
			transformer = ContextTransformer(ctx)
			transformer.transform()

	def _ctx_inherits_from_context(self, ctx: ClassDefContext) -> bool:
		"""Check if class context inherits from Context."""
		# Check via type info if available
		if ctx.cls.info:
			for base_info in ctx.cls.info.mro:
				if base_info.fullname == "mahonia.context.Context":
					return True
		return False


class ContextTransformer:
	"""Transform Context classes to add proper type information."""

	def __init__(self, ctx: ClassDefContext) -> None:
		self.ctx = ctx

	def transform(self) -> None:
		"""Transform a Context subclass."""
		# Get field information from the class
		field_types = self._collect_field_types()

		if field_types:
			# Create the vars type and add it to the class
			self._add_vars_attribute(field_types)

	def _collect_field_types(self) -> dict[str, Type]:
		"""Collect field types from class annotations."""
		field_types: dict[str, Type] = {}

		# Look at the class AST for field annotations
		for stmt in self.ctx.cls.defs.body:
			# Check if this is an annotated assignment statement
			if isinstance(stmt, AssignmentStmt) and stmt.type is not None:
				# This is an annotated assignment
				for lvalue in stmt.lvalues:
					if isinstance(lvalue, NameExpr):
						field_name = lvalue.name
						if not field_name.startswith("_") and field_name != "vars":
							# Analyze the type annotation
							field_type = self.ctx.api.anal_type(stmt.type)
							if field_type:
								field_types[field_name] = field_type

		# If we didn't find any annotations in the AST, try the symbol table
		if not field_types and self.ctx.cls.info:
			for name, symbol in self.ctx.cls.info.names.items():
				if (
					not name.startswith("_")
					and name != "vars"
					and symbol.node
					and isinstance(symbol.node, MypyVar)
					and symbol.node.type is not None
				):
					field_types[name] = symbol.node.type

		return field_types

	def _add_vars_attribute(self, field_types: dict[str, Type]) -> None:
		"""Add the vars attribute with correct type to the class."""
		# Skip if vars already exists
		if "vars" in self.ctx.cls.info.names:
			return

		try:
			if not field_types:
				# For classes with no fields, create empty tuple like the runtime does
				empty_tuple = TupleType([], fallback=self.ctx.api.named_type("builtins.tuple"))
				vars_var = MypyVar("vars", empty_tuple)
				vars_var.info = self.ctx.cls.info
				vars_var.is_classvar = True
				symbol_node = SymbolTableNode(MDEF, vars_var, plugin_generated=True)
				self.ctx.cls.info.names["vars"] = symbol_node
				return

			# Get the Context type (this class) - mirror what the metaclass does
			context_type = Instance(self.ctx.cls.info, [], line=self.ctx.cls.line)

			# Create Var[field_type, context_type] for each field - same as metaclass
			var_types: list[Type] = []
			for field_name, field_type in field_types.items():
				# Skip private fields and 'vars' - same filtering as metaclass
				if not field_name.startswith("_") and field_name != "vars":
					var_type = self._create_var_type(field_type, context_type)
					if var_type:
						var_types.append(var_type)

			if var_types:
				# Create a TupleType with the exact Var types - this should match runtime
				vars_tuple_type = TupleType(
					var_types, fallback=self.ctx.api.named_type("builtins.tuple")
				)

				vars_var = MypyVar("vars", vars_tuple_type)
				vars_var.info = self.ctx.cls.info
				vars_var.is_classvar = True
				vars_var.is_property = False
				vars_var.is_self = False
				vars_var.is_initialized_in_class = True

				# Add to class symbol table
				symbol_node = SymbolTableNode(MDEF, vars_var, plugin_generated=True)
				self.ctx.cls.info.names["vars"] = symbol_node
			else:
				# Fallback to simple tuple if no valid fields
				builtin_tuple = self.ctx.api.named_type("builtins.tuple")
				vars_var = MypyVar("vars", builtin_tuple)
				vars_var.info = self.ctx.cls.info
				vars_var.is_classvar = True
				symbol_node = SymbolTableNode(MDEF, vars_var, plugin_generated=True)
				self.ctx.cls.info.names["vars"] = symbol_node

		except Exception:
			# Fallback to simple tuple if anything goes wrong
			try:
				builtin_tuple = self.ctx.api.named_type("builtins.tuple")
				vars_var = MypyVar("vars", builtin_tuple)
				vars_var.info = self.ctx.cls.info
				vars_var.is_classvar = True
				symbol_node = SymbolTableNode(MDEF, vars_var, plugin_generated=True)
				self.ctx.cls.info.names["vars"] = symbol_node
			except Exception:
				pass

	def _create_var_type(self, field_type: Type, context_type: Type) -> Optional[Type]:
		"""Create a Var[field_type, context_type] type."""
		try:
			# Get Var from mahonia
			var_sym = self._lookup_qualified_name("mahonia.Var")
			if not var_sym:
				return None

			# Create Var[field_type, context_type]
			var_instance = Instance(var_sym, [field_type, context_type], line=self.ctx.cls.line)
			return var_instance

		except Exception:
			return None

	def _lookup_qualified_name(self, name: str) -> Optional[TypeInfo]:
		"""Look up a qualified name like 'mahonia.Var'."""
		try:
			module_name, _, class_name = name.rpartition(".")
			module = self.ctx.api.modules.get(module_name)
			if not module:
				return None

			symbol = module.names.get(class_name)
			if not symbol or not isinstance(symbol.node, TypeInfo):
				return None

			return symbol.node
		except Exception:
			return None


def plugin(version: str) -> TypingType[MahoniaPlugin]:
	"""Plugin entry point for mypy."""
	return MahoniaPlugin
