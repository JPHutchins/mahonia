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
	ARG_POS,
	MDEF,
	Argument,
	CallExpr,
	ClassDef,
	Decorator,
	MemberExpr,
	NameExpr,
	SymbolTableNode,
	TypeInfo,
	Var as MypyVar,
)
from mypy.plugin import AttributeContext, ClassDefContext, Plugin
from mypy.types import AnyType, Instance, TupleType, Type, TypeOfAny


class MahoniaPlugin(Plugin):
	"""MyPy plugin for mahonia Context classes."""

	def get_base_class_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
		"""Hook for base class analysis - Context classes."""
		# Disable base class hook since we handle everything in dataclass hook
		return None

	def get_class_decorator_hook(
		self, fullname: str
	) -> Optional[Callable[[ClassDefContext], None]]:
		"""Hook for class decorators - dataclass decorator."""
		if fullname == "dataclasses.dataclass":
			return self._dataclass_class_hook
		return None

	def _context_class_hook(self, ctx: ClassDefContext) -> None:
		"""Process Context subclasses to add vars attribute."""
		# Only process if it's not the base Context class
		if ctx.cls.name != "Context":
			print(f"PLUGIN: Transforming Context subclass {ctx.cls.name} (base hook)")
			transformer = ContextTransformer(ctx)
			transformer.transform()

	def _dataclass_class_hook(self, ctx: ClassDefContext) -> None:
		"""Process dataclass decorators on Context subclasses."""
		# Check if this class inherits from Context
		if self._ctx_inherits_from_context(ctx):
			transformer = ContextTransformer(ctx)
			transformer.transform()

	def _ctx_inherits_from_context(self, ctx: ClassDefContext) -> bool:
		"""Check if class context inherits from Context."""
		for base in ctx.cls.base_type_exprs:
			if hasattr(base, "name") and base.name == "Context":
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
		field_types = {}

		# Try to get field types from the class info first
		for name, symbol in self.ctx.cls.info.names.items():
			if (
				not name.startswith("_")
				and name != "vars"
				and hasattr(symbol.node, "type")
				and symbol.node.type is not None
			):
				field_types[name] = symbol.node.type

		# If that didn't work, try looking at the AST
		if not field_types:
			for stmt in self.ctx.cls.defs.body:
				if hasattr(stmt, "name") and hasattr(stmt, "type"):
					field_name = stmt.name
					if (
						not field_name.startswith("_")
						and field_name != "vars"
						and stmt.type is not None
					):
						# Analyze the type annotation
						field_type = self.ctx.api.anal_type(stmt.type)
						if field_type:
							field_types[field_name] = field_type

		return field_types

	def _add_vars_attribute(self, field_types: dict[str, Type]) -> None:
		"""Add the vars attribute with correct type to the class."""
		# Create Var types for each field
		var_types = []

		for field_name, field_type in field_types.items():
			var_type = self._create_var_type(field_type)
			if var_type:
				var_types.append(var_type)

		if var_types:
			try:
				# Create tuple type - use the correct API
				tuple_type_info = self.ctx.api.named_type("builtins.tuple").type
				vars_type = TupleType(var_types, Instance(tuple_type_info, []))

				# Create and add the vars variable to the class
				vars_var = MypyVar("vars")
				vars_var.type = vars_type
				vars_var.is_classvar = True

				# Create symbol table node
				symbol_node = SymbolTableNode(MDEF, vars_var)

				# Add to the class's namespace
				self.ctx.cls.info.names["vars"] = symbol_node
			except Exception:
				pass

	def _create_var_type(self, field_type: Type) -> Optional[Type]:
		"""Create a Var[field_type, context_type] type."""
		try:
			# Get Var from mahonia
			var_sym = self._lookup_qualified_name("mahonia.Var")
			if not var_sym:
				return None

			# Create the context type (this class)
			context_type = Instance(self.ctx.cls.info, [], line=self.ctx.cls.line)

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
