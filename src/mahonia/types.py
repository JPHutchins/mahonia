# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from typing import (
	Any,
	Iterable,
	Protocol,
	Sized,
	TypeVar,
	TypeVarTuple,
)

T = TypeVar("T")
"""The type of the expression's operands."""

U = TypeVar("U")
"""A generic type parameter for mapped values."""

R = TypeVar("R")
"""The type returned by eval() and unwrap()."""


class ContextProtocol(Protocol):
	def __getattribute__(self, name: str, /) -> Any: ...


S = TypeVar("S", bound=ContextProtocol)
"""The type of the expression's context."""

T_co = TypeVar("T_co", covariant=True)


Ss = TypeVarTuple("Ss")


class MergeContextProtocol(Protocol[*Ss]):
	def __getattribute__(self, name: str, /) -> Any: ...


S_contra = TypeVar("S_contra", bound=ContextProtocol, contravariant=True)
"""The contravariant type of the expression's context."""


class SizedIterable(Sized, Iterable[T_co], Protocol[T_co]):
	"""Protocol for containers that are both sized and iterable with indexing support."""

	def __getitem__(self, index: int, /) -> T_co: ...


class SupportsArithmetic(Protocol):
	def __add__(self, other: Any, /) -> Any: ...
	def __sub__(self, other: Any, /) -> Any: ...
	def __mul__(self, other: Any, /) -> Any: ...
	def __truediv__(self, other: Any, /) -> Any: ...
	def __mod__(self, other: Any, /) -> Any: ...
	def __abs__(self, /) -> Any: ...
	def __pow__(self, power: Any, /) -> Any: ...


TSupportsArithmetic = TypeVar("TSupportsArithmetic", bound=SupportsArithmetic)


class SupportsEquality(Protocol):
	def __eq__(self, other: Any, /) -> bool: ...
	def __ne__(self, other: Any, /) -> bool: ...


TSupportsEquality = TypeVar("TSupportsEquality", bound=SupportsEquality)


class SupportsComparison(Protocol):
	def __lt__(self, other: Any, /) -> bool: ...
	def __le__(self, other: Any, /) -> bool: ...
	def __gt__(self, other: Any, /) -> bool: ...
	def __ge__(self, other: Any, /) -> bool: ...


TSupportsComparison = TypeVar("TSupportsComparison", bound=SupportsComparison)


class SupportsLogic(Protocol):
	def __and__(self, other: Any, /) -> Any: ...
	def __or__(self, other: Any, /) -> Any: ...
	def __invert__(self, /) -> Any: ...


TSupportsLogic = TypeVar("TSupportsLogic", bound=SupportsLogic)


class EvalError(Exception):
	"""Raised when variable evaluation fails due to missing context attributes."""

	pass


R_Eval = TypeVar("R_Eval")
"""Result type for the Eval protocol (no default, since T is not in scope)."""
