import inspect
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, ClassVar, Final, Protocol, overload, runtime_checkable

from mahonia import (
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	Failure,
	ToString,
)
from mahonia.types import (
	ContextProtocol,
	S,
	TSupportsArithmetic,
	TSupportsComparison,
	TSupportsEquality,
)


@runtime_checkable
class ResultExpr[T, S](Protocol):
	_is_result_type: ClassVar[bool]

	def eval(self, ctx: S) -> Const[T | Failure]: ...

	def to_string(self, ctx: S | None = None) -> str: ...


class ResultBinaryOperationOverloads[T, S](Expr[T, S, T | Failure]):
	_is_result_type: ClassVar[bool] = True

	@overload
	def __add__(self, other: TSupportsArithmetic) -> "ResultAdd[TSupportsArithmetic, S]": ...

	@overload
	def __add__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultAdd[TSupportsArithmetic, S]": ...

	@overload
	def __add__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultAdd[TSupportsArithmetic, S]": ...

	def __add__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultAdd[TSupportsArithmetic, S]":
		return ResultAdd(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __radd__(self, other: TSupportsArithmetic) -> "ResultAdd[TSupportsArithmetic, S]": ...

	@overload
	def __radd__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultAdd[TSupportsArithmetic, S]": ...

	@overload
	def __radd__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultAdd[TSupportsArithmetic, S]": ...

	def __radd__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultAdd[TSupportsArithmetic, S]":
		return ResultAdd(other if isinstance(other, Expr) else Const(None, other), self)

	@overload
	def __sub__(self, other: TSupportsArithmetic) -> "ResultSub[TSupportsArithmetic, S]": ...

	@overload
	def __sub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultSub[TSupportsArithmetic, S]": ...

	@overload
	def __sub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultSub[TSupportsArithmetic, S]": ...

	def __sub__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultSub[TSupportsArithmetic, S]":
		return ResultSub(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __rsub__(self, other: TSupportsArithmetic) -> "ResultSub[TSupportsArithmetic, S]": ...

	@overload
	def __rsub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultSub[TSupportsArithmetic, S]": ...

	@overload
	def __rsub__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultSub[TSupportsArithmetic, S]": ...

	def __rsub__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultSub[TSupportsArithmetic, S]":
		return ResultSub(other if isinstance(other, Expr) else Const(None, other), self)

	@overload
	def __mul__(self, other: TSupportsArithmetic) -> "ResultMul[TSupportsArithmetic, S]": ...

	@overload
	def __mul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultMul[TSupportsArithmetic, S]": ...

	@overload
	def __mul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultMul[TSupportsArithmetic, S]": ...

	def __mul__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultMul[TSupportsArithmetic, S]":
		return ResultMul(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __rmul__(self, other: TSupportsArithmetic) -> "ResultMul[TSupportsArithmetic, S]": ...

	@overload
	def __rmul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic]
	) -> "ResultMul[TSupportsArithmetic, S]": ...

	@overload
	def __rmul__(
		self, other: Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
	) -> "ResultMul[TSupportsArithmetic, S]": ...

	def __rmul__(
		self,
		other: (
			Expr[TSupportsArithmetic, S, TSupportsArithmetic | Failure]
			| Expr[TSupportsArithmetic, S, TSupportsArithmetic]
			| TSupportsArithmetic
		),
	) -> "ResultMul[TSupportsArithmetic, S]":
		return ResultMul(other if isinstance(other, Expr) else Const(None, other), self)

	@overload
	def __truediv__(self, other: float) -> "ResultDiv[float, S]": ...

	@overload
	def __truediv__(self, other: Expr[float, S, float]) -> "ResultDiv[float, S]": ...

	@overload
	def __truediv__(self, other: Expr[float, S, float | Failure]) -> "ResultDiv[float, S]": ...

	def __truediv__(
		self, other: Expr[float, S, float | Failure] | Expr[float, S, float] | float
	) -> "ResultDiv[float, S]":
		return ResultDiv(self, other if isinstance(other, Expr) else Const[float](None, other))

	@overload
	def __rtruediv__(self, other: float) -> "ResultDiv[float, S]": ...

	@overload
	def __rtruediv__(self, other: Expr[float, S, float]) -> "ResultDiv[float, S]": ...

	@overload
	def __rtruediv__(self, other: Expr[float, S, float | Failure]) -> "ResultDiv[float, S]": ...

	def __rtruediv__(
		self, other: Expr[float, S, float | Failure] | Expr[float, S, float] | float
	) -> "ResultDiv[float, S]":
		return ResultDiv(other if isinstance(other, Expr) else Const[float](None, other), self)

	@overload
	def __lt__(self, other: TSupportsComparison) -> "ResultLt[TSupportsComparison, S]": ...

	@overload
	def __lt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "ResultLt[TSupportsComparison, S]": ...

	@overload
	def __lt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison | Failure]
	) -> "ResultLt[TSupportsComparison, S]": ...

	def __lt__(
		self,
		other: (
			Expr[TSupportsComparison, S, TSupportsComparison | Failure]
			| Expr[TSupportsComparison, S, TSupportsComparison]
			| TSupportsComparison
		),
	) -> "ResultLt[TSupportsComparison, S]":
		return ResultLt(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __le__(self, other: TSupportsComparison) -> "ResultLe[TSupportsComparison, S]": ...

	@overload
	def __le__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "ResultLe[TSupportsComparison, S]": ...

	@overload
	def __le__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison | Failure]
	) -> "ResultLe[TSupportsComparison, S]": ...

	def __le__(
		self,
		other: (
			Expr[TSupportsComparison, S, TSupportsComparison | Failure]
			| Expr[TSupportsComparison, S, TSupportsComparison]
			| TSupportsComparison
		),
	) -> "ResultLe[TSupportsComparison, S]":
		return ResultLe(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __gt__(self, other: TSupportsComparison) -> "ResultGt[TSupportsComparison, S]": ...

	@overload
	def __gt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "ResultGt[TSupportsComparison, S]": ...

	@overload
	def __gt__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison | Failure]
	) -> "ResultGt[TSupportsComparison, S]": ...

	def __gt__(
		self,
		other: (
			Expr[TSupportsComparison, S, TSupportsComparison | Failure]
			| Expr[TSupportsComparison, S, TSupportsComparison]
			| TSupportsComparison
		),
	) -> "ResultGt[TSupportsComparison, S]":
		return ResultGt(self, other if isinstance(other, Expr) else Const(None, other))

	@overload
	def __ge__(self, other: TSupportsComparison) -> "ResultGe[TSupportsComparison, S]": ...

	@overload
	def __ge__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison]
	) -> "ResultGe[TSupportsComparison, S]": ...

	@overload
	def __ge__(
		self, other: Expr[TSupportsComparison, S, TSupportsComparison | Failure]
	) -> "ResultGe[TSupportsComparison, S]": ...

	def __ge__(
		self,
		other: (
			Expr[TSupportsComparison, S, TSupportsComparison | Failure]
			| Expr[TSupportsComparison, S, TSupportsComparison]
			| TSupportsComparison
		),
	) -> "ResultGe[TSupportsComparison, S]":
		return ResultGe(self, other if isinstance(other, Expr) else Const(None, other))

	@overload  # type: ignore[override]
	def __eq__(self, other: TSupportsEquality) -> "ResultEq[TSupportsEquality, S]": ...

	@overload
	def __eq__(  # pyright: ignore[reportOverlappingOverload]
		self, other: Expr[TSupportsEquality, S, TSupportsEquality]
	) -> "ResultEq[TSupportsEquality, S]": ...

	@overload
	def __eq__(  # pyright: ignore[reportOverlappingOverload]
		self, other: Expr[TSupportsEquality, S, TSupportsEquality | Failure]
	) -> "ResultEq[TSupportsEquality, S]": ...

	def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
		self,
		other: (
			Expr[TSupportsEquality, S, TSupportsEquality | Failure]
			| Expr[TSupportsEquality, S, TSupportsEquality]
			| TSupportsEquality
		),
	) -> "ResultEq[TSupportsEquality, S]":
		return ResultEq(self, other if isinstance(other, Expr) else Const(None, other))  # pyright: ignore[reportUnknownArgumentType]

	@overload  # type: ignore[override]
	def __ne__(self, other: TSupportsEquality) -> "ResultNe[TSupportsEquality, S]": ...

	@overload
	def __ne__(  # pyright: ignore[reportOverlappingOverload]
		self, other: Expr[TSupportsEquality, S, TSupportsEquality]
	) -> "ResultNe[TSupportsEquality, S]": ...

	@overload
	def __ne__(  # pyright: ignore[reportOverlappingOverload]
		self, other: Expr[TSupportsEquality, S, TSupportsEquality | Failure]
	) -> "ResultNe[TSupportsEquality, S]": ...

	def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
		self,
		other: (
			Expr[TSupportsEquality, S, TSupportsEquality | Failure]
			| Expr[TSupportsEquality, S, TSupportsEquality]
			| TSupportsEquality
		),
	) -> "ResultNe[TSupportsEquality, S]":
		return ResultNe(self, other if isinstance(other, Expr) else Const(None, other))  # pyright: ignore[reportUnknownArgumentType]


@dataclass(frozen=True, eq=False, slots=True)
class ResultBinaryOp[T, S, R](ToString[S], Expr[T, S, R]):
	op: ClassVar[str] = " ? "
	template: ClassVar[str] = "({left}{op}{right})"
	template_eval: ClassVar[str] = "({left}{op}{right} -> {out})"

	left: Expr[Any, S, Any]
	right: Expr[Any, S, Any]

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		right: Final = self.right.to_string(ctx)
		if ctx is None:
			return self.template.format(left=left, op=self.op, right=right)
		return self.template_eval.format(
			left=left, op=self.op, right=right, out=self.eval(ctx).value
		)

	def partial(self, ctx: Any) -> "Expr[T, Any, R]":
		return type(self)(self.left.partial(ctx), self.right.partial(ctx))


class ResultAdd(
	ResultBinaryOp[TSupportsArithmetic, S, TSupportsArithmetic | Failure],
	ResultBinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " + "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv + rv)


class ResultSub(
	ResultBinaryOp[TSupportsArithmetic, S, TSupportsArithmetic | Failure],
	ResultBinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " - "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv - rv)


class ResultMul(
	ResultBinaryOp[TSupportsArithmetic, S, TSupportsArithmetic | Failure],
	ResultBinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " * "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv * rv)


class ResultDiv(
	ResultBinaryOp[TSupportsArithmetic, S, TSupportsArithmetic | Failure],
	ResultBinaryOperationOverloads[TSupportsArithmetic, S],
	BooleanBinaryOperationOverloads[TSupportsArithmetic, S],
):
	op: ClassVar[str] = " / "

	def eval(self, ctx: S) -> Const[TSupportsArithmetic | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv / rv)


class ResultLt(
	ResultBinaryOp[TSupportsComparison, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " < "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv < rv)


class ResultLe(
	ResultBinaryOp[TSupportsComparison, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " <= "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv <= rv)


class ResultGt(
	ResultBinaryOp[TSupportsComparison, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " > "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv > rv)


class ResultGe(
	ResultBinaryOp[TSupportsComparison, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " >= "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv >= rv)


class ResultEq(
	ResultBinaryOp[TSupportsEquality, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " == "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv == rv)


class ResultNe(
	ResultBinaryOp[TSupportsEquality, S, bool | Failure],
	BooleanBinaryOperationOverloads[bool, S],
):
	op: ClassVar[str] = " != "

	def eval(self, ctx: S) -> Const[bool | Failure]:
		lv, rv = self.left.eval(ctx).value, self.right.eval(ctx).value
		match (lv, rv):
			case (Failure() as f1, Failure() as f2):
				return Const(None, f1 + f2)
			case (Failure() as f, _) | (_, Failure() as f):
				return Const(None, f)
			case _:
				return Const(None, lv != rv)


def _func_name(func: Callable[..., Any]) -> str:
	return func.__name__ if func.__name__ != "<lambda>" else "lambda"


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc0[R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[], R]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:  # noqa: ARG002
		try:
			return Const(None, self.func())
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.name}()"
		return f"{self.name}() -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc0[R, Any]":  # noqa: ARG002
		return self


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc1[T1, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1], R]
	arg1: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (self.arg1.eval(ctx).value,)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(a.to_string(ctx) for a in (self.arg1,))
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc1[T1, R, Any]":
		return PythonFunc1(self.func, self.arg1.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc2[T1, T2, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (self.arg1.eval(ctx).value, self.arg2.eval(ctx).value)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(a.to_string(ctx) for a in (self.arg1, self.arg2))
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc2[T1, T2, R, Any]":
		return PythonFunc2(self.func, self.arg1.partial(ctx), self.arg2.partial(ctx))


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc3[T1, T2, T3, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (self.arg1.eval(ctx).value, self.arg2.eval(ctx).value, self.arg3.eval(ctx).value)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(a.to_string(ctx) for a in (self.arg1, self.arg2, self.arg3))
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc3[T1, T2, T3, R, Any]":
		return PythonFunc3(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc4[T1, T2, T3, T4, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(a.to_string(ctx) for a in (self.arg1, self.arg2, self.arg3, self.arg4))
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc4[T1, T2, T3, T4, R, Any]":
		return PythonFunc4(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc5[T1, T2, T3, T4, T5, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx) for a in (self.arg1, self.arg2, self.arg3, self.arg4, self.arg5)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc5[T1, T2, T3, T4, T5, R, Any]":
		return PythonFunc5(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc6[T1, T2, T3, T4, T5, T6, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (self.arg1, self.arg2, self.arg3, self.arg4, self.arg5, self.arg6)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc6[T1, T2, T3, T4, T5, T6, R, Any]":
		return PythonFunc6(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc7[T1, T2, T3, T4, T5, T6, T7, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc7[T1, T2, T3, T4, T5, T6, T7, R, Any]":
		return PythonFunc7(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc8[T1, T2, T3, T4, T5, T6, T7, T8, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]
	arg8: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
			self.arg8.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
				self.arg8,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc8[T1, T2, T3, T4, T5, T6, T7, T8, R, Any]":
		return PythonFunc8(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
			self.arg8.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc9[T1, T2, T3, T4, T5, T6, T7, T8, T9, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]
	arg8: Expr[Any, S, Any]
	arg9: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
			self.arg8.eval(ctx).value,
			self.arg9.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
				self.arg8,
				self.arg9,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc9[T1, T2, T3, T4, T5, T6, T7, T8, T9, R, Any]":
		return PythonFunc9(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
			self.arg8.partial(ctx),
			self.arg9.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc10[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]
	arg8: Expr[Any, S, Any]
	arg9: Expr[Any, S, Any]
	arg10: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
			self.arg8.eval(ctx).value,
			self.arg9.eval(ctx).value,
			self.arg10.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
				self.arg8,
				self.arg9,
				self.arg10,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(
		self,
		ctx: Any,
	) -> "PythonFunc10[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R, Any]":
		return PythonFunc10(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
			self.arg8.partial(ctx),
			self.arg9.partial(ctx),
			self.arg10.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc11[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]
	arg8: Expr[Any, S, Any]
	arg9: Expr[Any, S, Any]
	arg10: Expr[Any, S, Any]
	arg11: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
			self.arg8.eval(ctx).value,
			self.arg9.eval(ctx).value,
			self.arg10.eval(ctx).value,
			self.arg11.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
				self.arg8,
				self.arg9,
				self.arg10,
				self.arg11,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(
		self,
		ctx: Any,
	) -> "PythonFunc11[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R, Any]":
		return PythonFunc11(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
			self.arg8.partial(ctx),
			self.arg9.partial(ctx),
			self.arg10.partial(ctx),
			self.arg11.partial(ctx),
		)


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R, S: ContextProtocol](
	ResultBinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12], R]
	arg1: Expr[Any, S, Any]
	arg2: Expr[Any, S, Any]
	arg3: Expr[Any, S, Any]
	arg4: Expr[Any, S, Any]
	arg5: Expr[Any, S, Any]
	arg6: Expr[Any, S, Any]
	arg7: Expr[Any, S, Any]
	arg8: Expr[Any, S, Any]
	arg9: Expr[Any, S, Any]
	arg10: Expr[Any, S, Any]
	arg11: Expr[Any, S, Any]
	arg12: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		vals = (
			self.arg1.eval(ctx).value,
			self.arg2.eval(ctx).value,
			self.arg3.eval(ctx).value,
			self.arg4.eval(ctx).value,
			self.arg5.eval(ctx).value,
			self.arg6.eval(ctx).value,
			self.arg7.eval(ctx).value,
			self.arg8.eval(ctx).value,
			self.arg9.eval(ctx).value,
			self.arg10.eval(ctx).value,
			self.arg11.eval(ctx).value,
			self.arg12.eval(ctx).value,
		)
		failures = tuple(v for v in vals if isinstance(v, Failure))
		if failures:
			return Const(None, reduce(Failure.__add__, failures))
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		args = ", ".join(
			a.to_string(ctx)
			for a in (
				self.arg1,
				self.arg2,
				self.arg3,
				self.arg4,
				self.arg5,
				self.arg6,
				self.arg7,
				self.arg8,
				self.arg9,
				self.arg10,
				self.arg11,
				self.arg12,
			)
		)
		if ctx is None:
			return f"{self.name}({args})"
		return f"{self.name}({args}) -> {self.eval(ctx).value}"

	def partial(
		self,
		ctx: Any,
	) -> "PythonFunc12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R, Any]":
		return PythonFunc12(
			self.func,
			self.arg1.partial(ctx),
			self.arg2.partial(ctx),
			self.arg3.partial(ctx),
			self.arg4.partial(ctx),
			self.arg5.partial(ctx),
			self.arg6.partial(ctx),
			self.arg7.partial(ctx),
			self.arg8.partial(ctx),
			self.arg9.partial(ctx),
			self.arg10.partial(ctx),
			self.arg11.partial(ctx),
			self.arg12.partial(ctx),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc0Wrapper[R]:  # type: ignore[misc]
	func: Callable[[], R]

	def __call__[S: ContextProtocol](self) -> PythonFunc0[R, S]:  # pyright: ignore[reportInvalidTypeVarUse]
		return PythonFunc0(self.func)


@dataclass(frozen=True, slots=True)
class PythonFunc1Wrapper[T1, R]:  # type: ignore[misc]
	func: Callable[[T1], R]

	def __call__[S: ContextProtocol](self, arg1: T1 | Expr[Any, S, Any]) -> PythonFunc1[T1, R, S]:
		return PythonFunc1(self.func, arg1 if isinstance(arg1, Expr) else Const(None, arg1))  # pyright: ignore[reportUnknownArgumentType]


@dataclass(frozen=True, slots=True)
class PythonFunc2Wrapper[T1, T2, R]:  # type: ignore[misc]
	func: Callable[[T1, T2], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
	) -> PythonFunc2[T1, T2, R, S]:
		return PythonFunc2(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc3Wrapper[T1, T2, T3, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
	) -> PythonFunc3[T1, T2, T3, R, S]:
		return PythonFunc3(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc4Wrapper[T1, T2, T3, T4, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
	) -> PythonFunc4[T1, T2, T3, T4, R, S]:
		return PythonFunc4(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc5Wrapper[T1, T2, T3, T4, T5, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
	) -> PythonFunc5[T1, T2, T3, T4, T5, R, S]:
		return PythonFunc5(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc6Wrapper[T1, T2, T3, T4, T5, T6, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
	) -> PythonFunc6[T1, T2, T3, T4, T5, T6, R, S]:
		return PythonFunc6(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc7Wrapper[T1, T2, T3, T4, T5, T6, T7, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
	) -> PythonFunc7[T1, T2, T3, T4, T5, T6, T7, R, S]:
		return PythonFunc7(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc8Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
		arg8: T8 | Expr[Any, S, Any],
	) -> PythonFunc8[T1, T2, T3, T4, T5, T6, T7, T8, R, S]:
		return PythonFunc8(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc9Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
		arg8: T8 | Expr[Any, S, Any],
		arg9: T9 | Expr[Any, S, Any],
	) -> PythonFunc9[T1, T2, T3, T4, T5, T6, T7, T8, T9, R, S]:
		return PythonFunc9(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
			arg9 if isinstance(arg9, Expr) else Const(None, arg9),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc10Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
		arg8: T8 | Expr[Any, S, Any],
		arg9: T9 | Expr[Any, S, Any],
		arg10: T10 | Expr[Any, S, Any],
	) -> PythonFunc10[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R, S]:
		return PythonFunc10(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
			arg9 if isinstance(arg9, Expr) else Const(None, arg9),  # pyright: ignore[reportUnknownArgumentType]
			arg10 if isinstance(arg10, Expr) else Const(None, arg10),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc11Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
		arg8: T8 | Expr[Any, S, Any],
		arg9: T9 | Expr[Any, S, Any],
		arg10: T10 | Expr[Any, S, Any],
		arg11: T11 | Expr[Any, S, Any],
	) -> PythonFunc11[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R, S]:
		return PythonFunc11(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
			arg9 if isinstance(arg9, Expr) else Const(None, arg9),  # pyright: ignore[reportUnknownArgumentType]
			arg10 if isinstance(arg10, Expr) else Const(None, arg10),  # pyright: ignore[reportUnknownArgumentType]
			arg11 if isinstance(arg11, Expr) else Const(None, arg11),  # pyright: ignore[reportUnknownArgumentType]
		)


@dataclass(frozen=True, slots=True)
class PythonFunc12Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, Any],
		arg2: T2 | Expr[Any, S, Any],
		arg3: T3 | Expr[Any, S, Any],
		arg4: T4 | Expr[Any, S, Any],
		arg5: T5 | Expr[Any, S, Any],
		arg6: T6 | Expr[Any, S, Any],
		arg7: T7 | Expr[Any, S, Any],
		arg8: T8 | Expr[Any, S, Any],
		arg9: T9 | Expr[Any, S, Any],
		arg10: T10 | Expr[Any, S, Any],
		arg11: T11 | Expr[Any, S, Any],
		arg12: T12 | Expr[Any, S, Any],
	) -> PythonFunc12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R, S]:
		return PythonFunc12(
			self.func,
			arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
			arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
			arg9 if isinstance(arg9, Expr) else Const(None, arg9),  # pyright: ignore[reportUnknownArgumentType]
			arg10 if isinstance(arg10, Expr) else Const(None, arg10),  # pyright: ignore[reportUnknownArgumentType]
			arg11 if isinstance(arg11, Expr) else Const(None, arg11),  # pyright: ignore[reportUnknownArgumentType]
			arg12 if isinstance(arg12, Expr) else Const(None, arg12),  # pyright: ignore[reportUnknownArgumentType]
		)


@overload
def python_func[R](f: Callable[[], R]) -> PythonFunc0Wrapper[R]: ...


@overload
def python_func[T1, R](f: Callable[[T1], R]) -> PythonFunc1Wrapper[T1, R]: ...


@overload
def python_func[T1, T2, R](f: Callable[[T1, T2], R]) -> PythonFunc2Wrapper[T1, T2, R]: ...


@overload
def python_func[T1, T2, T3, R](
	f: Callable[[T1, T2, T3], R],
) -> PythonFunc3Wrapper[T1, T2, T3, R]: ...


@overload
def python_func[T1, T2, T3, T4, R](
	f: Callable[[T1, T2, T3, T4], R],
) -> PythonFunc4Wrapper[T1, T2, T3, T4, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, R](
	f: Callable[[T1, T2, T3, T4, T5], R],
) -> PythonFunc5Wrapper[T1, T2, T3, T4, T5, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, R](
	f: Callable[[T1, T2, T3, T4, T5, T6], R],
) -> PythonFunc6Wrapper[T1, T2, T3, T4, T5, T6, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
) -> PythonFunc7Wrapper[T1, T2, T3, T4, T5, T6, T7, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, T8, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
) -> PythonFunc8Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
) -> PythonFunc9Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
) -> PythonFunc10Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11], R],
) -> PythonFunc11Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R]: ...


@overload
def python_func[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R](
	f: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12], R],
) -> PythonFunc12Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R]: ...


def python_func(f: Callable[..., Any]) -> Any:
	match len(inspect.signature(f).parameters):
		case 0:
			return PythonFunc0Wrapper(f)
		case 1:
			return PythonFunc1Wrapper(f)
		case 2:
			return PythonFunc2Wrapper(f)
		case 3:
			return PythonFunc3Wrapper(f)
		case 4:
			return PythonFunc4Wrapper(f)
		case 5:
			return PythonFunc5Wrapper(f)
		case 6:
			return PythonFunc6Wrapper(f)
		case 7:
			return PythonFunc7Wrapper(f)
		case 8:
			return PythonFunc8Wrapper(f)
		case 9:
			return PythonFunc9Wrapper(f)
		case 10:
			return PythonFunc10Wrapper(f)
		case 11:
			return PythonFunc11Wrapper(f)
		case 12:
			return PythonFunc12Wrapper(f)
		case n:
			raise ValueError(f"python_func supports 0-12 args, got {n}")
