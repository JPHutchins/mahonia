import inspect
from dataclasses import dataclass
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
	arg: Expr[Any, S, Any]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R | Failure]:
		arg_val = self.arg.eval(ctx).value
		if isinstance(arg_val, Failure):
			return Const(None, arg_val)
		try:
			return Const(None, self.func(arg_val))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.name}({self.arg.to_string()})"
		return f"{self.name}({self.arg.to_string(ctx)}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc1[T1, R, Any]":
		return PythonFunc1(self.func, self.arg.partial(ctx))


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
		arg1_val = self.arg1.eval(ctx).value
		arg2_val = self.arg2.eval(ctx).value
		if isinstance(arg1_val, Failure) and isinstance(arg2_val, Failure):
			return Const(None, arg1_val + arg2_val)
		if isinstance(arg1_val, Failure):
			return Const(None, arg1_val)
		if isinstance(arg2_val, Failure):
			return Const(None, arg2_val)
		try:
			return Const(None, self.func(arg1_val, arg2_val))
		except Exception as e:
			return Const(None, Failure((e,)))

	def to_string(self, ctx: S | None = None) -> str:
		if ctx is None:
			return f"{self.name}({self.arg1.to_string()}, {self.arg2.to_string()})"
		return f"{self.name}({self.arg1.to_string(ctx)}, {self.arg2.to_string(ctx)}) -> {self.eval(ctx).value}"

	def partial(self, ctx: Any) -> "PythonFunc2[T1, T2, R, Any]":
		return PythonFunc2(self.func, self.arg1.partial(ctx), self.arg2.partial(ctx))


@dataclass(frozen=True, slots=True)
class PythonFunc0Wrapper[R]:  # type: ignore[misc]
	func: Callable[[], R]

	def __call__[S: ContextProtocol](self) -> PythonFunc0[R, S]:  # pyright: ignore[reportInvalidTypeVarUse]
		return PythonFunc0(self.func)


@dataclass(frozen=True, slots=True)
class PythonFunc1Wrapper[T1, R]:  # type: ignore[misc]
	func: Callable[[T1], R]

	def __call__[S: ContextProtocol](self, arg: T1 | Expr[Any, S, Any]) -> PythonFunc1[T1, R, S]:
		return PythonFunc1(self.func, arg if isinstance(arg, Expr) else Const(None, arg))  # pyright: ignore[reportUnknownArgumentType]


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


@overload
def python_func[R](f: Callable[[], R]) -> PythonFunc0Wrapper[R]: ...


@overload
def python_func[T1, R](f: Callable[[T1], R]) -> PythonFunc1Wrapper[T1, R]: ...


@overload
def python_func[T1, T2, R](f: Callable[[T1, T2], R]) -> PythonFunc2Wrapper[T1, T2, R]: ...


def python_func(f: Callable[..., Any]) -> Any:
	match len(inspect.signature(f).parameters):
		case 0:
			return PythonFunc0Wrapper(f)
		case 1:
			return PythonFunc1Wrapper(f)
		case 2:
			return PythonFunc2Wrapper(f)
		case n:
			raise ValueError(f"python_func supports 0-2 args, got {n}")
