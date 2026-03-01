import inspect
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Final, overload

from mahonia import (
	BinaryOperationOverloads,
	BooleanBinaryOperationOverloads,
	Const,
	Expr,
	Failure,
	ResultBase,
)
from mahonia.tolerance import ConstTolerance
from mahonia.types import (
	ContextProtocol,
	SupportsArithmetic,
)


@dataclass(frozen=True, eq=False, slots=True)
class ResultApproximately[T: SupportsArithmetic, S](
	Expr[T, S, bool],
	BooleanBinaryOperationOverloads[bool, S],
):
	"""Approximate equality in the Result context: propagates Failure from the left operand.

	Constructed via ``==`` on a ResultBase expression with a ConstTolerance.

	>>> from typing import NamedTuple
	>>> from mahonia import Var, PlusMinus
	>>> from mahonia.python_func import python_func
	>>> class Ctx(NamedTuple):
	... 	x: float
	>>> x = Var[float, Ctx]("x")
	>>> def my_sqrt(v: float) -> float:
	... 	if v < 0: raise ValueError(f"neg: {v}")
	... 	return v ** 0.5
	>>> safe_sqrt = python_func(my_sqrt)
	>>> expr = safe_sqrt(x) == PlusMinus("T", 2.0, 0.1)
	>>> expr.to_string()
	'(my_sqrt(x) \\u2248 T:2.0 \\xb1 0.1)'
	>>> expr.to_string(Ctx(x=4.0))
	'(my_sqrt(x:4.0) -> 2.0 \\u2248 T:2.0 \\xb1 0.1 -> True)'
	>>> expr.to_string(Ctx(x=-1.0))
	"(my_sqrt(x:-1.0) -> Failure(exceptions=(ValueError('neg: -1.0'),)) \\u2248 T:2.0 \\xb1 0.1 -> Failure(exceptions=(ValueError('neg: -1.0'),)))"
	"""

	op: ClassVar[str] = " ≈ "

	left: Expr[Any, S, Any]
	right: ConstTolerance[T]

	def eval(self, ctx: S) -> Const[bool]:
		lv: Final = self.left.unwrap(ctx)
		if isinstance(lv, Failure):
			return lv  # type: ignore[return-value]
		return Const(None, abs(lv - self.right.value) <= self.right.max_abs_error)

	def unwrap(self, ctx: S) -> bool | Failure:  # type: ignore[override]
		result = self.eval(ctx)
		return result if isinstance(result, Failure) else result.value

	def to_string(self, ctx: S | None = None) -> str:
		left: Final = self.left.to_string(ctx)
		right: Final = self.right.to_string(ctx)
		if ctx is None:
			return f"({left}{self.op}{right})"
		return f"({left}{self.op}{right} -> {self.unwrap(ctx)})"

	def partial(self, ctx: Any) -> "ResultApproximately[T, Any]":
		return ResultApproximately(self.left.partial(ctx), self.right)


def _func_name(func: Callable[..., Any]) -> str:
	return func.__name__ if func.__name__ != "<lambda>" else "lambda"


class PythonFuncBase[R, S: ContextProtocol](  # pyright: ignore[reportGeneralTypeIssues]
	ResultBase[R, S],
	BinaryOperationOverloads[R, S],
	BooleanBinaryOperationOverloads[R, S],
):
	"""FFI bridge: wraps a Python callable into the Result context.

	Exceptions raised by the wrapped function become Failure values.
	When multiple arguments are themselves ResultBase, their Failures
	are accumulated before the function is called.

	>>> from typing import NamedTuple
	>>> from mahonia import Failure, Var
	>>> from mahonia.python_func import python_func
	>>> class Ctx(NamedTuple):
	... 	x: float
	... 	y: float
	>>> x, y = Var[float, Ctx]("x"), Var[float, Ctx]("y")
	>>> def my_sqrt(v: float) -> float:
	... 	if v < 0: raise ValueError(f"neg: {v}")
	... 	return v ** 0.5
	>>> safe_sqrt = python_func(my_sqrt)

	Success:

	>>> safe_sqrt(x).to_string(Ctx(x=4.0, y=0.0))
	'my_sqrt(x:4.0) -> 2.0'

	Exception caught as Failure:

	>>> safe_sqrt(x).to_string(Ctx(x=-1.0, y=0.0))
	"my_sqrt(x:-1.0) -> Failure(exceptions=(ValueError('neg: -1.0'),))"

	Nested calls accumulate failures from all arguments:

	>>> def div(a: float, b: float) -> float:
	... 	return a / b
	>>> safe_div = python_func(div)
	>>> r = safe_div(safe_sqrt(x), safe_sqrt(y)).unwrap(Ctx(x=-1.0, y=-4.0))
	>>> isinstance(r, Failure) and len(r.exceptions)
	2
	"""

	func: Callable[..., R]
	args: tuple[Expr[Any, S, Any], ...]

	@property
	def name(self) -> str:
		return _func_name(self.func)

	def eval(self, ctx: S) -> Const[R]:  # pyright: ignore[reportIncompatibleMethodOverride]
		vals = tuple(arg.unwrap(ctx) for arg in self.args)
		failures = tuple(e for v in vals if isinstance(v, Failure) for e in v.exceptions)
		if failures:
			return Failure(failures)  # type: ignore[return-value]
		try:
			return Const(None, self.func(*vals))
		except Exception as e:
			return Failure((e,))  # type: ignore[return-value]

	def to_string(self, ctx: S | None = None) -> str:
		args_str = ", ".join(a.to_string(ctx) for a in self.args)
		if ctx is None:
			return f"{self.name}({args_str})"
		return f"{self.name}({args_str}) -> {self.unwrap(ctx)}"

	def partial(self, ctx: Any) -> "PythonFuncBase[R, Any]":  # pyright: ignore[reportIncompatibleMethodOverride]
		return type(self)(self.func, tuple(arg.partial(ctx) for arg in self.args))  # type: ignore[call-arg]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc0[R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[], R]
	args: tuple[Expr[Any, S, Any], ...] = ()


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc1[T1, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1], R]
	args: tuple[Expr[Any, S, T1] | Expr[Any, S, T1 | Failure]]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc2[T1, T2, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc3[T1, T2, T3, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc4[T1, T2, T3, T4, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3, T4], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc5[T1, T2, T3, T4, T5, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3, T4, T5], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc6[T1, T2, T3, T4, T5, T6, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3, T4, T5, T6], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc7[T1, T2, T3, T4, T5, T6, T7, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc8[T1, T2, T3, T4, T5, T6, T7, T8, R, S: ContextProtocol](PythonFuncBase[R, S]):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc9[T1, T2, T3, T4, T5, T6, T7, T8, T9, R, S: ContextProtocol](
	PythonFuncBase[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc10[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R, S: ContextProtocol](
	PythonFuncBase[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc11[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R, S: ContextProtocol](
	PythonFuncBase[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
		Expr[Any, S, T11] | Expr[Any, S, T11 | Failure],
	]


@dataclass(frozen=True, eq=False, slots=True)
class PythonFunc12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R, S: ContextProtocol](
	PythonFuncBase[R, S],
):
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12], R]
	args: tuple[
		Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
		Expr[Any, S, T11] | Expr[Any, S, T11 | Failure],
		Expr[Any, S, T12] | Expr[Any, S, T12 | Failure],
	]


@dataclass(frozen=True, slots=True)
class PythonFunc0Wrapper[R]:  # type: ignore[misc]
	func: Callable[[], R]

	def __call__[S: ContextProtocol](self) -> PythonFunc0[R, S]:  # pyright: ignore[reportInvalidTypeVarUse]
		return PythonFunc0(self.func)


@dataclass(frozen=True, slots=True)
class PythonFunc1Wrapper[T1, R]:  # type: ignore[misc]
	func: Callable[[T1], R]

	def __call__[S: ContextProtocol](
		self, arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure]
	) -> PythonFunc1[T1, R, S]:
		return PythonFunc1(self.func, (arg1 if isinstance(arg1, Expr) else Const(None, arg1),))  # pyright: ignore[reportUnknownArgumentType]


@dataclass(frozen=True, slots=True)
class PythonFunc2Wrapper[T1, T2, R]:  # type: ignore[misc]
	func: Callable[[T1, T2], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
	) -> PythonFunc2[T1, T2, R, S]:
		return PythonFunc2(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc3Wrapper[T1, T2, T3, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
	) -> PythonFunc3[T1, T2, T3, R, S]:
		return PythonFunc3(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc4Wrapper[T1, T2, T3, T4, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
	) -> PythonFunc4[T1, T2, T3, T4, R, S]:
		return PythonFunc4(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc5Wrapper[T1, T2, T3, T4, T5, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
	) -> PythonFunc5[T1, T2, T3, T4, T5, R, S]:
		return PythonFunc5(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
				arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc6Wrapper[T1, T2, T3, T4, T5, T6, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
	) -> PythonFunc6[T1, T2, T3, T4, T5, T6, R, S]:
		return PythonFunc6(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
				arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
				arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc7Wrapper[T1, T2, T3, T4, T5, T6, T7, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
	) -> PythonFunc7[T1, T2, T3, T4, T5, T6, T7, R, S]:
		return PythonFunc7(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
				arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
				arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
				arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc8Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		arg8: T8 | Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
	) -> PythonFunc8[T1, T2, T3, T4, T5, T6, T7, T8, R, S]:
		return PythonFunc8(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
				arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
				arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
				arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
				arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc9Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		arg8: T8 | Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		arg9: T9 | Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
	) -> PythonFunc9[T1, T2, T3, T4, T5, T6, T7, T8, T9, R, S]:
		return PythonFunc9(
			self.func,
			(
				arg1 if isinstance(arg1, Expr) else Const(None, arg1),  # pyright: ignore[reportUnknownArgumentType]
				arg2 if isinstance(arg2, Expr) else Const(None, arg2),  # pyright: ignore[reportUnknownArgumentType]
				arg3 if isinstance(arg3, Expr) else Const(None, arg3),  # pyright: ignore[reportUnknownArgumentType]
				arg4 if isinstance(arg4, Expr) else Const(None, arg4),  # pyright: ignore[reportUnknownArgumentType]
				arg5 if isinstance(arg5, Expr) else Const(None, arg5),  # pyright: ignore[reportUnknownArgumentType]
				arg6 if isinstance(arg6, Expr) else Const(None, arg6),  # pyright: ignore[reportUnknownArgumentType]
				arg7 if isinstance(arg7, Expr) else Const(None, arg7),  # pyright: ignore[reportUnknownArgumentType]
				arg8 if isinstance(arg8, Expr) else Const(None, arg8),  # pyright: ignore[reportUnknownArgumentType]
				arg9 if isinstance(arg9, Expr) else Const(None, arg9),  # pyright: ignore[reportUnknownArgumentType]
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc10Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		arg8: T8 | Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		arg9: T9 | Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		arg10: T10 | Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
	) -> PythonFunc10[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R, S]:
		return PythonFunc10(
			self.func,
			(
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
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc11Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		arg8: T8 | Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		arg9: T9 | Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		arg10: T10 | Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
		arg11: T11 | Expr[Any, S, T11] | Expr[Any, S, T11 | Failure],
	) -> PythonFunc11[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, R, S]:
		return PythonFunc11(
			self.func,
			(
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
			),
		)


@dataclass(frozen=True, slots=True)
class PythonFunc12Wrapper[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R]:  # type: ignore[misc]
	func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12], R]

	def __call__[S: ContextProtocol](
		self,
		arg1: T1 | Expr[Any, S, T1] | Expr[Any, S, T1 | Failure],
		arg2: T2 | Expr[Any, S, T2] | Expr[Any, S, T2 | Failure],
		arg3: T3 | Expr[Any, S, T3] | Expr[Any, S, T3 | Failure],
		arg4: T4 | Expr[Any, S, T4] | Expr[Any, S, T4 | Failure],
		arg5: T5 | Expr[Any, S, T5] | Expr[Any, S, T5 | Failure],
		arg6: T6 | Expr[Any, S, T6] | Expr[Any, S, T6 | Failure],
		arg7: T7 | Expr[Any, S, T7] | Expr[Any, S, T7 | Failure],
		arg8: T8 | Expr[Any, S, T8] | Expr[Any, S, T8 | Failure],
		arg9: T9 | Expr[Any, S, T9] | Expr[Any, S, T9 | Failure],
		arg10: T10 | Expr[Any, S, T10] | Expr[Any, S, T10 | Failure],
		arg11: T11 | Expr[Any, S, T11] | Expr[Any, S, T11 | Failure],
		arg12: T12 | Expr[Any, S, T12] | Expr[Any, S, T12 | Failure],
	) -> PythonFunc12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R, S]:
		return PythonFunc12(
			self.func,
			(
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
			),
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
	"""Lift a Python callable into the Result context.

	Returns a wrapper whose ``__call__`` accepts Expr or literal arguments and
	produces a PythonFunc expression (a ResultBase).  Supports 0-12 parameters.

	>>> from typing import NamedTuple
	>>> from mahonia import Var
	>>> from mahonia.python_func import python_func
	>>> class Ctx(NamedTuple):
	... 	x: float
	>>> x = Var[float, Ctx]("x")
	>>> def my_sqrt(v: float) -> float:
	... 	if v < 0: raise ValueError(f"neg: {v}")
	... 	return v ** 0.5
	>>> safe_sqrt = python_func(my_sqrt)
	>>> safe_sqrt(x).to_string()
	'my_sqrt(x)'
	>>> safe_sqrt(x).to_string(Ctx(x=4.0))
	'my_sqrt(x:4.0) -> 2.0'
	>>> safe_sqrt(x).to_string(Ctx(x=-1.0))
	"my_sqrt(x:-1.0) -> Failure(exceptions=(ValueError('neg: -1.0'),))"
	"""
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
