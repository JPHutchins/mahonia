import asyncio
from typing import Any, Awaitable, Callable, Final, Generic, NamedTuple, TypeVar, assert_never

import mahonia

type Testable = mahonia.BoolExpr[Any, Any] | mahonia.Predicate[Any]

TContext = TypeVar("TContext", bound=mahonia.ContextProtocol)
T = TypeVar("T")


class CachedContext(Generic[TContext]):
	class Initial(NamedTuple): ...

	class InProgress(NamedTuple, Generic[T]):
		future: asyncio.Future[T]

	def __init__(self, func: Callable[[], Awaitable[TContext]]) -> None:
		self._func: Final = func
		self._state: CachedContext.Initial | CachedContext.InProgress[TContext] = self.Initial()

	async def func(self) -> TContext:
		match self._state:
			case self.Initial():
				self._state = self.InProgress(future=asyncio.Future())
				try:
					self._state.future.set_result(await self._func())
				except Exception as e:
					self._state.future.set_exception(e)
				return await self._state.future
			case self.InProgress() as in_progress:
				return await in_progress.future
			case _:
				assert_never(self._state)


class TestStep(NamedTuple, Generic[TContext]):
	"""Test step with a context-providing function and test expressions."""

	context_func: Callable[[], Awaitable[TContext]]
	tests: mahonia.SizedIterable[Testable]


def cached_context(
	context_func: Callable[[], Awaitable[TContext]],
) -> Callable[[], TestStep[TContext]]:
	"""
	Decorator that converts a context-producing async function into a cached
	test step factory function.

	Takes an async function that produces context and returns a callable that produces
	TestStep instances with an empty test tuple. The context function is wrapped in
	CachedContext to ensure it executes at most once, even if the returned callable
	is invoked multiple times.

	>>> import time
	>>> from typing import NamedTuple
	>>>
	>>> class Power(NamedTuple):
	...     voltage: float
	...     current: float
	...     ts: float
	>>>
	>>> @cached_context
	... async def step0_check_power() -> Power:
	...     return Power(voltage=4.2, current=0.42, ts=time.time())
	>>>
	>>> test_step = step0_check_power()
	>>> test_step.tests
	()
	>>>
	>>> import asyncio
	>>> result1 = asyncio.run(test_step.context_func())
	>>> result2 = asyncio.run(test_step.context_func())
	>>> result1.ts == result2.ts  # Function executed only once
	True
	"""
	cached: Final = CachedContext(context_func)

	def wrapper() -> TestStep[TContext]:
		return TestStep(cached.func, ())

	wrapper.__name__ = context_func.__name__
	return wrapper
