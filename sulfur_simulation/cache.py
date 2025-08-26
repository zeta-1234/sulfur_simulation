from __future__ import annotations

import datetime
import pickle  # noqa: S403
from functools import update_wrapper, wraps
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path


def timed[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """
    Log the time taken for f to run.

    Parameters
    ----------
    f : Callable[P, R]
        The function to time

    Returns
    -------
    Callable[P, R]
        The decorated function
    """

    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        try:
            result = f(*args, **kw)
        finally:
            te = datetime.datetime.now(tz=datetime.UTC)
            print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")  # noqa: T201
        return result

    return wrap  # type: ignore[return-value]


CallType = Literal[
    "load_or_call_cached",
    "load_or_call_uncached",
    "call_uncached",
    "call_cached",
]


class CachedFunction[**P, R]:
    """A function wrapper which is used to cache the output."""

    def __init__(
        self,
        function: Callable[P, R],
        path: Path | Callable[P, Path | None] | None,
        *,
        default_call: CallType = "load_or_call_cached",
    ) -> None:
        self._inner = function
        self.Path = path

        self.default_call: CallType = default_call

    def _get_cache_path(self, *args: P.args, **kw: P.kwargs) -> Path | None:
        cache_path = self.Path(*args, **kw) if callable(self.Path) else self.Path
        if cache_path is None:
            return None
        return cache_path

    def call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, without using the cache."""
        return self._inner(*args, **kw)

    def call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, and save the result to the cache."""
        obj = self.call_uncached(*args, **kw)
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is not None:
            buffers = list[pickle.PickleBuffer]()
            with cache_path.open("wb") as f:
                pickler = pickle.Pickler(
                    f,
                    pickle.HIGHEST_PROTOCOL,
                    buffer_callback=buffers.append,
                )
                pickler.dump(obj)

            np.savez(cache_path.with_suffix(".buffer.npz"), *buffers)
        return obj

    def _load_cache(self, *args: P.args, **kw: P.kwargs) -> R | None:
        """Load data from cache."""
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is None:
            return None

        try:
            buffer_data = np.load(cache_path.with_suffix(".buffer.npz"))

            def _get_buffer() -> Generator[memoryview, memoryview]:
                i = 0
                while True:
                    try:
                        yield buffer_data[f"arr_{i}"]
                    except KeyError:
                        break
                    i += 1

            buffers = _get_buffer()
        except FileNotFoundError:
            buffers = list[pickle.PickleBuffer]()

        try:
            with cache_path.open("rb") as f:
                unpickler = pickle.Unpickler(f, buffers=buffers)  # noqa: S301
                return unpickler.load()
        except FileNotFoundError:
            return None

    def load_or_call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function uncached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_uncached(*args, **kw)
        return obj

    def load_or_call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function cached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_cached(*args, **kw)
        return obj

    def __call__(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function using the cache."""
        match self.default_call:
            case "call_cached":
                return self.call_cached(*args, **kw)
            case "call_uncached":
                return self.call_uncached(*args, **kw)
            case "load_or_call_cached":
                return self.load_or_call_cached(*args, **kw)
            case "load_or_call_uncached":
                return self.load_or_call_uncached(*args, **kw)


@overload
def cached[**P, R](
    path: Path | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


@overload
def cached[**P, R](
    path: Callable[P, Path | None],
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


def cached[**P, R](
    path: Path | Callable[P, Path | None] | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]:
    """
    Cache the response of the function at the given path using pickle.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _cached(f: Callable[P, R]) -> CachedFunction[P, R]:
        return update_wrapper(  # type: ignore aaa
            CachedFunction(f, path, default_call=default_call),
            f,
        )

    return _cached
