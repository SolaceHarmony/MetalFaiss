import mlx.core as mx


def assert_allclose(a: mx.array, b: mx.array, rtol: float = 1e-5, atol: float = 1e-8, msg: str | None = None) -> None:
    ok = mx.allclose(a, b, rtol=rtol, atol=atol)
    if not bool(ok):
        raise AssertionError(msg or f"Arrays not close within rtol={rtol}, atol={atol}")


def assert_array_equal(a: mx.array, b: mx.array, msg: str | None = None) -> None:
    ok = mx.all(a == b)
    if not bool(ok):
        raise AssertionError(msg or "Arrays are not equal")


def randint(low: int, high: int, shape: tuple[int, ...], dtype=mx.int32) -> mx.array:
    return mx.random.randint(low, high, shape=shape, dtype=dtype)


def randn(shape: tuple[int, ...], dtype=mx.float32) -> mx.array:
    return mx.random.normal(shape=shape).astype(dtype) if hasattr(mx.random.normal(shape=(1,)), 'astype') else mx.random.normal(shape=shape)

