Keyed Randomness in MLX

Use functional RNG keys for reproducibility across kernels and MLX code.

Basics

```python
k = mx.random.key(123)
k1, k2 = mx.random.split(k, num=2)
a = mx.random.normal(shape=(1024, 32), key=k1)
b = mx.random.normal(shape=(1024, 32), key=k2)
```

Why keys?

- Avoid global RNG state; deterministic across runs and code paths.
- Easy to fork reproducible substreams per module/kernel.

References

- Curated doc: `random.md`
- Ember ML wrappers: simple `Generator` interfaces that hold a key.

