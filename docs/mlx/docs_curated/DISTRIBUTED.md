# Distributed (Multi‑Process)

MLX exposes basic distributed primitives for data/model parallel work.

## Initialization

```python
from mlx.core.distributed import init, Group, is_available

if is_available():
    init()  # configure via env: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT
```

`Group` lets you define custom process groups; the default world covers all ranks.

## Collectives

```python
from mlx.core.distributed import all_gather, all_sum

x = mx.arange(4) + rank  # rank‑local data
g = all_gather(x)        # concat across ranks
s = all_sum(mx.array([1.0]))

# Mean across ranks
world_size = int(os.environ.get("WORLD_SIZE", "1"))
mean = all_sum(mx.array([loss])) / mx.array([world_size])
```

## Point‑to‑Point

```python
from mlx.core.distributed import send, recv, recv_like

if rank == 0:
    send(mx.arange(10), dst=1, tag=0)
else:
    y = recv_like(mx.zeros(10), src=0, tag=0)
```

Notes:

- Ensure ranks agree on shapes/dtypes.
- Set up launchers with `mpirun`, `torchrun`, or your preferred multi‑proc runner.
- APIs evolve; check your MLX version for available collectives.
- Start with data parallelism: split data by `rank/world_size`, keep models identical.

Seeding and shuffles per rank
```python
def rank_key(base_seed, rank):
    return mx.random.key(int(base_seed) ^ (rank * 0x9E3779B97F4A7C15))

k = rank_key(0, rank)
perm = mx.random.permutation(N, key=k)
```

Device/streams
- Distributed calls operate on MLX arrays; use default/scoped devices as usual.
- Synchronize only when needed; MLX executes lazily until eval/host use.
