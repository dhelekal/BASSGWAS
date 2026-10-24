# Cache and sampler benchmarks

Run from the repository root:

```sh
julia --project=BASSGWAS --startup-file=no --threads=1 BASSGWAS/benchmark/cache_benchmarks.jl
julia --project=BASSGWAS --startup-file=no --threads=4 BASSGWAS/benchmark/cache_benchmarks.jl
```

Add `--quick` to run only the 10,000-predictor cases. The full matrix also
includes 100,000 predictors, retained ranks 100 and 300, and 11 or 41 active
columns including the intercept. Each case uses 300 observations and seeded
synthetic inputs. Large cases require several GB of available memory.

The script reports median warmed elapsed time and allocated bytes. Model setup
and the opposite toggle are excluded from addition/removal measurements.
Capacity growth is warmed before measuring. `same_state_update` measures reuse
without a flip; `add_and_update` and `remove_and_update` include the actual
model change. `covariance_and_response` changes `r` before each refresh, whereas
`response_refresh` leaves `r, s` unchanged. Proposal measurements evaluate 25
different parameter pairs. Auxiliary transitions include all continuous draws,
cache refresh and subsequent variable-selection state refresh; they are
stateful, so results summarize a short run rather than one fixed latent state.

BLAS is set to one thread by the benchmark driver, never by the library. The
library uses bounded worker-owned scratch blocks and falls back to serial
outer linear algebra when BLAS is configured with multiple threads. Small
cache kernels also use a serial path. Run benchmark processes separately to
avoid CPU and memory contention.

For an identical workload against an older source snapshot, set
`BASSGWAS_BASELINE_SOURCE` to that snapshot's `src/BASSGWAS.jl`. Use the same
project environment, Julia version and thread count for both runs. The driver
selects the old proposal helper when the workspace API is unavailable.

## Storage and invalidation

`SCache.data` holds read-only fixed inputs and projections;
`SCache.covariance` holds quantities depending on `r, s`; and
`SCache.response` holds projections and forms depending on the latent response.
Use `update_response!` after changing the response and `update_covariance!`
after changing `r, s`. `updateSCache!` remains the compatibility operation and
always reprojects the response, including when it was modified in place.
Replacing or modifying fixed inputs requires reconstructing the cache.

`QCache` retains raw and covariance-adjusted active-by-predictor products.
Each grows geometrically with active-model capacity. A flip adds or removes
one row; a covariance-generation change rebuilds adjusted rows. The `X1` and
`XtX` compatibility properties return views that must not be retained across
model changes. Accepted factorizations are keyed by model generation,
covariance generation, `c` and `mc`; their response solves are keyed separately.
Proposal workspaces never overwrite accepted state. Independent samplers own
all mutable storage; concurrent operations on one sampler are unsupported.
