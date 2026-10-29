# Numerical test suite

Run the complete suite from the package directory:

```sh
julia --project=. --startup-file=no --threads=1 test/runtests.jl
julia --project=. --startup-file=no --threads=4 test/runtests.jl
```

The enclosing testset allows all groups to run
even if a regression fails. No checks are marked broken or skipped.

## Block task execution

`block_tasks_tests.jl` checks contiguous, exactly-once block coverage, partial
blocks, logical worker indices, worker-owned scratch, deterministic failure
selection and per-worker stopping. It also checks independent concurrent
runners, preserved matrix and view types for BLAS dispatch, argument
validation, and status allocations bounded by worker count. A three-thread
run exercises uneven partitions.
Candidate failure tests in `workspace_tests.jl` cover different worker ranges,
multiple failures, and recovery with single- and multi-threaded BLAS settings.

## Floating-point state storage

`state_type_tests.jl` checks `ParState2{Float64}` and `wTGSState2{Float64}`
construction from mixed integer, rational and Float64 inputs, explicit type
selection, dense conversion of views, shared cache inputs, and invalid shapes
and parameter values. Complete Float64 sampler transitions exercise both
sampled and fixed coefficient-prior variance and all adaptation stages.
They also verify proposal adaptation with fixed or adapted `xi`, freezing at
exactly `nadapt` transitions, and disabling all adaptation with `nadapt=0`.
Float32 numerical validation is intentionally deferred.

## Independent numerical references

`numerical_helpers.jl` enumerates every inclusion vector for one to four
predictors. It constructs each observation-space covariance, factors it with
dense Cholesky, and evaluates Gaussian/Student-t likelihoods and the integrated
Beta–Bernoulli vector prior independently of package calculation helpers. The
intercept is always included and has its own variance. These are posteriors
conditional on fixed latent responses and continuous parameters; they do not
integrate the full binary-response probit posterior over those quantities.

The tests verify conditional inclusion/current-state probabilities, likelihood
ratios, move rates, cumulative move selection, detailed balance, and exact
importance-weighted recovery of model probabilities and marginal inclusion
probabilities. For the finite transition check, continuous parameters are frozen
and auxiliary events are self-transitions.

Cache tests visit all four-predictor models in Gray-code order, return to the
empty model, and repeat after response and parameter changes. Fixtures include
full/truncated covariance eigensystems, non-unit residual variance, and
correlated, duplicate, and zero predictor columns.

Gaussian coefficient draws use an independent observation-space conditional
mean/covariance reference, 20,000 local seeded draws per fixture, and whitened
moment checks with six-standard-error bounds. Residual-variance tests use dense
quadratic forms and cloned RNGs to verify the inverse-gamma draw and liability
rescaling. These two draw routines assume unit residual variance. Deterministic
comparisons use `rtol=1e-8` and `atol=1e-10` (normalization checks use the stated
absolute tolerance).

An intercept-only binary probit regression also checks the full sampler's
posterior second moment against independent numerical integration. Three seeded
chains and batch-mean error estimates exercise the shift augmentation, beyond
the fixed-latent-response conditional checks above.
## Dependency caches and reusable workspaces

`workspace_tests.jl` checks independent response/covariance invalidation,
zero and near-zero eigenvalues, empty and rank-one bases, guarded weight
arithmetic, proposal isolation, and conditional response forms. A long model
walk forces growth in allocated model size and compares incremental covariance-adjusted
products with dense reconstruction after additions, removals, and variance
changes. Allocation checks warm each workspace before measuring and exclude
predictor-sized temporary arrays. Independent sampler instances are also run
concurrently with separate RNGs and compared with serial execution.

`SCache` no longer stores `F = Y'C^-1`; cache checks instead validate the
covariance-adjusted design diagonal against the independent dense precision.
Timing benchmarks are separate from correctness tests; see
[`../benchmark/README.md`](../benchmark/README.md).

Rate storage uses `logrates`, with `logtot` holding the log of the unscaled
intensity sum. `rsums` contains cumulative `exp(logrates - maximum(logrates))`;
its endpoint is the scaled total, not necessarily one. Extreme-rate tests use
independent BigFloat probability definitions and check selection, detailed
balance, importance correction, zero regularization and auxiliary intensity,
auxiliary-only states, and independent concurrent kernels. Run focused uneven
partitions with `julia --project=. --threads=3 test/rate_tests.jl`.
