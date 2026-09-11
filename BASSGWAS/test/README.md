# Numerical test suite

Run the complete suite from the package directory:

```sh
julia --project=. --startup-file=no --threads=1 test/runtests.jl
julia --project=. --startup-file=no --threads=4 test/runtests.jl
```

The suite currently exits unsuccessfully because it contains ordinary regression
assertions for two confirmed production defects. The enclosing testset allows
all groups to run before reporting failure. No checks are marked broken or
skipped, and production source is unchanged.

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

## Validation results

Validated on 2026-09-11 with Julia 1.12.7. One-thread and four-thread runs each
reported **21,790 passes, 24 failures, zero errors, and zero broken tests**.

| Group | Pass | Fail |
| --- | ---: | ---: |
| Original BVS tests | 16 | 0 |
| Exact conditional posteriors | 5,564 | 0 |
| Update rates and transitions | 1,831 | 0 |
| Numerical cache updates | 13,248 | 0 |
| Continuous conditional draws | 1,125 | 0 |
| Full-model regressions | 6 | 24 |

All failures are in `boundary_regression_tests.jl`, for both likelihood types:

- With `beta=0.3`, `comp_llrs!`, `init_state2`, and `update_state2!` each raise a
  `DomainError` at the fully included model: six failed assertions in total.
- With `beta=1.0`, direct posterior ratios are `NaN`; initialization and refresh
  also produce `NaN` conditional inclusion probabilities, current-state
  probabilities, and predictor rates: eighteen failed assertions in total.

Both cases originate in `comp_llrs!` evaluating the impossible addition prior
for `p+1` included predictors. The correct finite values are supplied by exact
enumeration. Exceptions are captured as values solely to let every regression
execute; they still fail ordinary assertions.

## Line coverage

Collect source-only LCOV records, using a unique output filename per process:

```sh
julia --project=. --startup-file=no --threads=1 --code-coverage=@src --code-coverage=/tmp/bassgwas-%p.info test/runtests.jl
```

The original one-thread suite hit 507 source lines; the expanded one-thread
suite hit 563, adding 56 executed lines. The four-thread run reported 556 lines:
seven lines inside the threaded `comp_rates2!` loop were not recorded as hits,
despite the same passing rate assertions. The table uses the one-thread results.
These are Julia-reported executed-line counts, not whole-package coverage
percentages; uncompiled methods need not appear in Julia's LCOV denominator.

| Source | Original hit lines | Expanded hit lines |
| --- | ---: | ---: |
| `ParState.jl` | 7 | 7 |
| `ProbitSampler.jl` | 0 | 22 |
| `QCache.jl` | 32 | 52 |
| `SCache.jl` | 53 | 53 |
| `bvs_funcs.jl` | 381 | 389 |
| `likelihoods.jl` | 12 | 18 |
| `utils.jl` | 22 | 22 |

Newly exercised numerical functions include `remove_active_QC`, `sample_next2`,
`draw_betas`, `draw_Ysigma!`, and the scalar and vectorized standard-normal
likelihood evaluators. The enumeration assertions also independently validate
previously executed posterior/rate code that the original reference tests shared
with production.
