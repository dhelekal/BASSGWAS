# Copyright (C) 2025–2026 David Helekal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
    MLMCEig(idx_rem, Ufull, Sfull, X, icept, betas, us, weights;
            rng=Random.default_rng(), omega=4, pilot_samples=48, M0=1)

Fixed-pool antithetic multilevel Monte Carlo estimator of expected information
gain. Candidate positions refer to the ordering of `idx_rem`, as in `CIDEIG`.
The constructor computes predictive log probabilities and uses stratified
resampling of posterior rows according to their importance weights, allowing
repeated source rows. The pool has the same size as the input pool; `posterior_rows`
records its source indices. It shuffles these indices once before sampling
independent integer outcomes for every pool entry (including repeated source rows).
Subsequent scoring is deterministic and uses only unweighted Monte Carlo means.

Weights must be finite and strictly positive and are used only for resampling.
`M0` must be a positive integer; level `l` uses `M0 * 2^l` inner
entries plus one outer entry. The pool must support at least two level-zero
realizations. `omega` must be finite and strictly positive; it multiplies the
estimated bias in both level selection and reported accuracy. `pilot_samples`
(at least two) is the minimum number of complete realizations required to estimate
a level's allocation moments.
Each eligible level uses the entire pool of complete blocks in one upfront pass
per design evaluation. Evaluating a design requires at least `(M0 + 1) * pilot_samples`
pool entries; otherwise lower `pilot_samples` or supply more posterior draws.

`eig`, `bias`, and `accuracy` describe the selected design. `bias` is the unscaled
truncation-bias estimate fitted to the upfront pilot means. `accuracy` is the
bias-adjusted RMSE `sqrt(sampling_variance + (omega * bias)^2)`; with `omega=1`
it is the ordinary estimated RMSE. Level selection minimizes this same objective
using the frozen upfront moments. Reporting retains the selected pilot bias fit
and estimates sampling variance from the final allocation. `allocations[l+1]`
counts realizations at level `l`, through `max_level`. Upfront moment estimates may
share pool entries across levels, but not within a level. Every final realization
has its own outer entry and disjoint inner entries. Distinct entries may originate
from the same source posterior row.
These diagnostics follow Algorithm 1 of Goda et al. (arXiv:1811.07546v5); they
are heuristic for a finite, potentially correlated posterior pool. Stratified
resampling also introduces dependence between pool entries.
"""
struct MLMCEig{T <: AbstractFloat} <: EIGest
    lp_success::Matrix{T}
    lp_fail::Matrix{T}
    outcomes::Matrix{Int}
    posterior_rows::Vector{Int}
    selected::Vector{Int}
    omega::T
    pilot_samples::Int
    M0::Int
    eig::T
    bias::T
    accuracy::T
    allocations::Vector{Int}
    max_level::Int
end

"""
    mlmc_stratified_rows(rng, weights)

Resample one source index per equal-probability stratum, then shuffle the indices
so consecutive MLMC blocks do not inherit source-row ordering. The constructor
validates that weights are finite, positive, and nonempty.
"""
function mlmc_stratified_rows(rng::AbstractRNG, weights::AbstractVector{T}) where {T <: AbstractFloat}
    F = promote_type(Float64, T)
    # Scale before accumulating, then normalize the cumulative weights once.
    cdf = cumsum(F.(weights) ./ F(maximum(weights)))
    cdf ./= last(cdf)
    cdf[end] = one(F)
    m = length(weights)
    rows = Vector{Int}(undef, m)
    row = 1
    for i in 1:m
        # Rounding must not move a draw onto its stratum's upper endpoint.
        u = min((F(i - 1) + rand(rng, F)) / m, prevfloat(F(i) / m))
        while u >= cdf[row]
            row += 1
        end
        rows[i] = row
    end
    return shuffle!(rng, rows)
end

function MLMCEig(
        idx_rem::AbstractVector{<:Integer}, Ufull::AbstractMatrix{T},
        Sfull::AbstractVector{T}, X::AbstractMatrix{T}, icept::AbstractVector{T},
        betas::AbstractMatrix{T}, us::AbstractMatrix{T}, weights::AbstractVector{T};
        rng::AbstractRNG = Random.default_rng(), omega::Real = 4,
        pilot_samples::Integer = 48, M0::Integer = 1
    ) where {T <: AbstractFloat}
    m = length(icept)
    M0 > 0 || throw(ArgumentError("M0 must be a positive integer"))
    M0 <= m ÷ 2 - 1 || throw(
        ArgumentError(
            "MLMC requires at least two level-zero realizations of M0 + 1 posterior draws each"
        )
    )
    length(weights) == m || throw(DimensionMismatch("one weight is required per draw"))
    all(w -> isfinite(w) && w > 0, weights) ||
        throw(ArgumentError("posterior weights must be finite and strictly positive"))
    isfinite(T(omega)) && T(omega) > 0 ||
        throw(ArgumentError("omega must be finite and strictly positive"))
    pilot_samples >= 2 || throw(ArgumentError("pilot_samples must be at least two"))

    lp1, lp0 = postpred(Ufull, Sfull, X, icept, betas, us)
    F = eltype(lp1)
    posterior_rows = mlmc_stratified_rows(rng, weights)
    lp1, lp0 = lp1[posterior_rows, idx_rem], lp0[posterior_rows, idx_rem]
    outcomes = Matrix{Int}(undef, size(lp1))
    for i in eachindex(outcomes)
        outcomes[i] = rand(rng, F) < exp(lp1[i]) ? 1 : 0
    end
    return MLMCEig(
        lp1, lp0, outcomes, posterior_rows, Int[], F(omega),
        Int(pilot_samples), Int(M0), zero(F), zero(F), zero(F), Int[], -1
    )
end

"""Evaluate the outcome stored at pool entry `outer` under likelihood row `row`."""
function mlmc_loglik(outer::Integer, design_idx, row, ec::MLMCEig{T}) where {T}
    checkbounds(ec.outcomes, outer, :)
    value = zero(T)
    for idx in design_idx
        y = ec.outcomes[outer, idx]
        value += y == 1 ? ec.lp_success[row, idx] : ec.lp_fail[row, idx]
    end
    return value
end

"""Log mean likelihood over `rows` for the outcome stored at pool entry `outer`."""
function mlmc_logmean(outer::Integer, design_idx, rows, ec::MLMCEig{T}) where {T}
    checkbounds(ec.outcomes, outer, :)
    total = T(-Inf)
    for row in rows
        total = logaddexp(total, mlmc_loglik(outer, design_idx, row, ec))
    end
    return total - log(T(length(rows)))
end

"""
    mlmc_inner(outer, design_idx, draw_idx, level, ec)

Use the outcome stored at resampled-pool entry `outer`, not its source posterior
row, for every likelihood in the full block and both antithetic halves.
Use exactly `M0 * 2^level` inner rows. At level zero return the outer log likelihood
minus the log marginal estimate. At higher levels return the
antithetic correction: the average half-block log marginal minus the full-block
log marginal. All resampled entries have equal weight, so the full marginal is
the arithmetic mean of its halves and the correction is nonpositive.
"""
function mlmc_inner(
        outer::Integer, design_idx::AbstractVector,
        draw_idx::AbstractVector, level::Integer, ec::MLMCEig{T}
    ) where {T}
    level >= 0 || throw(ArgumentError("level must be nonnegative"))
    length(draw_idx) == Base.checked_mul(ec.M0, Base.checked_pow(2, level)) ||
        throw(DimensionMismatch("level l requires M0 * 2^l inner rows"))
    checkbounds(ec.outcomes, outer, :)

    if level == 0
        return mlmc_loglik(outer, design_idx, outer, ec) -
            mlmc_logmean(outer, design_idx, draw_idx, ec)
    end

    half = length(draw_idx) ÷ 2
    rows_a, rows_b = @view(draw_idx[1:half]), @view(draw_idx[(half + 1):end])
    a = mlmc_logmean(outer, design_idx, rows_a, ec)
    b = mlmc_logmean(outer, design_idx, rows_b, ec)
    # Full marginal = (exp(a) + exp(b))/2. The outer log likelihood cancels in
    # P_l - (P_{l-1}^a + P_{l-1}^b)/2, leaving (a+b)/2 - log(full marginal).
    # Its equivalent -log(cosh((a-b)/2)) avoids cancellation and overflow.
    d = abs(a / 2 - b / 2)
    return d < 1 ? -log1p(2 * sinh(d / 2)^2) : -(d + log1p(exp(-2d)) - log(T(2)))
end

# `levels` is a count: entries correspond to levels 0, ..., levels-1.
function mlmc_costs(levels::Integer, M0::Integer = 1)
    levels >= 1 || throw(ArgumentError("at least one level is required"))
    return [Base.checked_add(Base.checked_mul(M0, Base.checked_pow(2, l)), 1) for l in 0:(levels - 1)]
end

"""
    allocate_draws(levels, variances, ec)

Allocate realization counts (not individual rows) with cost `M0 * 2^l + 1`, at least
two per level, and continuous targets proportional to `sqrt(V_l / C_l)`.
Floor the continuous targets, then greedily add affordable blocks by variance
reduction per row. Never remove useful blocks to spend the remaining budget;
unused rows are allowed. This is a rounding heuristic, not an exact integer
optimizer. Zero-variance levels retain the minimum allocation.
"""
function allocate_draws(levels::Integer, variances::AbstractVector, ec::MLMCEig{T}) where {T}
    costs = mlmc_costs(levels, ec.M0)
    length(variances) == levels || throw(DimensionMismatch("one variance is required per level"))
    all(v -> isfinite(v) && v >= 0, variances) ||
        throw(ArgumentError("variances must be finite and nonnegative"))
    budget = size(ec.outcomes, 1)
    sum(costs) <= budget ÷ 2 || throw(ArgumentError("pool cannot support two realizations per level"))
    v = T.(variances)
    all(iszero, v) && return fill(2, levels)
    v ./= maximum(v)
    rates = sqrt.(v ./ costs)
    counts = fill(2, levels)
    active = collect(1:levels)
    remaining = budget
    # Fix lower-bound levels, then solve the budget constraint on the others.
    while !isempty(active)
        scale = remaining / sum(costs[i] * rates[i] for i in active)
        fixed = [i for i in active if scale * rates[i] < 2]
        if isempty(fixed)
            for i in active
                counts[i] = max(2, floor(Int, scale * rates[i]))
            end
            break
        end
        remaining -= 2sum(costs[i] for i in fixed)
        active = setdiff(active, fixed)
    end
    remaining = budget - sum(costs .* counts)
    while remaining > 0
        best, gain = 0, zero(T)
        for i in eachindex(counts)
            if costs[i] <= remaining
                improvement = v[i] / (costs[i] * T(counts[i]) * (counts[i] + 1))
                if improvement > gain
                    best, gain = i, improvement
                end
            end
        end
        best == 0 && break
        counts[best] += 1
        remaining -= costs[best]
    end
    return counts
end

"""
    partition_draws(allocations, ec)

Map `(level, realization)` to disjoint resampled-pool entry vectors in pool
order. Each block has its outer entry first, then `M0 * 2^level` inner entries.
The indices address the pool, not the possibly repeated source posterior rows.
"""
function partition_draws(allocations::AbstractVector, ec::MLMCEig)
    costs = mlmc_costs(length(allocations), ec.M0)
    all(n -> n isa Integer && n >= 0, allocations) ||
        throw(ArgumentError("allocations must be nonnegative integers"))
    blocks = Dict{Tuple{Int, Int}, Vector{Int}}()
    cursor = 1
    for i in eachindex(allocations), realization in 1:allocations[i]
        cost = costs[i]
        cost <= size(ec.outcomes, 1) - cursor + 1 || throw(ArgumentError("allocation exceeds the draw pool"))
        blocks[(i - 1, realization)] = collect(cursor:(cursor + cost - 1))
        cursor += cost
    end
    return blocks
end

function mlmc_level_moments(design, level, blocks, ec::MLMCEig{T}) where {T}
    mean, m2, count = zero(T), zero(T), 0
    for rows in blocks
        outer = first(rows)
        value = mlmc_inner(outer, design, @view(rows[2:end]), level, ec)
        isfinite(value) || throw(DomainError(value, "nonfinite MLMC contribution; check predictive likelihoods"))
        # Welford's update for ordinary means and unbiased sample variances.
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    end
    count >= 2 || throw(ArgumentError("level moments require at least two realizations"))
    return mean, max(zero(T), m2 / (count - 1))
end

function mlmc_moments(design, allocations, ec::MLMCEig{T}) where {T}
    blocks = partition_draws(allocations, ec)
    means, variances = zeros(T, length(allocations)), zeros(T, length(allocations))
    for i in eachindex(allocations)
        level_blocks = (blocks[(i - 1, n)] for n in 1:allocations[i])
        means[i], variances[i] = mlmc_level_moments(design, i - 1, level_blocks, ec)
    end
    return means, variances
end

"""
    mlmc_allocation_moments(design, ec)

Compute allocation means and variances once using every complete block at each eligible
level. Level `l` uses `floor(M / (M0 * 2^l + 1))` realizations and must have at least
`pilot_samples`. Each level restarts at the beginning of the resampled pool;
entries may be shared between levels, but never within one level's estimate.
"""
function mlmc_allocation_moments(design, ec::MLMCEig{T}) where {T}
    budget = size(ec.outcomes, 1)
    budget ÷ (ec.M0 + 1) >= ec.pilot_samples || throw(
        ArgumentError(
            "pool cannot provide pilot_samples=$(ec.pilot_samples) level-zero realizations; " *
                "lower pilot_samples or supply more posterior draws"
        )
    )
    means, variances = T[], T[]
    level, inner = 0, ec.M0
    while inner + 1 <= budget ÷ ec.pilot_samples
        cost = inner + 1
        count = budget ÷ cost
        blocks = Iterators.partition(1:(count * cost), cost)
        mean, variance = mlmc_level_moments(design, level, blocks, ec)
        push!(means, mean)
        push!(variances, variance)
        level += 1
        inner *= 2
    end
    return means, variances
end

"""
Fit `log2(abs(mean_l)) = intercept - alpha * l` to the finest three nonzero
positive-level pilot means (or all available if fewer). Extrapolate the fitted
magnitude to the requested finest level, including when its observed mean is zero.
With only one nonzero level, retain the first-order `alpha=1` fallback. Without
nonzero corrections the bias is unknown, unless all pilot means and variances
vanish. Nondecaying fits also return `Inf`.
"""
function mlmc_bias(means::AbstractVector{T}, variances::AbstractVector) where {T}
    length(means) == 1 && return T(Inf)
    all(iszero, means) && all(iszero, variances) && return zero(T)
    usable = [i for i in 2:length(means) if means[i] != 0]
    isempty(usable) && return T(Inf)
    finest = usable[max(1, length(usable) - 2):end]
    x = T.(finest .- 1)
    y = log2.(abs.(means[finest]))
    xmean, ymean = sum(x) / length(x), sum(y) / length(y)
    alpha = one(T)
    if length(finest) >= 2
        x .-= xmean
        y .-= ymean
        alpha = -sum(x .* y) / sum(abs2, x)
    end
    isfinite(alpha) && alpha > 0 || return T(Inf)
    log_magnitude = ymean - alpha * (length(means) - 1 - xmean)
    return exp2(log_magnitude) / expm1(alpha * log(T(2)))
end

function mlmc_estimate(design, ec::MLMCEig{T}) where {T}
    budget = size(ec.outcomes, 1)
    allocation_means, allocation_variances = mlmc_allocation_moments(design, ec)
    max_level, spent, inner = -1, 0, ec.M0
    while inner + 1 <= (budget - spent) ÷ 2
        spent += 2(inner + 1)
        max_level += 1
        inner *= 2
    end
    max_level = min(max_level, length(allocation_variances) - 1)
    best = nothing
    for level in 0:max_level
        means = @view allocation_means[1:(level + 1)]
        variances = @view allocation_variances[1:(level + 1)]
        allocations = allocate_draws(level + 1, variances, ec)
        bias = mlmc_bias(means, variances)
        sampling_variance = sum(variances ./ allocations)
        predicted_rmse = hypot(sqrt(sampling_variance), ec.omega * bias)
        if isnothing(best) || predicted_rmse < best.predicted_rmse
            best = (; predicted_rmse, allocations, bias, max_level = level)
        end
    end
    means, variances = mlmc_moments(design, best.allocations, ec)
    bias = best.bias
    sampling_variance = sum(variances ./ best.allocations)
    return (;
        eig = sum(means), bias, accuracy = hypot(sqrt(sampling_variance), ec.omega * bias),
        allocations = best.allocations, max_level = best.max_level, variances = variances,
    )
end

"""Return total MLMC EIG for each candidate addition, without changing `ec`."""
function eig(ec::MLMCEig{T}, idx_set::Vector{I}) where {T, I <: Integer}
    values = Vector{T}(undef, length(idx_set))
    for (i, idx) in enumerate(idx_set)
        checkbounds(ec.lp_success, :, idx)
        values[i] = idx in ec.selected ? ec.eig :
            mlmc_estimate(sort!([ec.selected; Int(idx)]), ec).eig
    end
    return values
end

"""Return a new state with candidate `idx` selected and its MLMC diagnostics."""
function update_eig(ec::MLMCEig, idx::I) where {I <: Integer}
    checkbounds(ec.lp_success, :, idx)
    idx in ec.selected && return ec
    selected = sort!([ec.selected; Int(idx)])
    result = mlmc_estimate(selected, ec)
    println(result)
    return MLMCEig(
        ec.lp_success, ec.lp_fail, ec.outcomes, ec.posterior_rows, selected,
        ec.omega, ec.pilot_samples, ec.M0, result.eig, result.bias, result.accuracy,
        result.allocations, result.max_level
    )
end
