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
            rng=Random.default_rng(), omega=0.25, pilot_samples=16)

Fixed-pool antithetic multilevel Monte Carlo estimator of expected information
gain. Candidate positions refer to the ordering of `idx_rem`, as in `CIDEIG`.
The constructor computes predictive log probabilities, samples integer outcomes,
and fixes a random row order. Subsequent scoring is deterministic and immutable.

Weights must be finite and strictly positive and are normalized globally to sum
to the number of posterior draws. Neither inner nor outer blocks renormalize
them. At least four rows are required. `omega` is the desired fraction of mean
squared error due to bias; `pilot_samples` (at least two) caps pilots per level.

`eig`, `bias`, and `accuracy` describe the selected design; `accuracy` is the
empirical RMSE, including estimated truncation bias. `allocations[l+1]` counts
realizations at level `l`, through `max_level`. Pilot passes may repartition the
pool, but every final realization has its own outer row and disjoint inner rows.
These diagnostics follow Algorithm 1 of Goda et al. (arXiv:1811.07546v5); they
are heuristic for a finite, potentially correlated posterior pool.
"""
struct MLMCEig{T<:AbstractFloat} <: EIGest
    lp_success::Matrix{T}
    lp_fail::Matrix{T}
    outcomes::Matrix{Int}
    Z::Vector{T}
    selected::Vector{Int}
    draw_order::Vector{Int}
    omega::T
    pilot_samples::Int
    eig::T
    bias::T
    accuracy::T
    allocations::Vector{Int}
    max_level::Int
end

function MLMCEig(idx_rem::AbstractVector{<:Integer}, Ufull::AbstractMatrix{T},
        Sfull::AbstractVector{T}, X::AbstractMatrix{T}, icept::AbstractVector{T},
        betas::AbstractMatrix{T}, us::AbstractMatrix{T}, weights::AbstractVector{T};
        rng::AbstractRNG=Random.default_rng(), omega::Real=0.25,
        pilot_samples::Integer=16) where {T<:AbstractFloat}
    m = length(icept)
    m >= 4 || throw(ArgumentError("MLMC requires at least four posterior draws"))
    length(weights) == m || throw(DimensionMismatch("one weight is required per draw"))
    all(w -> isfinite(w) && w > 0, weights) ||
        throw(ArgumentError("posterior weights must be finite and strictly positive"))
    0 < T(omega) < 1 || throw(ArgumentError("omega must lie strictly between zero and one"))
    pilot_samples >= 2 || throw(ArgumentError("pilot_samples must be at least two"))

    lp1, lp0 = postpred(Ufull, Sfull, X, icept, betas, us)
    lp1, lp0 = lp1[:, idx_rem], lp0[:, idx_rem]
    F = eltype(lp1)
    outcomes = Matrix{Int}(undef, size(lp1))
    for i in eachindex(outcomes)
        outcomes[i] = rand(rng, F) < exp(lp1[i]) ? 1 : 0
    end
    # Scale before summing to avoid overflow; this is one global normalization.
    Z = F.(weights) ./ maximum(weights)
    Z .*= F(m) / sum(Z)
    all(>(zero(F)), Z) || throw(ArgumentError("normalized weights underflowed to zero"))
    MLMCEig(lp1, lp0, outcomes, Z, Int[], randperm(rng, m), F(omega),
            Int(pilot_samples), zero(F), zero(F), zero(F), Int[], -1)
end

function mlmc_loglik(outcome, design_idx, row, ec::MLMCEig{T}) where T
    value = zero(T)
    for (y, idx) in zip(outcome, design_idx)
        value += y == 1 ? ec.lp_success[row, idx] : ec.lp_fail[row, idx]
    end
    value
end

function mlmc_logmean(outcome, design_idx, rows, ec::MLMCEig{T}) where T
    total = T(-Inf)
    for row in rows
        total = logaddexp(total, log(ec.Z[row]) + mlmc_loglik(outcome, design_idx, row, ec))
    end
    total - log(T(length(rows)))
end

"""
    mlmc_inner(outcome, design_idx, draw_idx, level, ec)

Use exactly `2^level` inner rows. At level zero return the negative log marginal
estimate (the caller adds the outer log likelihood). At higher levels return the
antithetic correction: the average half-block log marginal minus the full-block
log marginal. The outer row and its weight are handled by the caller.
"""
function mlmc_inner(outcome::AbstractVector, design_idx::AbstractVector,
        draw_idx::AbstractVector, level::Integer, ec::MLMCEig{T}) where T
    level >= 0 || throw(ArgumentError("level must be nonnegative"))
    length(draw_idx) == Base.checked_pow(2, level) ||
        throw(DimensionMismatch("level l requires 2^l inner rows"))
    length(outcome) == length(design_idx) || throw(DimensionMismatch("outcome and design lengths differ"))
    level == 0 && return -mlmc_logmean(outcome, design_idx, draw_idx, ec)

    half = length(draw_idx) ÷ 2
    a = mlmc_logmean(outcome, design_idx, @view(draw_idx[1:half]), ec)
    b = mlmc_logmean(outcome, design_idx, @view(draw_idx[half+1:end]), ec)
    # The full-block log mean is log((exp(a)+exp(b))/2): equal half sizes
    # and globally normalized weights make the full mean their arithmetic mean.
    # For P_l = outer_loglik - full_logmean, the outer term cancels in
    # P_l - (P_{l-1}^a + P_{l-1}^b)/2. The full-level contribution is included by
    # (a+b)/2 - log((exp(a)+exp(b))/2) = -log(cosh((a-b)/2)).
    # Avoid cancellation for similar halves and overflow for very different ones.
    d = abs(a / 2 - b / 2)
    d < 1 ? -log1p(2 * sinh(d / 2)^2) : -(d + log1p(exp(-2d)) - log(T(2)))
end

# `levels` is a count: entries correspond to levels 0, ..., levels-1.
function mlmc_costs(levels::Integer)
    levels >= 1 || throw(ArgumentError("at least one level is required"))
    [Base.checked_add(Base.checked_pow(2, l), 1) for l in 0:levels-1]
end

"""
    allocate_draws(levels, variances, ec)

Allocate realization counts (not individual rows) with cost `2^l + 1`, at least
two per level, and continuous targets proportional to `sqrt(V_l / C_l)`.
Integer rounding spends the maximum feasible row budget. Zero variances receive
the minimum continuous allocation; if all variances vanish, equal surrogate
variances determine the allocation. Rounding may add blocks at any level.
"""
function allocate_draws(levels::Integer, variances::AbstractVector, ec::MLMCEig{T}) where T
    costs = mlmc_costs(levels)
    length(variances) == levels || throw(DimensionMismatch("one variance is required per level"))
    all(v -> isfinite(v) && v >= 0, variances) ||
        throw(ArgumentError("variances must be finite and nonnegative"))
    budget = length(ec.Z)
    sum(costs) <= budget ÷ 2 || throw(ArgumentError("pool cannot support two realizations per level"))
    v = T.(variances)
    if all(iszero, v)
        fill!(v, one(T))
    else
        v ./= maximum(v)
    end
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
    while remaining >= 2
        best, gain = 0, -one(T)
        for i in eachindex(counts)
            if costs[i] <= remaining
                improvement = v[i] / (costs[i] * T(counts[i]) * (counts[i] + 1))
                if improvement > gain
                    best, gain = i, improvement
                end
            end
        end
        counts[best] += 1
        remaining -= costs[best]
    end
    # All costs except level zero are odd. If one row remains, exchange an
    # extra block to fill it, choosing the exchange with least variance cost.
    if remaining == 1 && levels > 1
        best, penalty = 0, T(Inf)
        for i in eachindex(counts)
            counts[i] > 2 || continue
            if i == 1
                change = v[1] / (T(counts[1]) * (counts[1] - 1)) -
                         v[2] / (T(counts[2]) * (counts[2] + 1))
            else
                extra = (costs[i] + 1) ÷ 2
                change = v[i] / (T(counts[i]) * (counts[i] - 1)) -
                         v[1] * extra / (T(counts[1]) * (counts[1] + extra))
            end
            if change < penalty
                best, penalty = i, change
            end
        end
        if best == 1
            counts[1] -= 1
            counts[2] += 1
        elseif best > 1
            counts[best] -= 1
            counts[1] += (costs[best] + 1) ÷ 2
        end
    end
    counts
end

"""
    partition_draws(allocations, ec)

Map `(level, realization)` to disjoint posterior row vectors in the fixed random
order. Each block contains its outer row first, followed by `2^level` inner rows.
"""
function partition_draws(allocations::AbstractVector, ec::MLMCEig)
    costs = mlmc_costs(length(allocations))
    all(n -> n isa Integer && n >= 0, allocations) ||
        throw(ArgumentError("allocations must be nonnegative integers"))
    blocks = Dict{Tuple{Int,Int},Vector{Int}}()
    cursor = 1
    for i in eachindex(allocations), realization in 1:allocations[i]
        cost = costs[i]
        cost <= length(ec.Z) - cursor + 1 || throw(ArgumentError("allocation exceeds the draw pool"))
        blocks[(i - 1, realization)] = ec.draw_order[cursor:cursor+cost-1]
        cursor += cost
    end
    blocks
end

function mlmc_moments(design, allocations, ec::MLMCEig{T}) where T
    blocks = partition_draws(allocations, ec)
    means, variances = zeros(T, length(allocations)), zeros(T, length(allocations))
    for i in eachindex(allocations)
        m2 = zero(T)
        for n in 1:allocations[i]
            rows = blocks[(i - 1, n)]
            outer = first(rows)
            outcome = @view ec.outcomes[outer, design]
            value = mlmc_inner(outcome, design, @view(rows[2:end]), i - 1, ec)
            i == 1 && (value += mlmc_loglik(outcome, design, outer, ec))
            value *= ec.Z[outer]
            isfinite(value) || throw(DomainError(value, "nonfinite MLMC contribution; check predictive likelihoods"))
            delta = value - means[i]
            means[i] += delta / n
            m2 += delta * (value - means[i])
        end
        variances[i] = max(zero(T), m2 / (allocations[i] - 1))
    end
    means, variances
end

function mlmc_bias(means::AbstractVector{T}, variances::AbstractVector) where T
    length(means) == 1 && return T(Inf)
    means[end] == 0 && variances[end] == 0 && return zero(T)
    usable = [i for i in 2:length(means) if means[i] != 0]
    alpha = one(T)
    if length(usable) >= 2
        finest = usable[max(1, length(usable) - 2):end]
        x = T.(finest .- 1)
        y = log2.(abs.(means[finest]))
        x .-= sum(x) / length(x)
        y .-= sum(y) / length(y)
        alpha = -sum(x .* y) / sum(abs2, x)
    end
    alpha > 0 || return T(Inf)
    abs(means[end]) / expm1(alpha * log(T(2)))
end

function mlmc_estimate(design, ec::MLMCEig{T}) where T
    budget = length(ec.Z)
    max_level, spent, inner = -1, 0, 1
    while inner + 1 <= (budget - spent) ÷ 2
        spent += 2(inner + 1)
        max_level += 1
        inner *= 2
    end
    best = nothing
    for level in min(2, max_level):max_level
        costs = mlmc_costs(level + 1)
        pilots = fill(min(ec.pilot_samples, budget ÷ sum(costs)), level + 1)
        _, pilot_variances = mlmc_moments(design, pilots, ec)
        allocations = allocate_draws(level + 1, pilot_variances, ec)
        means, variances = mlmc_moments(design, allocations, ec)
        bias = mlmc_bias(means, variances)
        sampling_variance = sum(variances ./ allocations)
        result = (; eig=sum(means), bias, accuracy=hypot(sqrt(sampling_variance), bias),
                   allocations, max_level=level)
        if isnothing(best) || result.accuracy < best.accuracy
            best = result
        end
        bias <= sqrt(ec.omega / (1 - ec.omega)) * sqrt(sampling_variance) && return result
    end
    best
end

"""Return total MLMC EIG for each candidate addition, without changing `ec`."""
function eig(ec::MLMCEig{T}, idx_set::Vector{I}) where {T,I<:Integer}
    values = Vector{T}(undef, length(idx_set))
    for (i, idx) in enumerate(idx_set)
        checkbounds(ec.lp_success, :, idx)
        values[i] = idx in ec.selected ? ec.eig :
            mlmc_estimate(sort!([ec.selected; Int(idx)]), ec).eig
    end
    values
end

"""Return a new state with candidate `idx` selected and its MLMC diagnostics."""
function update_eig(ec::MLMCEig, idx::I) where I<:Integer
    checkbounds(ec.lp_success, :, idx)
    idx in ec.selected && return ec
    selected = sort!([ec.selected; Int(idx)])
    result = mlmc_estimate(selected, ec)
    MLMCEig(ec.lp_success, ec.lp_fail, ec.outcomes, ec.Z, selected, ec.draw_order,
            ec.omega, ec.pilot_samples, result.eig, result.bias, result.accuracy,
            result.allocations, result.max_level)
end
