# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using Distributions
using Random
using Statistics

function mlmc_fixture(m=256; T=Float64)
    rng = Xoshiro(91)
    idx_rem = [3, 1, 4]
    U = T[0.2; -0.1; 0.5; 0.3;;]
    S = T[0.7]
    X = T[-1.0 0.2; 0.3 -0.8; 1.1 0.5; -0.4 0.7]
    icept = T(0.4) .* randn(rng, T, m)
    betas = T(0.3) .* randn(rng, T, m, 2)
    us = T(0.2) .* randn(rng, T, m, 1)
    weights = collect(range(T(0.5), T(1.5); length=m))
    (; idx_rem, U, S, X, icept, betas, us, weights)
end

make_mlmc(f; seed=12, weights=f.weights, pilot_samples=16, kwargs...) =
    MLMCEig(f.idx_rem, f.U, f.S, f.X, f.icept, f.betas, f.us, weights;
            rng=Xoshiro(seed), pilot_samples, kwargs...)

# Explicit tables allow independent checks of arithmetic and row ownership.
function table_mlmc(p; posterior_rows=collect(1:size(p, 1)), seed=32, omega=0.25,
        outcomes=nothing, pilot_samples=16, M0=1)
    m = size(p, 1)
    if isnothing(outcomes)
        rng = Xoshiro(seed)
        outcomes = [rand(rng) < p[j, i] ? 1 : 0 for j in 1:m, i in axes(p, 2)]
    end
    MLMCEig(log.(p), log1p.(-p), outcomes, posterior_rows, Int[],
            omega, pilot_samples, M0, 0.0, 0.0, 0.0, Int[], -1)
end

# Direct products, with no package likelihood or moment helpers.
function direct_level_samples(ec, design, allocations)
    blocks = BASSGWAS.partition_draws(allocations, ec)
    samples = [Float64[] for _ in allocations]
    for i in eachindex(allocations), n in 1:allocations[i]
        rows = blocks[(i - 1, n)]
        outer, inner = rows[1], rows[2:end]
        y = ec.outcomes[outer, design]
        prob(row) = prod(exp(y[k] == 1 ? ec.lp_success[row, idx] : ec.lp_fail[row, idx])
                         for (k, idx) in enumerate(design))
        marginal(indices) = sum(prob(j) for j in indices) / length(indices)
        value = if i == 1
            log(prob(outer)) - log(marginal(inner))
        else
            half = length(inner) ÷ 2
            (log(marginal(inner[1:half])) + log(marginal(inner[half+1:end]))) / 2 -
                log(marginal(inner))
        end
        push!(samples[i], value)
    end
    samples
end

# Ordinary Monte Carlo moments, independently calculated from direct likelihoods.
function direct_level_moments(ec, design, allocations)
    samples = direct_level_samples(ec, design, allocations)
    mean.(samples), var.(samples)
end

@testset "MLMC constructor and immutable interface" begin
    for T in (Float32, Float64)
        f = mlmc_fixture(; T)
        snapshot = deepcopy(f)
        ec = make_mlmc(f)
        same = make_mlmc(f)
        lp1, lp0 = postpred(f.U, f.S, f.X, f.icept, f.betas, f.us)
        @test ec isa EIGest
        @test ec.lp_success == lp1[ec.posterior_rows, f.idx_rem]
        @test ec.lp_fail == lp0[ec.posterior_rows, f.idx_rem]
        @test ec.outcomes isa Matrix{Int}
        @test all(y -> y == 0 || y == 1, ec.outcomes)
        @test !hasproperty(ec, :draw_order)
        @test length(ec.posterior_rows) == length(f.weights)
        @test all(j -> 1 <= j <= length(f.weights), ec.posterior_rows)
        @test length(unique(ec.posterior_rows)) < length(f.weights)
        @test !hasproperty(ec, :Z)
        @test f == snapshot
        @test ec.eig == ec.bias == ec.accuracy == 0
        @test isempty(ec.allocations) && ec.max_level == -1
        @test all(getfield(ec, field) == getfield(same, field) for field in fieldnames(typeof(ec)))
        before = deepcopy(ec)
        values = eig(ec, Int32[3, 1, 2])
        @test values == eig(ec, Int32[3, 1, 2])
        @test isempty(eig(ec, Int[]))
        updated = update_eig(ec, Int32(1))
        @test updated.eig == values[2]
        @test updated.eig == only(eig(updated, [1]))
        @test update_eig(updated, 1) === updated
        @test all(getfield(ec, field) == getfield(before, field) for field in fieldnames(typeof(ec)))
        @test update_eig(updated, 3).eig == update_eig(update_eig(ec, 3), 1).eig
        @test BASSGWAS.comp_eig(ec, [3, 1]) == update_eig(updated, 3).eig
        @test BASSGWAS.comp_eig(ec, Int[]) == 0
        result = optimize_design(ec, f.idx_rem, 2; idx_fixed=[f.idx_rem[2]])
        @test result.batch[1] == f.idx_rem[2]
        positions = [findfirst(==(idx), f.idx_rem) for idx in result.batch]
        @test result.obj == BASSGWAS.comp_eig(ec, positions)
        @test length(unique(result.batch)) == 2
        @test_throws BoundsError eig(ec, [0])
        @test_throws BoundsError update_eig(ec, 4)
        scaled = make_mlmc(f; weights=8 .* f.weights)
        @test ec.posterior_rows == scaled.posterior_rows
        @test values == eig(scaled, Int32[3, 1, 2])
    end
    f = mlmc_fixture(20)
    @test_throws ArgumentError make_mlmc(mlmc_fixture(3))
    @test_throws DimensionMismatch make_mlmc(f; weights=ones(19))
    for bad in (0.0, -1.0, Inf, NaN)
        weights = copy(f.weights)
        weights[1] = bad
        @test_throws ArgumentError make_mlmc(f; weights)
    end
    for omega in (0, 1, Inf, NaN)
        @test_throws ArgumentError make_mlmc(f; omega)
    end
    @test_throws ArgumentError make_mlmc(f; pilot_samples=1)
    large = make_mlmc(f; weights=1e308 .* f.weights)
    @test length(large.posterior_rows) == 20 && all(j -> 1 <= j <= 20, large.posterior_rows)
end

@testset "Configurable MLMC base inner sample count" begin
    f = mlmc_fixture()
    default = make_mlmc(f)
    @test default.M0 == 1
    for invalid in (0, -1, -2, 3, 6, 12)
        @test_throws "M0 must be a positive power of two" make_mlmc(f; M0=invalid)
    end
    @test_throws ArgumentError make_mlmc(f; M0=256)
    @test_throws OverflowError BASSGWAS.mlmc_costs(64, 2)
    for M0 in (1, 2, 4, 8)
        ec = make_mlmc(f; M0=Int32(M0))
        @test ec.M0 === M0
        @test ec.posterior_rows == default.posterior_rows
        @test ec.outcomes == default.outcomes
        @test update_eig(ec, 1).M0 == M0
        @test eig(ec, [1, 2]) == eig(make_mlmc(f; M0), [1, 2])
        @test BASSGWAS.mlmc_costs(4, M0) == [M0 * 2^l + 1 for l in 0:3]
        @test_throws ArgumentError make_mlmc(mlmc_fixture(2(M0 + 1) - 1); M0)
        minimal = make_mlmc(mlmc_fixture(2(M0 + 1)); M0, pilot_samples=2)
        state = update_eig(minimal, 1)
        @test state.allocations == [2]
        @test state.max_level == 0
        @test state.bias == state.accuracy == Inf

        # Check the full marginal and equal halves directly for each base size.
        p = reshape(collect(range(0.1, 0.9; length=8M0)), :, 1)
        table = table_mlmc(p; M0)
        for level in 0:3
            n = M0 * 2^level
            rows = collect(1:n)
            actual = BASSGWAS.mlmc_inner([1], [1], rows, level, table)
            expected = if level == 0
                -log(mean(p[rows, 1]))
            else
                half = n ÷ 2
                (log(mean(p[1:half, 1])) + log(mean(p[half+1:n, 1]))) / 2 -
                    log(mean(p[rows, 1]))
            end
            @test actual ≈ expected
            level > 0 && @test actual <= 0
            @test_throws DimensionMismatch BASSGWAS.mlmc_inner([1], [1], rows[1:end-1], level, table)
        end
    end
end

@testset "Weighted posterior resampling at initialization" begin
    m = 4096
    f = mlmc_fixture(m)
    weights = [fill(9.0, m ÷ 2); ones(m ÷ 2)]
    ec = make_mlmc(f; weights)
    # The first half has 90% posterior mass. This rejects uniform resampling.
    count_first = count(<=(m ÷ 2), ec.posterior_rows)
    @test abs(count_first - 0.9m) < 6sqrt(m * 0.9 * 0.1)
    @test length(unique(ec.posterior_rows)) < m
    lp1, lp0 = postpred(f.U, f.S, f.X, f.icept, f.betas, f.us)
    @test ec.lp_success == lp1[ec.posterior_rows, f.idx_rem]
    @test ec.lp_fail == lp0[ec.posterior_rows, f.idx_rem]

    # With essentially all mass on one row, source rows repeat but outcomes
    # must be sampled anew for each copy, rather than copying one outcome row.
    concentrated = make_mlmc(f; weights=[1.0; fill(1e-300, m - 1)])
    @test all(==(1), concentrated.posterior_rows)
    @test all(concentrated.lp_success[:, i] == fill(lp1[1, idx], m)
              for (i, idx) in enumerate(f.idx_rem))
    for (i, idx) in enumerate(f.idx_rem)
        probability = exp(lp1[1, idx])
        successes = sum(concentrated.outcomes[:, i])
        @test 0 < successes < m
        @test abs(successes - m * probability) < 6sqrt(m * probability * (1 - probability))
    end
    counts = BASSGWAS.allocate_draws(3, ones(3), concentrated)
    blocks = BASSGWAS.partition_draws(counts, concentrated)
    entries = reduce(vcat, values(blocks))
    @test length(unique(entries)) == length(entries)
    @test length(unique(concentrated.posterior_rows[entries])) == 1
    @test update_eig(concentrated, 1).eig ≈ 0 atol=1e-12
    original = deepcopy(concentrated)
    @test eig(concentrated, [1, 2]) == eig(concentrated, [1, 2])
    @test all(getfield(concentrated, field) == getfield(original, field)
              for field in fieldnames(typeof(concentrated)))
end

@testset "Unweighted MLMC arithmetic and numerical stability" begin
    p = [0.1 0.2; 0.3 0.4; 0.7 0.6; 0.9 0.8]
    ec = table_mlmc(p)
    @test BASSGWAS.mlmc_inner([1, 0], [1, 2], [2], 0, ec) ≈ -log(0.3 * 0.6)
    a = sum(p[1:2, 1] .* (1 .- p[1:2, 2])) / 2
    b = sum(p[3:4, 1] .* (1 .- p[3:4, 2])) / 2
    correction = BASSGWAS.mlmc_inner([1, 0], [1, 2], [1, 2, 3, 4], 2, ec)
    full = sum(p[:, 1] .* (1 .- p[:, 2])) / 4
    @test correction ≈ (log(a) + log(b)) / 2 - log(full)
    @test correction <= 0
    @test_throws DimensionMismatch BASSGWAS.mlmc_inner([1], [1], [1, 2], 0, ec)
    @test_throws DimensionMismatch BASSGWAS.mlmc_inner([1], [1, 2], [1], 0, ec)
    @test_throws ArgumentError BASSGWAS.mlmc_inner([1], [1], [1], -1, ec)

    large = table_mlmc(repeat(p, 10, 1))
    allocations = [5, 3, 2]
    samples = direct_level_samples(large, [1, 2], allocations)
    means, variances = BASSGWAS.mlmc_moments([1, 2], allocations, large)
    @test means ≈ mean.(samples)
    @test variances ≈ var.(samples)
    @test means[1] ≈ sum(samples[1]) / allocations[1]

    extreme = table_mlmc(fill(0.5, 4, 2))
    extreme.lp_success .= [-1000.0 -1000; -1001 -1001; -2000 -2000; -2001 -2001]
    @test BASSGWAS.mlmc_inner([1, 1], [1, 2], [1], 0, extreme) == 2000
    @test BASSGWAS.mlmc_inner([1, 1], [1, 2], [1, 2, 3, 4], 2, extreme) ≈ -1000 + log(2)
    extreme.lp_success .= -1000
    @test BASSGWAS.mlmc_inner([1, 1], [1, 2], [1, 2, 3, 4], 2, extreme) == 0
    extreme.lp_success[3:4, :] .-= 1e-6
    tiny = BASSGWAS.mlmc_inner([1, 1], [1, 2], [1, 2, 3, 4], 2, extreme)
    @test tiny < 0 && isapprox(tiny, -5e-13; rtol=1e-6)
end

@testset "Explicit full-level coupling and fixed-pool telescoping" begin
    source_p = [0.1 0.2; 0.3 0.4; 0.7 0.6; 0.9 0.8; 0.45 0.55]
    outcomes = [1 0; 1 1; 0 0; 0 1; 1 0]
    for posterior_rows in ([1, 2, 3, 4, 5], [1, 1, 3, 3, 5])
        p = source_p[posterior_rows, :]
        ec = table_mlmc(p; posterior_rows, outcomes)
        expected_full = zeros(BigFloat, 3)
        expected_correction = zeros(BigFloat, 3)
        for level in 0:2, outer in 1:5
            outcome = outcomes[outer, :]
            likelihoods = [prod(BigFloat(outcome[k] == 1 ? p[j, k] : 1 - p[j, k])
                                for k in 1:2) for j in 1:5]
            # Evaluate P_l directly, including its outer log likelihood, with
            # high precision and the full inner block (no half-block shortcut).
            full_estimate(rows) = log(likelihoods[outer]) -
                log(sum(likelihoods[j] for j in rows) / length(rows))
            others = setdiff(1:5, [outer])
            ninner = 2^level
            full_sum, correction_sum, count = BigFloat(0), BigFloat(0), 0
            # Enumerate every ordered, without-replacement block conditional on
            # this outer row. Its fixed outcome is the same in all three terms.
            for rows in Iterators.product(ntuple(_ -> others, ninner)...)
                length(unique(rows)) == ninner || continue
                full = full_estimate(rows)
                actual = BASSGWAS.mlmc_inner(outcome, [1, 2], collect(rows), level, ec)
                if level == 0
                    actual += sum(outcome[k] == 1 ? ec.lp_success[outer, k] :
                                  ec.lp_fail[outer, k] for k in 1:2)
                    reference = full
                else
                    half = ninner ÷ 2
                    coarse_a = full_estimate(rows[1:half])
                    coarse_b = full_estimate(rows[half+1:end])
                    reference = full - (coarse_a + coarse_b) / 2
                    @test actual <= 0
                end
                @test isapprox(actual, reference; atol=2e-14, rtol=2e-14)
                full_sum += full
                correction_sum += actual
                count += 1
            end
            expected_full[level+1] += full_sum / (5count)
            expected_correction[level+1] += correction_sum / (5count)
        end
        @test isapprox(expected_correction[1], expected_full[1]; atol=2e-14)
        for level in 1:2
            @test isapprox(expected_correction[level+1],
                           expected_full[level+1] - expected_full[level]; atol=2e-14)
            @test isapprox(sum(expected_correction[1:level+1]),
                           expected_full[level+1]; atol=2e-14)
        end
    end
end

# Enumerate all attainable budgets independently of the allocator.
function feasible_costs(costs, budget)
    reachable = Set([2sum(costs)])
    for total in minimum(reachable):budget
        if total in reachable
            for cost in costs
                total + cost <= budget && push!(reachable, total + cost)
            end
        end
    end
    reachable
end

@testset "MLMC allocation and disjoint partitions" begin
    rng = Xoshiro(100)
    for M0 in (1, 2, 4, 8, 16), m in 4:100, levels in 1:4
        costs = [M0 * 2^l + 1 for l in 0:levels-1]
        2sum(costs) <= m || continue
        ec = table_mlmc(fill(0.5, m, 1); M0)
        for v in (zeros(levels), ones(levels), [1.0; zeros(levels-1)],
                  [zeros(levels-1); 1.0], 10.0 .^ (6 .* randn(rng, levels)))
            counts = BASSGWAS.allocate_draws(levels, v, ec)
            @test all(>=(2), counts)
            @test sum(costs .* counts) == maximum(feasible_costs(costs, m))
            @test counts == BASSGWAS.allocate_draws(levels, 7 .* v, ec)
            blocks = BASSGWAS.partition_draws(counts, ec)
            rows = reduce(vcat, values(blocks))
            @test length(blocks) == sum(counts)
            @test length(unique(rows)) == length(rows)
            @test all(1 <= row <= m for row in rows)
            @test all(length(blocks[(l, n)]) == M0 * 2^l + 1
                      for l in 0:levels-1 for n in 1:counts[l+1])
            @test reduce(vcat, [blocks[(l, n)] for l in 0:levels-1
                               for n in 1:counts[l+1]]) == collect(1:sum(costs .* counts))
        end
    end
    ec = table_mlmc(fill(0.5, 2000, 1))
    counts = BASSGWAS.allocate_draws(3, [8.0, 3.0, 1.25], ec)
    @test isapprox(counts ./ counts[3], [4, 2, 1]; atol=0.05)
    @test_throws ArgumentError BASSGWAS.allocate_draws(0, Float64[], ec)
    @test_throws DimensionMismatch BASSGWAS.allocate_draws(2, [1.0], ec)
    @test_throws ArgumentError BASSGWAS.allocate_draws(2, [-1.0, 1.0], ec)
    @test_throws ArgumentError BASSGWAS.allocate_draws(2, [Inf, 1.0], ec)
    @test_throws ArgumentError BASSGWAS.allocate_draws(10, ones(10), ec)
    @test_throws ArgumentError BASSGWAS.partition_draws([1001], ec)
    @test_throws ArgumentError BASSGWAS.partition_draws([-1], ec)
    @test_throws ArgumentError BASSGWAS.partition_draws([2.0], ec)
end

# Reference uses explicit disjoint blocks and direct likelihood products,
# without the implementation's upfront-pass or single-level moment helpers.
function direct_allocation_moments(ec, design)
    means, variances = Float64[], Float64[]
    level = 0
    while size(ec.outcomes, 1) ÷ (ec.M0 * 2^level + 1) >= ec.pilot_samples
        count = size(ec.outcomes, 1) ÷ (ec.M0 * 2^level + 1)
        allocations = [zeros(Int, level); count]
        samples = last(direct_level_samples(ec, design, allocations))
        push!(means, mean(samples))
        push!(variances, var(samples))
        level += 1
    end
    means, variances
end

@testset "Upfront full-pool allocation moments" begin
    for M0 in (1, 2, 4, 8), pilot_samples in (2, 3, 16)
        budgets = unique([max(2(M0 + 1), pilot_samples * cost + offset)
                          for cost in (M0 + 1, 2M0 + 1, 4M0 + 1, 8M0 + 1) for offset in (-1, 0, 1)])
        for m in budgets
            ec = make_mlmc(mlmc_fixture(m); pilot_samples, M0)
            if m ÷ (M0 + 1) < pilot_samples
                @test_throws ArgumentError BASSGWAS.mlmc_allocation_moments([1], ec)
                @test_throws "lower pilot_samples" eig(ec, [1])
                @test_throws ArgumentError update_eig(ec, 1)
                continue
            end
            actual = BASSGWAS.mlmc_allocation_moments([1, 2], ec)
            expected = direct_allocation_moments(ec, [1, 2])
            @test actual[1] ≈ expected[1]
            @test actual[2] ≈ expected[2]
            @test length(actual[1]) == length(actual[2]) == length(expected[1])
            used = Vector{Int}[]
            for level in 0:length(actual[1])-1
                count = m ÷ (ec.M0 * 2^level + 1)
                @test count >= pilot_samples
                allocations = [zeros(Int, level); count]
                blocks = BASSGWAS.partition_draws(allocations, ec)
                rows = reduce(vcat, [blocks[(level, n)] for n in 1:count])
                @test rows == collect(1:count*(M0*2^level+1))
                @test length(unique(rows)) == length(rows)
                @test m - length(rows) < M0 * 2^level + 1
                push!(used, rows)
            end
            @test m ÷ (M0 * 2^length(actual[1]) + 1) < pilot_samples
            if length(used) > 1
                @test !isempty(intersect(used[1], used[2]))
            end
            @test actual == BASSGWAS.mlmc_allocation_moments([1, 2], ec)
        end
    end

    ec = make_mlmc(mlmc_fixture(83))
    frozen = BASSGWAS.mlmc_allocation_moments([1], ec)
    @test length(frozen[1]) == length(frozen[2]) == 3
    samples = only(direct_level_samples(ec, [1], [41]))
    @test frozen[1][1] ≈ mean(samples)
    @test frozen[2][1] ≈ var(samples)
    @test !isapprox(frozen[2][1], var(only(direct_level_samples(ec, [1], [16]))))
    snapshot = deepcopy(frozen)
    for levels in 1:3
        counts = BASSGWAS.allocate_draws(levels, @view(frozen[2][1:levels]), ec)
        BASSGWAS.mlmc_moments([1], counts, ec)
        @test frozen == snapshot
    end
    # Level two consumes the first 80 entries, so changes to its trailing
    # incomplete block cannot affect its upfront moments.
    changed = deepcopy(ec)
    changed.lp_success[81:83, :] .= -200
    changed.lp_fail[81:83, :] .= -100
    new_moments = BASSGWAS.mlmc_allocation_moments([1], changed)
    for i in 1:2
        @test new_moments[i][3] == frozen[i][3]
        @test new_moments[i][1] != frozen[i][1]
        @test BASSGWAS.mlmc_allocation_moments([2], ec)[i] ≈ direct_allocation_moments(ec, [2])[i]
    end
    @test BASSGWAS.mlmc_allocation_moments([2], ec) != frozen

    # Level two has two pilot realizations but cannot fit jointly with the
    # required two realizations on each of levels zero and one.
    small = make_mlmc(mlmc_fixture(10); pilot_samples=2)
    @test length(first(BASSGWAS.mlmc_allocation_moments([1], small))) == 3
    @test update_eig(small, 1).max_level == 1
    # Conversely, the pilot minimum can limit levels before the joint budget.
    limited = make_mlmc(mlmc_fixture(47))
    @test length(first(BASSGWAS.mlmc_allocation_moments([1], limited))) == 1
    @test update_eig(limited, 1).max_level == 0
end

@testset "MLMC minimum-RMSE level selection and diagnostics" begin
    @test BASSGWAS.mlmc_bias([1.0], [1.0]) == Inf
    @test BASSGWAS.mlmc_bias([1.0, -0.25], [1.0, 1.0]) == 0.25
    @test BASSGWAS.mlmc_bias([1.0, -0.5, -0.25, -0.125], ones(4)) ≈ 0.125
    @test BASSGWAS.mlmc_bias([1.0, -0.25, -0.0625, -0.015625], ones(4)) ≈ 0.015625 / 3
    @test BASSGWAS.mlmc_bias([1.0, -0.125, -0.25], ones(3)) == Inf
    @test BASSGWAS.mlmc_bias([1.0, -0.25, -0.25], ones(3)) == Inf
    @test BASSGWAS.mlmc_bias([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0
    for m in (4, 5, 9)
        state = update_eig(make_mlmc(mlmc_fixture(m); pilot_samples=2), 1)
        @test state.max_level == 0
        @test state.bias == state.accuracy == Inf
        @test state.allocations == [m ÷ 2]
    end
    constant = update_eig(table_mlmc(fill(0.5, 200, 2)), 1)
    @test constant.eig == constant.bias == constant.accuracy == 0
    # Every positive level ties at zero predicted RMSE; prefer level one.
    @test constant.max_level == 1
    @test sum(BASSGWAS.mlmc_costs(2, constant.M0) .* constant.allocations) == 200

    # Independently enumerate every feasible maximum level using full-pool
    # moments for selection, then verify diagnostics on the final allocation.
    for M0 in (1, 2, 4, 8), m in (20, 47, 83, 256, 1024), design in ([1], [1, 2])
        ec = make_mlmc(mlmc_fixture(m); pilot_samples=2, M0)
        allocation_means, allocation_variances = direct_allocation_moments(ec, design)
        candidates = []
        for level in 0:length(allocation_variances)-1
            costs = [M0 * 2^l + 1 for l in 0:level]
            2sum(costs) <= m || break
            mu, v = allocation_means[1:level+1], allocation_variances[1:level+1]
            counts = BASSGWAS.allocate_draws(level + 1, v, ec)
            bias = 4 * BASSGWAS.mlmc_bias(mu, v)
            predicted_rmse = hypot(sqrt(sum(v ./ counts)), bias)
            push!(candidates, (; level, counts, predicted_rmse))
        end
        expected = candidates[argmin([c.predicted_rmse for c in candidates])]
        result = BASSGWAS.mlmc_estimate(design, ec)
        @test result.max_level == expected.level
        @test result.allocations == expected.counts
        mu, v = direct_level_moments(ec, design, expected.counts)
        bias = BASSGWAS.mlmc_bias(mu, v)
        @test result.eig ≈ sum(mu)
        @test result.bias ≈ bias
        @test result.accuracy ≈ hypot(sqrt(sum(v ./ expected.counts)), bias)
        # Neither the former bias tolerance nor per-trial final moments affect
        # the chosen allocation or the resulting final diagnostics.
        other = make_mlmc(mlmc_fixture(m); pilot_samples=2, omega=1e-20, M0)
        @test BASSGWAS.mlmc_estimate(design, other) == result
    end
end

@testset "MLMC agreement with exact weighted EIG" begin
    errors = Dict{Tuple{Int,Int,Int},Float64}()
    for M0 in (1, 4), m in (4096, 16384)
        icept = repeat([-0.7, 0.7], m ÷ 2)
        weights = repeat([0.5, 1.5], m ÷ 2)
        args = ([1, 2], zeros(2, 1), ones(1), zeros(2, 1), icept, zeros(m, 1), zeros(m, 1))
        rb = CIDEIG(args..., weights ./ sum(weights))
        for design in ([1], [1, 2])
            exact = 0.0
            for outcome in Iterators.product(ntuple(_ -> (0, 1), length(design))...)
                likelihoods = [prod(outcome[k] == 1 ? rb.p_success[j, idx] : rb.p_fail[j, idx]
                                    for (k, idx) in enumerate(design)) for j in 1:m]
                marginal = sum(weights .* likelihoods) / sum(weights)
                exact += sum(weights .* likelihoods .* log.(likelihoods ./ marginal)) / sum(weights)
            end
            @test BASSGWAS.comp_eig(rb, design) ≈ exact
            estimates = [BASSGWAS.comp_eig(MLMCEig(args..., weights; rng=Xoshiro(seed), M0), design)
                         for seed in 1:40]
            # Six Monte Carlo standard errors plus a finite-level bias allowance.
            @test abs(mean(estimates) - exact) < 6std(estimates) / sqrt(length(estimates)) + 0.01
            errors[(M0, m, length(design))] = sqrt(mean((estimates .- exact).^2))
            if m == 16384
                @test abs(mean(estimates) - exact) < 0.03
                @test errors[(M0, m, length(design))] < 0.075
            end
        end
    end
    for M0 in (1, 4), size in 1:2
        @test errors[(M0, 16384, size)] < errors[(M0, 4096, size)]
    end
end
