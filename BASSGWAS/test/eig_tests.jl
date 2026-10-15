# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using Distributions
using Random

function eig_fixture(; tied=false)
    idx_rem = [6, 2, 5, 3]
    U = reshape([0.2, -0.1, 0.5, 0.3, -0.4, 0.6], :, 1)
    S = [0.7]
    X = [-1.0 0.2; 0.3 -0.8; 1.1 0.5; -0.4 0.7; 0.6 1.2; -0.8 -0.3]
    if tied
        U .= 0.2
        X .= 0.4
    end
    icept = [-0.2, 0.4, -0.5, 0.1, 0.3]
    betas = [0.3 -0.4; -0.5 0.2; 0.7 0.6; -0.1 -0.7; 0.4 0.1]
    us = reshape([0.2, -0.3, 0.1, 0.4, -0.2], :, 1)
    Z = [0.1, 0.2, 0.3, 0.15, 0.25]
    (; idx_rem, U, S, X, icept, betas, us, Z)
end

make_eig(f) = CIDEIG(f.idx_rem, f.U, f.S, f.X, f.icept, f.betas, f.us, f.Z)

# Compute predictive probabilities separately from the package's postpred path.
function reference_probabilities(f)
    p = zeros(length(f.Z), length(f.idx_rem))
    for (i, target) in enumerate(f.idx_rem), j in eachindex(f.Z)
        mean = f.icept[j] + sum(f.X[target, k] * f.betas[j, k] for k in axes(f.X, 2)) +
               sum(f.U[target, k] * sqrt(f.S[k]) * f.us[j, k] for k in eachindex(f.S))
        p[j, i] = cdf(Normal(), mean)
    end
    p
end

# Direct mutual information over all binary patterns and posterior draws,
# without using the estimator's entropy decomposition or likelihood helpers.
function reference_eig(p, Z, selected)
    total = 0.0
    for outcomes in Iterators.product(ntuple(_ -> (false, true), length(selected))...)
        likelihoods = [prod(outcome ? p[j, idx] : 1 - p[j, idx]
                            for (idx, outcome) in zip(selected, outcomes); init=1.0)
                       for j in eachindex(Z)]
        marginal = sum(Z .* likelihoods)
        for j in eachindex(Z)
            if likelihoods[j] > 0
                total += Z[j] * likelihoods[j] * log(likelihoods[j] / marginal)
            end
        end
    end
    total
end

@testset "Rao–Blackwellized estimator" begin
    f = eig_fixture()
    ec = make_eig(f)
    original = deepcopy(ec)
    p = reference_probabilities(f)
    @test ec isa EIGest
    @test ec.p_success ≈ p
    @test ec.p_fail ≈ 1 .- p
    @test eig(ec, Int32[4, 1, 3]) ≈ [reference_eig(p, f.Z, [i]) for i in [4, 1, 3]]

    state = ec
    selected = Int[]
    for idx in [3, 1, 4, 2]
        candidates = setdiff(1:4, selected)
        expected = [reference_eig(p, f.Z, [selected; i]) for i in candidates]
        @test isapprox(eig(state, candidates), expected; rtol=1e-8, atol=1e-10)
        previous = state
        snapshot = deepcopy(previous)
        state = update_eig(state, Int32(idx))
        push!(selected, idx)
        @test isapprox(state.eig, reference_eig(p, f.Z, selected); rtol=1e-8, atol=1e-10)
        @test size(state.mlik_arr) == (length(f.Z), 2^length(selected))
        @test all(sum(state.mlik_arr; dims=2) .≈ 1)
        @test state.mlik_arr !== previous.mlik_arr
        for field in fieldnames(typeof(previous))
            @test getfield(previous, field) == getfield(snapshot, field)
        end
    end
    for field in fieldnames(typeof(ec))
        @test getfield(ec, field) == getfield(original, field)
    end

    for positions in ([3, 1], [4, 2, 1], [2], Int[])
        @test isapprox(BASSGWAS.comp_eig(ec, positions), reference_eig(p, f.Z, positions);
                       rtol=1e-8, atol=1e-10)
    end
    @test BASSGWAS.comp_eig(ec, [3, 1]) ≈ BASSGWAS.comp_eig(ec, [1, 3])
    @test BASSGWAS.comp_eig(ec, Int32[4, 2]) ≈ reference_eig(p, f.Z, [4, 2])
    @test BASSGWAS.comp_eig(update_eig(ec, 2), [4, 1]) ≈ reference_eig(p, f.Z, [2, 4, 1])
end

@testset "Greedy design with original target identifiers" begin
    f = eig_fixture()
    ec = make_eig(f)
    original = deepcopy(ec)
    p = reference_probabilities(f)
    for fixed in (Int[], [f.idx_rem[3], f.idx_rem[1]]), batch_sz in (0, 2, 4)
        result = optimize_design(ec, f.idx_rem, batch_sz; idx_fixed=fixed)
        selected = Int[]
        for i in 1:batch_sz
            remaining = setdiff(1:4, selected)
            next = i <= length(fixed) ? findfirst(==(fixed[i]), f.idx_rem) :
                   remaining[argmax([reference_eig(p, f.Z, [selected; j]) for j in remaining])]
            push!(selected, next)
        end
        @test result.batch == f.idx_rem[selected]
        @test isapprox(result.obj, reference_eig(p, f.Z, selected); rtol=1e-8, atol=1e-10)
        @test length(unique(result.batch)) == batch_sz
    end
    for fixed in (Int[], [f.idx_rem[3]])
        Random.seed!(412)
        remaining = collect(1:4)
        selected = Int[]
        for i in 1:4
            pos = i <= length(fixed) ? findfirst(==(fixed[i]), f.idx_rem[remaining]) : rand(1:length(remaining))
            push!(selected, remaining[pos])
            deleteat!(remaining, pos)
        end
        Random.seed!(412)
        result = optimize_design(ec, f.idx_rem, 4; randdes=true, idx_fixed=fixed)
        @test result.batch == f.idx_rem[selected]
        @test result.obj ≈ reference_eig(p, f.Z, selected)
    end
    for field in fieldnames(typeof(ec))
        @test getfield(ec, field) == getfield(original, field)
    end
    @test f.idx_rem == [6, 2, 5, 3]
    tied = eig_fixture(; tied=true)
    @test optimize_design(make_eig(tied), tied.idx_rem, 4).batch == tied.idx_rem
end

# An order-sensitive objective with no CIDEIG fields verifies dispatch and the
# exact sequence of updates, including fixed-design positions supplied by users.
struct OrderedEIG <: EIGest
    weights::Vector{Float64}
    selected::Vector{Int}
end

function BASSGWAS.eig(ec::OrderedEIG, idx::Vector{T}) where T<:Integer
    total = foldl((value, i) -> 10value + ec.weights[i], ec.selected; init=0.0)
    [10total + ec.weights[i] for i in idx]
end

function BASSGWAS.update_eig(ec::OrderedEIG, idx::T) where T<:Integer
    OrderedEIG(ec.weights, [ec.selected; idx])
end

struct UnimplementedEIG <: EIGest end

@testset "Estimator interface dispatch" begin
    ec = OrderedEIG([1.0, 7.0, 4.0, 7.0], Int[])
    targets = [90, 20, 70, 10]
    @test optimize_design(ec, targets, 3) == (batch=[20, 10, 70], obj=774.0)
    @test optimize_design(ec, targets, 2; idx_fixed=[70]) == (batch=[70, 20], obj=47.0)
    @test optimize_design(ec, targets, 0) == (batch=Int[], obj=0.0)
    @test BASSGWAS.comp_eig(ec, [3, 1]) == 41.0
    @test BASSGWAS.comp_eig(ec, [1, 3]) == 14.0
    @test BASSGWAS.comp_eig(update_eig(ec, 2), Int32[3, 1]) == 741.0
    @test BASSGWAS.comp_eig(ec, Int[]) == 0.0
    @test isempty(ec.selected)
    @test_throws "Unimplemented EIG estimator" eig(UnimplementedEIG(), [1])
    @test_throws "Unimplemented EIG estimator" update_eig(UnimplementedEIG(), 1)
end
