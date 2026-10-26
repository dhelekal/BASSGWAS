# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using Random
using Statistics

# Independent high-precision inverse-CDF reference (no forward scan).
function reference_stratified_rows(rng, weights)
    cdf = cumsum(BigFloat.(weights))
    cdf ./= cdf[end]
    m = length(weights)
    uniforms = [(i - 1 + rand(rng)) / m for i in 1:m]
    rows = [findfirst(>(u), cdf) for u in uniforms]
    return shuffle!(rng, rows)
end

# Scripted stratum offsets exercise exact boundaries; shuffling uses a normal RNG.
mutable struct StratifiedBoundaryRNG <: AbstractRNG
    offsets::Vector{Float64}
    cursor::Int
    shuffle_rng::Xoshiro
end
StratifiedBoundaryRNG(offsets) = StratifiedBoundaryRNG(offsets, 1, Xoshiro(7))
function Random.rand(rng::StratifiedBoundaryRNG, ::Type{Float64})
    value = rng.offsets[rng.cursor]
    rng.cursor += 1
    return value
end
Random.shuffle!(rng::StratifiedBoundaryRNG, rows::AbstractVector) = shuffle!(rng.shuffle_rng, rows)

@testset "Stratified posterior resampling" begin
    for T in (Float32, Float64)
        cases = (
            ones(T, 32), T[1, 3, 2, 6],
            T[floatmin(T), 1, 2, floatmax(T)],
            floatmax(T) .* T[0.25, 0.5, 0.75, 1],
            floatmin(T) .* T[1, 2, 3, 4],
        )
        for weights in cases, seed in 1:20
            before = copy(weights)
            actual = BASSGWAS.mlmc_stratified_rows(Xoshiro(seed), weights)
            expected = reference_stratified_rows(Xoshiro(seed), weights)
            @test actual == expected
            @test actual isa Vector{Int}
            @test length(actual) == length(weights)
            @test all(1 <= row <= length(weights) for row in actual)
            @test weights == before
        end
        weights = T[1, 3, 2, 6]
        @test BASSGWAS.mlmc_stratified_rows(Xoshiro(9), weights) ==
            BASSGWAS.mlmc_stratified_rows(Xoshiro(9), 8 .* weights)
        uniform = BASSGWAS.mlmc_stratified_rows(Xoshiro(9), ones(T, 128))
        @test sort(uniform) == collect(1:128)
        @test !issorted(uniform)
    end
    for offsets in (zeros(4), fill(prevfloat(1.0), 4), [0.1, 0.9, 0.2, 0.8])
        rng = StratifiedBoundaryRNG(offsets)
        rows = BASSGWAS.mlmc_stratified_rows(rng, ones(4))
        @test sort(rows) == collect(1:4)
        @test rng.cursor == 5
    end
    # u=0.125 lies exactly on the first CDF boundary and must select row two.
    rows = BASSGWAS.mlmc_stratified_rows(StratifiedBoundaryRNG(fill(0.5, 4)), [1.0, 3, 2, 2])
    @test sort(rows) == [2, 2, 3, 4]
    # Underflowed leading mass is skipped even when the first uniform is zero.
    rows = BASSGWAS.mlmc_stratified_rows(
        StratifiedBoundaryRNG(zeros(4)),
        [nextfloat(0.0), floatmax(Float64), 1.0, 1.0]
    )
    @test rows == fill(2, 4)

    weights = [1.0, 3, 2, 6]
    counts = zeros(Int, 4, 2000)
    for seed in axes(counts, 2)
        for row in BASSGWAS.mlmc_stratified_rows(Xoshiro(seed), weights)
            counts[row, seed] += 1
        end
    end
    expected = 4 .* weights ./ sum(weights)
    for row in 1:4
        @test abs(mean(counts[row, :]) - expected[row]) <
            6std(counts[row, :]) / sqrt(size(counts, 2)) + 0.01
    end
end

@testset "Stratified constructor pool ordering and outcomes" begin
    for T in (Float32, Float64)
        m = 128
        args = (
            [1, 2], zeros(T, 2, 1), ones(T, 1), zeros(T, 2, 1),
            [fill(T(-1.2), 64); fill(T(1.2), 64)], zeros(T, m, 1), zeros(T, m, 1),
        )
        weights = [ones(T, 64); fill(T(3), 64)]
        rng = Xoshiro(19)
        expected_rows = reference_stratified_rows(rng, weights)
        lp1, lp0 = postpred(args[2:end]...)
        selected_lp1 = lp1[expected_rows, :]
        expected_outcomes = Matrix{Int}(undef, m, 2)
        for i in eachindex(expected_outcomes)
            expected_outcomes[i] = rand(rng, eltype(lp1)) < exp(selected_lp1[i]) ? 1 : 0
        end
        actual_rng = Xoshiro(19)
        ec = MLMCEig(args..., weights; rng = actual_rng)
        @test ec.posterior_rows == expected_rows
        @test ec.lp_success == lp1[expected_rows, :]
        @test ec.lp_fail == lp0[expected_rows, :]
        @test ec.outcomes == expected_outcomes
        @test count(<=(64), ec.posterior_rows) == 32
        @test !issorted(ec.posterior_rows)
        # Ordered source groups are dispersed across both halves of the pool.
        @test 0 < count(<=(64), ec.posterior_rows[1:64]) < 32
        @test !hasproperty(ec, :draw_order)
        @test rand(rng) == rand(actual_rng)
    end
end
