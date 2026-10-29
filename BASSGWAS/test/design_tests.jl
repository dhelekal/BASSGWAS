# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using Distributions

@testset "Experimental design validates constraints and honors fixed targets" begin
    U, S, X = zeros(6, 0), Float64[], zeros(6, 0)
    icept, betas, us, Z = [-1.0, 0.0, 1.0, 2.0], zeros(4, 0), zeros(4, 0), fill(0.25, 4)
    candidates = [6, 2, 4, 1]
    for randdes in (false, true)
        design(ids, k; fixed = Int[]) = optimize_design(ids, U, S, X, icept, betas, us, Z, k;
            randdes, idx_fixed = fixed)
        for badsize in (-1, 1.5, 5)
            @test_throws ArgumentError design(candidates, badsize)
        end
        for badfixed in ([4, 6], [4, 4], [3], [0], [7], [1.5])
            @test_throws ArgumentError design(candidates, 1; fixed = badfixed)
        end
        # Duplicate fixed targets must also be rejected when they fit in the batch.
        @test_throws ArgumentError design(candidates, 2; fixed = [4, 4])
        for badids in ([1, 1], [0, 2], [7, 1], [1.5, 2.0])
            @test_throws ArgumentError design(badids, 1)
        end
        @test_throws ArgumentError design(candidates, 0; fixed = [4])
        @test_throws ArgumentError design(Int[], 1)
        @test design(candidates, 0) == (batch = Int[], obj = 0.0)
        @test design(Int[], 0) == (batch = Int[], obj = 0.0)

        fixed = [4, 6]
        result = design(candidates, 3; fixed)
        @test result.batch[1:2] == fixed
        @test length(result.batch) == 3
        @test allunique(result.batch)
        @test all(in(candidates), result.batch)
        @test candidates == [6, 2, 4, 1]
        @test fixed == [4, 6]
        # Independently enumerate the joint binary outcomes of the chosen batch.
        q = cdf.(Normal(), icept)
        joint = [sum(Z[j] * prod(((mask >> i) & 1) == 1 ? q[j] : 1-q[j] for i in 0:2)
            for j in eachindex(Z)) for mask in 0:7]
        conditional_entropy = -3sum(Z .* (q .* log.(q) .+ (1 .- q) .* log.(1 .- q)))
        reference = -sum(joint .* log.(joint)) - conditional_entropy
        @test result.obj ≈ reference atol=1e-12

        allfixed = [4, 6, 1, 2]
        @test design(candidates, 4; fixed = allfixed).batch == allfixed
        @test sort(design(candidates, 4).batch) == sort(candidates)
    end
end
