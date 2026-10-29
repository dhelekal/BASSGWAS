# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

@testset "Likelihood evaluators" begin
    dets = [-1.3, 0.0, 0.8, 4.1]
    ssqs = [0.0, 0.3, 2.7, 12.0]
    for ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8)),
            offset in (-2.4, 0.0, 1.7)
        expected = reference_loglik.(Ref(ll), dets, ssqs)
        @test isapprox(ll.(dets, ssqs), expected; rtol = NUM_RTOL, atol = NUM_ATOL)
        # A view also checks that in-place evaluators respect their output bounds.
        buffer = fill(-99.0, 6)
        BASSGWAS.ll_plus_a!(view(buffer, 2:5), dets, ssqs, offset, ll)
        @test isapprox(buffer[2:5], expected .+ offset; rtol = NUM_RTOL, atol = NUM_ATOL)
        @test buffer[[1, 6]] == [-99.0, -99.0]
    end
end

@testset "Integrated inclusion prior" begin
    for p in 1:4, (alpha, beta) in ((0.7, 2.3), (2.1, 0.3), (1.0, 1.0))
        priors = [
            reference_logprior(sum(inclusion_vector(mask, p)), p, alpha, beta)
                for mask in 0:(2^p - 1)
        ]
        @test isapprox(sum(exp, priors), 1.0; rtol = NUM_RTOL, atol = NUM_ATOL)
        for k in 0:p
            @test isapprox(
                BASSGWAS._inc_prior(k, p, alpha, beta),
                reference_logprior(k, p, alpha, beta);
                rtol = NUM_RTOL, atol = NUM_ATOL
            )
        end
    end
end

@testset "Enumerated conditional posteriors" begin
    for p in 1:4, rank in (3, 8),
            ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8))
        @testset "p=$p rank=$rank likelihood=$(typeof(ll))" begin
            ps = numerical_fixture(p; rank)
            sc = BASSGWAS.makeSCache(ps)
            exact = enumerate_posterior(ps, ll)
            @test isapprox(sum(exact.probabilities), 1.0; atol = NUM_ATOL)
            @test all(>(0), exact.probabilities)
            for mask in 0:(2^p - 1)
                @testset "model=$mask" begin
                    gamma = exact.gammas[mask + 1]
                    state, qc = BASSGWAS.init_state2(ll, ps, sc, gamma; eps = 0.6, xi_init = 1.3)
                    check_state(state, qc, ps, sc, ll, exact, mask)
                    check_qcache(qc, ps, gamma)
                    ref = reference_conditionals(exact, mask, p, state.eps)
                    llrs = fill(NaN, p)
                    # Feed independent dense inputs directly into the ratio routine.
                    BASSGWAS.comp_llrs!(
                        llrs, exact.dets[ref.neighbours],
                        exact.ssqs[ref.neighbours], exact.lls[mask + 1],
                        findall(==(1), gamma), findall(==(0), gamma), p, ll, ps.alpha, ps.beta
                    )
                    @test isapprox(llrs, ref.llrs; rtol = NUM_RTOL, atol = NUM_ATOL)
                    state.xi = 2.1
                    state.eps = 0.0
                    BASSGWAS.update_state2!(state, ps, sc, qc, ll)
                    check_state(state, qc, ps, sc, ll, exact, mask)
                end
            end
        end
    end
end

# Sub-unit and unit beta are valid for every model. Their currently defective
# full-model updates are isolated in boundary_regression_tests.jl.
@testset "Sparse-prior non-full models" begin
    for beta in (0.3, 1.0), ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8))
        ps = numerical_fixture(4; alpha = 2.1, beta)
        sc = BASSGWAS.makeSCache(ps)
        exact = enumerate_posterior(ps, ll)
        for mask in 0:14
            @testset "beta=$beta likelihood=$(typeof(ll)) model=$mask" begin
                state, qc = BASSGWAS.init_state2(ll, ps, sc, exact.gammas[mask + 1])
                check_state(state, qc, ps, sc, ll, exact, mask)
            end
        end
    end
end
