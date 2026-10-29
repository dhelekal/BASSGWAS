# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

@testset "Single-class phenotypes preserve liability signs" begin
    for n in (1, 4), outcome in (0, 1), p in (0, 1), sample_c in (false, true)
        @testset "n=$n outcome=$outcome p=$p sample_c=$sample_c" begin
            X = reshape([i / n - 0.5 for i in 1:n], n, 1)[:, 1:p]
            U = p == 0 ? zeros(n, 0) : fill(inv(sqrt(n)), n, 1)
            S = p == 0 ? Float64[] : [0.8]
            sampler = ProbitBVSSampler(X, fill(outcome, n), U, S, 1.0, 2.0;
                sample_c, nadapt = 0, xi_init = 100.0, rng = Xoshiro(418))
            # Cover startup and ordinary transitions, with all continuous moves enabled.
            for _ in 1:24
                next_step!(sampler)
                @test all(isfinite, sampler.parState.Y)
                @test all(y -> outcome == 1 ? y > 0 : y < 0, sampler.parState.Y)
                @test isfinite(sampler.wTGSState.logtot)
            end
            draw = get_state(sampler)
            @test all(isfinite, draw.Y)
            @test all(y -> outcome == 1 ? y > 0 : y < 0, draw.Y)
            @test all(isfinite, (draw.icept, draw.sigma, draw.r, draw.c, draw.Z))
        end
    end
end

function capture_update(f)
    return try
        f()
    catch err
        err
    end
end

@testset "Fully included model has valid conditional posteriors" begin
    for beta in (0.3, 1.0),
            ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8))
        @testset "beta=$beta likelihood=$(typeof(ll))" begin
            ps = numerical_fixture(3; beta)
            sc = BASSGWAS.makeSCache(ps)
            exact = enumerate_posterior(ps, ll)
            mask = 2^ps.p - 1
            ref = reference_conditionals(exact, mask, ps.p, 0.6)

            llrs = fill(NaN, ps.p)
            ratio_result = capture_update() do
                BASSGWAS.comp_llrs!(
                    llrs, exact.dets[ref.neighbours], exact.ssqs[ref.neighbours],
                    exact.lls[mask + 1], collect(1:ps.p), Int[], ps.p, ll, ps.alpha, ps.beta
                )
                llrs
            end
            @test !(ratio_result isa Exception)
            if !(ratio_result isa Exception)
                @test isapprox(llrs, ref.llrs; rtol = NUM_RTOL, atol = NUM_ATOL)
            end

            init_result = capture_update() do
                BASSGWAS.init_state2(ll, ps, sc, ones(Int64, ps.p); eps = 0.6, xi_init = 1.3)
            end
            @test !(init_result isa Exception)
            if !(init_result isa Exception)
                state, _ = init_result
                @test isapprox(state.llrs, ref.llrs; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(state.pips, ref.pips; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(state.conds, ref.conds; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(exp.(state.logrates), [ref.rates; 1.3]; rtol = NUM_RTOL, atol = NUM_ATOL)
            end

            # Also reproduce the failure when a valid non-full model is updated
            # after adding its last excluded predictor.
            state, qc = BASSGWAS.init_state2(ll, ps, sc, Int64[1, 1, 0]; eps = 0.6, xi_init = 1.3)
            BASSGWAS.flip_gamma!(state, ps, qc, 3)
            update_result = capture_update() do
                BASSGWAS.update_state2!(state, ps, sc, qc, ll)
                state
            end
            @test !(update_result isa Exception)
            if !(update_result isa Exception)
                @test isapprox(state.llrs, ref.llrs; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(state.pips, ref.pips; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(state.conds, ref.conds; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(exp.(state.logrates), [ref.rates; 1.3]; rtol = NUM_RTOL, atol = NUM_ATOL)
            end
        end
    end
end
