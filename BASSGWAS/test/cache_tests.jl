# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

function check_scache(sc, ps)
    C = ps.s * Matrix{Float64}(I, ps.n, ps.n) + ps.r * ps.U * Diagonal(ps.S) * ps.U'
    F = cholesky(Symmetric(C))
    precision = F \ Matrix{Float64}(I, ps.n, ps.n)
    @test isapprox(sc.covariance.gramdiag, diag(ps.Xaug' * precision * ps.Xaug); rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.FX, ps.Y' * precision * ps.Xaug; rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.qterm, dot(ps.Y, F \ ps.Y); rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.cdet, 2 * sum(log, diag(F.L)); rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.QX' * sc.QX, ps.Xaug' * (I / ps.s - precision) * ps.Xaug;
                   rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.YtX, ps.Y' * ps.Xaug; rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(sc.YtU, ps.Y' * ps.U; rtol=NUM_RTOL, atol=NUM_ATOL)
end

@testset "Cache updates along all model toggles" begin
    for design in (:correlated, :degenerate), rank in (3, 8),
        ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8))
        @testset "design=$design rank=$rank likelihood=$(typeof(ll))" begin
            ps = numerical_fixture(4; rank, design)
            sc = BASSGWAS.makeSCache(ps)
            state, qc = BASSGWAS.init_state2(ll, ps, sc, zeros(Int64, 4); eps=0.6, xi_init=0.9)
            # Consecutive Gray-code masks, including the final return to zero,
            # differ at exactly one bit. Every model is visited in each pass.
            path = [xor(i, i >> 1) for i in 0:15]
            push!(path, 0)
            for pass in 1:3
                @testset "refresh=$pass" begin
                    if pass == 2
                        ps.Y = reverse(ps.Y) .+ 0.25
                    elseif pass == 3
                        ps.Y .*= 0.7
                        ps.r = 1.6
                        ps.s = 0.65
                        ps.c = 0.45
                        ps.mc = 2.8
                        ps.alpha = 1.3
                        ps.beta = 3.1
                        state.eps = 2.2
                        state.xi = 1.7
                    end
                    BASSGWAS.updateSCache!(sc, ps)
                    check_scache(sc, ps)
                    exact = enumerate_posterior(ps, ll)
                    previous = 0
                    for mask in path
                        @testset "model=$mask" begin
                            if mask != previous
                                changed = xor(previous, mask)
                                @test count_ones(changed) == 1
                                BASSGWAS.flip_gamma!(state, ps, qc, trailing_zeros(changed) + 1)
                            end
                            BASSGWAS.update_state2!(state, ps, sc, qc, ll)
                            check_state(state, qc, ps, sc, ll, exact, mask)
                            check_qcache(qc, ps, exact.gammas[mask + 1])
                            fresh_sc = BASSGWAS.makeSCache(ps)
                            fresh, fresh_qc = BASSGWAS.init_state2(ll, ps, fresh_sc,
                                exact.gammas[mask + 1]; eps=state.eps, xi_init=state.xi)
                            for field in (:dets, :ssq, :llrs, :conds, :pips, :rates, :rsums, :ll_curr)
                                @test isapprox(getproperty(state, field), getproperty(fresh, field);
                                               rtol=NUM_RTOL, atol=NUM_ATOL)
                            end
                            @test isapprox(qc.XtX, fresh_qc.XtX; rtol=NUM_RTOL, atol=NUM_ATOL)
                            previous = mask
                        end
                    end
                    @test state.γ == zeros(Int64, 4)
                    @test qc.i1_set == [5]
                end
            end
        end
    end
end
