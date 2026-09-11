# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using LinearAlgebra
using Random
using Distributions
using SpecialFunctions: loggamma

const NUM_RTOL = 1e-8
const NUM_ATOL = 1e-10

# Every selectable model has a stable, one-based table index: bitmask + 1.
inclusion_vector(mask, p) = Int64[(mask >> (i - 1)) & 1 for i in 1:p]
toggle_index(mask, i) = xor(mask, 1 << (i - 1)) + 1

function numerical_fixture(p; rank=3, design=:correlated, alpha=0.7, beta=2.3,
                           s=1.4)
    n = 8
    x = [-1.2, 0.4, 1.1, -0.7, 0.2, 1.5, -0.3, 0.8]
    z = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    X = hcat(x, 0.6 .* x .+ z, z, reverse(x))[:, 1:p]
    if design == :degenerate
        @assert p == 4
        X = hcat(x, x, zeros(n), z)
    end
    # Deterministic orthonormal DCT basis: truncated and full eigensystems.
    U = [j == 1 ? 1 / sqrt(n) : sqrt(2 / n) * cospi((i - 0.5) * (j - 1) / n)
         for i in 1:n, j in 1:rank]
    S = collect(range(0.4, 1.8; length=rank))
    Y = [0.7, -1.1, 0.3, 1.4, -0.6, 0.2, -0.9, 0.8]
    BASSGWAS.ParState2(X, Y, U, S, alpha, beta, 1.7, 0.8, s, 4.2)
end

# These references deliberately do not call any BASSGWAS likelihood, prior,
# rate, cache, or dense-reference helper. Each model is factored independently.
function dense_model_terms(ps, gamma)
    X = ps.Xaug[:, findall(==(1), gamma)]
    C = ps.s * Matrix{Float64}(I, ps.n, ps.n) +
        ps.r * ps.U * Diagonal(ps.S) * ps.U' +
        ps.c * X * X' + ps.mc * ones(ps.n, ps.n)
    F = cholesky(Symmetric(C))
    (det=2 * sum(log, diag(F.L)), ssq=dot(ps.Y, F \ ps.Y))
end

reference_loglik(::BASSGWAS.StdNormLogLik, det, ssq) = -(det + ssq) / 2
reference_loglik(ll::BASSGWAS.TDistLogLik, det, ssq) =
    -det / 2 - (ll.nu + ll.n) / 2 * log(ll.nu * ll.lambda + ssq)

function reference_logprior(k, p, alpha, beta)
    # Probability of a particular vector, not the probability of its model size.
    loggamma(k + alpha) + loggamma(p - k + beta) - loggamma(p + alpha + beta) -
        loggamma(alpha) - loggamma(beta) + loggamma(alpha + beta)
end

function enumerate_posterior(ps, loglik)
    gammas = [inclusion_vector(mask, ps.p) for mask in 0:(2^ps.p - 1)]
    terms = [dense_model_terms(ps, gamma) for gamma in gammas]
    dets = getproperty.(terms, :det)
    ssqs = getproperty.(terms, :ssq)
    lls = reference_loglik.(Ref(loglik), dets, ssqs)
    logpriors = [reference_logprior(sum(g), ps.p, ps.alpha, ps.beta) for g in gammas]
    logweights = lls + logpriors
    weights = exp.(logweights .- maximum(logweights))
    probabilities = weights / sum(weights)
    (; gammas, dets, ssqs, lls, logpriors, logweights, probabilities)
end

function reference_conditionals(exact, mask, p, eps)
    row = mask + 1
    neighbours = [toggle_index(mask, i) for i in 1:p]
    llrs = exact.logweights[neighbours] .- exact.logweights[row]
    pair_mass = exact.probabilities[row] .+ exact.probabilities[neighbours]
    conds = exact.probabilities[row] ./ pair_mass
    included = [mask | (1 << (i - 1)) for i in 1:p] .+ 1
    pips = exact.probabilities[included] ./ pair_mass
    rates = (pips .+ eps / p) ./ (2 .* conds)
    (; neighbours, llrs, conds, pips, rates)
end

function check_state(state, qc, ps, sc, loglik, exact, mask)
    ref = reference_conditionals(exact, mask, ps.p, state.eps)
    @test state.γ == exact.gammas[mask + 1]
    @test isapprox(state.ll_curr, exact.lls[mask + 1]; rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(BASSGWAS.comp_ll2(sc, qc, ps, state.γ, loglik), exact.lls[mask + 1];
                   rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(state.dets, exact.dets[ref.neighbours]; rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(state.ssq, exact.ssqs[ref.neighbours]; rtol=NUM_RTOL, atol=NUM_ATOL)
    for field in (:llrs, :conds, :pips)
        @test isapprox(getproperty(state, field), getproperty(ref, field);
                       rtol=NUM_RTOL, atol=NUM_ATOL)
    end
    expected_rates = [ref.rates; state.xi]
    @test isapprox(state.rates, expected_rates; rtol=NUM_RTOL, atol=NUM_ATOL)
    @test isapprox(state.rsums, cumsum(expected_rates); rtol=NUM_RTOL, atol=NUM_ATOL)
    @test all(isfinite, state.rates)
    @test all(>=(0), state.rates)
    @test all(x -> 0 <= x <= 1, state.pips)
    @test all(x -> 0 <= x <= 1, state.conds)
    @test issorted(state.rsums)
end

function check_qcache(qc, ps, gamma)
    active = [findall(==(1), gamma); ps.p + 1]
    inactive = findall(==(0), gamma)
    @test qc.i1_set == active
    @test qc.i0_set == inactive
    @test qc.d == length(active)
    @test qc.p == length(inactive)
    @test qc.X1 == ps.Xaug[:, active]
    @test qc.X1[:, end] == ones(ps.n)
    @test isapprox(qc.XtX, ps.Xaug[:, active]' * ps.Xaug; rtol=NUM_RTOL, atol=NUM_ATOL)
end
