# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using LinearAlgebra
using Random
using Distributions
using Statistics: mean, std
using SpecialFunctions: loggamma

const NUM_RTOL = 1.0e-8
const NUM_ATOL = 1.0e-10

# Every selectable model has a stable, one-based table index: bitmask + 1.
inclusion_vector(mask, p) = Int64[(mask >> (i - 1)) & 1 for i in 1:p]
toggle_index(mask, i) = xor(mask, 1 << (i - 1)) + 1

function numerical_fixture(
        p; rank = 3, design = :correlated, alpha = 0.7, beta = 2.3,
        s = 1.4
    )
    n = 8
    x = [-1.2, 0.4, 1.1, -0.7, 0.2, 1.5, -0.3, 0.8]
    z = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    X = hcat(x, 0.6 .* x .+ z, z, reverse(x))[:, 1:p]
    if design == :degenerate
        @assert p == 4
        X = hcat(x, x, zeros(n), z)
    end
    # Deterministic orthonormal DCT basis: truncated and full eigensystems.
    U = [
        j == 1 ? 1 / sqrt(n) : sqrt(2 / n) * cospi((i - 0.5) * (j - 1) / n)
            for i in 1:n, j in 1:rank
    ]
    S = rank == 1 ? [0.4] : collect(range(0.4, 1.8; length = rank))
    Y = [0.7, -1.1, 0.3, 1.4, -0.6, 0.2, -0.9, 0.8]
    return BASSGWAS.ParState2(X, Y, U, S, alpha, beta, 1.7, 0.8, s, 4.2)
end

# Batch influence scores account for both serial dependence and normalization
# of weighted-TGS estimates. Pool raw weights only across identical targets.
function check_full_sigma_invariance(make_sampler, shape, scale; min_models = 1)
    probabilities = [0.1, 0.25, 0.5, 0.75, 0.9]
    cutoffs = quantile.(InverseGamma(shape, scale), probabilities)
    seeds = (810, 811, 812)
    batches_per_chain, batch_length = 40, 500
    nbatches = length(seeds) * batches_per_chain
    weighted_counts = zeros(nbatches, length(probabilities))
    batch_weights = zeros(nbatches)
    models = Set{Tuple}()
    for (chain, seed) in enumerate(seeds)
        sampler = make_sampler(seed)
        for _ in 1:2000
            next_step!(sampler)
        end
        @test is_adapted(sampler)
        @test sampler.wTGSState.xi == 5.0
        valid_sigma, valid_weights = true, true
        for batch in 1:batches_per_chain
            row = (chain - 1) * batches_per_chain + batch
            for _ in 1:batch_length
                next_step!(sampler)
                draw = get_state(sampler)
                valid_sigma &= isfinite(draw.sigma) && draw.sigma > 0
                valid_weights &= isfinite(draw.Z) && draw.Z > 0
                push!(models, Tuple(draw.γ))
                batch_weights[row] += draw.Z
                for j in eachindex(cutoffs)
                    weighted_counts[row, j] += draw.Z * (draw.sigma <= cutoffs[j])
                end
            end
        end
        @test valid_sigma
        @test valid_weights
        @test sampler.wTGSState.xi == 5.0
    end
    @test length(models) >= min_models
    estimate = vec(sum(weighted_counts; dims = 1)) / sum(batch_weights)
    scores = (weighted_counts .- batch_weights .* estimate') ./ mean(batch_weights)
    se = vec(std(scores; dims = 1)) / sqrt(nbatches)
    for j in eachindex(probabilities)
        @testset "CDF probability=$(probabilities[j])" begin
            # This is an approximate MCMC error criterion, not an iid DKW bound.
            @test 6se[j] <= 0.03
            @test abs(estimate[j] - probabilities[j]) <= 6se[j]
        end
    end
end

# These references deliberately do not call any BASSGWAS likelihood, prior,
# rate, cache, or dense-reference helper. Each model is factored independently.
function dense_model_terms(ps, gamma)
    X = ps.Xaug[:, findall(==(1), gamma)]
    C = ps.s * Matrix{Float64}(I, ps.n, ps.n) +
        ps.r * ps.U * Diagonal(ps.S) * ps.U' +
        ps.c * X * X' + ps.mc * ones(ps.n, ps.n)
    F = cholesky(Symmetric(C))
    return (det = 2 * sum(log, diag(F.L)), ssq = dot(ps.Y, F \ ps.Y))
end

reference_loglik(::BASSGWAS.StdNormLogLik, det, ssq) = -(det + ssq) / 2
reference_loglik(ll::BASSGWAS.TDistLogLik, det, ssq) =
    -det / 2 - (ll.nu + ll.n) / 2 * log(ll.nu * ll.lambda + ssq)

function reference_logprior(k, p, alpha, beta)
    # Probability of a particular vector, not the probability of its model size.
    return loggamma(k + alpha) + loggamma(p - k + beta) - loggamma(p + alpha + beta) -
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
    return (; gammas, dets, ssqs, lls, logpriors, logweights, probabilities)
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
    return (; neighbours, llrs, conds, pips, rates)
end

function check_state(state, qc, ps, sc, loglik, exact, mask)
    ref = reference_conditionals(exact, mask, ps.p, state.eps)
    @test state.γ == exact.gammas[mask + 1]
    @test isapprox(state.ll_curr, exact.lls[mask + 1]; rtol = NUM_RTOL, atol = NUM_ATOL)
    @test isapprox(
        BASSGWAS.comp_ll2(sc, qc, ps, state.γ, loglik), exact.lls[mask + 1];
        rtol = NUM_RTOL, atol = NUM_ATOL
    )
    @test isapprox(state.dets, exact.dets[ref.neighbours]; rtol = NUM_RTOL, atol = NUM_ATOL)
    @test isapprox(state.ssq, exact.ssqs[ref.neighbours]; rtol = NUM_RTOL, atol = NUM_ATOL)
    for field in (:llrs, :conds, :pips)
        @test isapprox(
            getproperty(state, field), getproperty(ref, field);
            rtol = NUM_RTOL, atol = NUM_ATOL
        )
    end
    expected_rates = [ref.rates; state.xi]
    @test isapprox(exp.(state.logrates), expected_rates; rtol = NUM_RTOL, atol = NUM_ATOL)
    @test isapprox(state.rsums, cumsum(expected_rates / maximum(expected_rates)); rtol = NUM_RTOL, atol = NUM_ATOL)
    @test state.logtot ≈ log(sum(expected_rates))
    @test all(isfinite, state.logrates)
    @test all(>=(0), exp.(state.logrates))
    @test all(x -> 0 <= x <= 1, state.pips)
    @test all(x -> 0 <= x <= 1, state.conds)
    return @test issorted(state.rsums)
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
    return @test isapprox(qc.XtX, ps.Xaug[:, active]' * ps.Xaug; rtol = NUM_RTOL, atol = NUM_ATOL)
end

# Prior-predictive simulation followed by a posterior draw must recover the
# prior marginal. The conditional PIT additionally detects draws ignoring Y.
function check_sigma_calibration(draw_sigma!, C, shape, scale, seed)
    N = 20_000
    n = size(C, 1)
    F = cholesky(Symmetric(C))
    prior = InverseGamma(shape, scale)
    prior_rng, response_rng, posterior_rng = Xoshiro(seed), Xoshiro(seed + 1), Xoshiro(seed + 2)
    prior_samples, posterior_samples, conditional_pits = zeros(N), zeros(N), zeros(N)
    for i in 1:N
        Ys = F.L * randn(response_rng, n)
        s = BASSGWAS.scale_Ysigma!(Ys, shape, scale, prior_rng)
        q = dot(Ys, F \ Ys) # Independent dense reference, before mutation.
        sigma = draw_sigma!(Ys, shape, scale, q, posterior_rng)
        prior_samples[i] = s
        posterior_samples[i] = sigma
        conditional_pits[i] = cdf(InverseGamma(shape + n / 2, scale + q / 2), sigma)
    end
    @test all(s -> isfinite(s) && s > 0, prior_samples)
    @test all(s -> isfinite(s) && s > 0, posterior_samples)
    # DKW: P(D_N > bound) <= 1e-6/36. The union bound covers all 12
    # configurations and their three CDF checks, without assuming independence.
    bound = sqrt(log(2 * 36 / 1e-6) / (2N))
    for (label, pits) in (
            ("prior draws", cdf.(prior, prior_samples)),
            ("posterior marginal", cdf.(prior, posterior_samples)),
            ("conditional posterior", conditional_pits),
        )
        @testset "$label" begin
            @test all(u -> isfinite(u) && 0 <= u <= 1, pits)
            sorted = sort(pits)
            ks = maximum(max(i / N - u, u - (i - 1) / N) for (i, u) in enumerate(sorted))
            @test ks <= bound
        end
    end
end
