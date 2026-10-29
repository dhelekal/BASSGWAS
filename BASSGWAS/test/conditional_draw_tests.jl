# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

@testset "Full probit posterior under shift augmentation" begin
    # With only an intercept b ~ Normal(0, 1) and two outcomes of each class,
    # the marginal posterior is proportional to phi(b) Phi(b)^2 Phi(-b)^2.
    # Numerical integration is independent of the latent-variable sampler.
    grid = collect(-8.0:0.001:8.0)
    normal = Normal()
    density = pdf.(normal, grid) .* cdf.(normal, grid) .^ 2 .* cdf.(normal, -grid) .^ 2
    expected_second_moment = sum(density .* grid .^ 2) / sum(density)

    batch_means = Float64[]
    for seed in 1:3
        sampler = ProbitBVSSampler(
            zeros(4, 0), [0, 0, 1, 1], zeros(4, 0), Float64[], 1.0, 1.0;
            mcinit = 1.0, sample_c = false, nadapt = 0, adapt_xi = false, rng = Xoshiro(seed)
        )
        for _ in 1:2000
            next_step!(sampler)
        end
        for _ in 1:40
            total = 0.0
            for _ in 1:500
                next_step!(sampler)
                total += get_state(sampler).icept^2
            end
            push!(batch_means, total / 500)
        end
    end
    estimate = sum(batch_means) / length(batch_means)
    # Batch means account for serial dependence
    se = sqrt(
        sum((batch_means .- estimate) .^ 2) /
            (length(batch_means) * (length(batch_means) - 1))
    )
    @test 6se < 0.03
    @test abs(estimate - expected_second_moment) < 6se + 0.001
end

@testset "Gaussian coefficient conditional moments" begin
    for rank in (3, 8), mask in (0, 5, 15)
        @testset "rank=$rank model=$mask" begin
            # draw_betas and draw_Ysigma! conditionals use unit residual variance.
            ps = numerical_fixture(4; rank, s = 1.0)
            gamma = inclusion_vector(mask, ps.p)
            X = ps.Xaug[:, [findall(==(1), gamma); ps.p + 1]]
            K = [X ps.U * Diagonal(sqrt.(ps.S))]
            l, m = size(X, 2), length(ps.S)
            # Condition the joint Gaussian via observation-space covariance;
            # production instead builds coefficient-space posterior precision.
            D = Diagonal([fill(ps.c, l - 1); ps.mc; fill(ps.r, m)])
            C = cholesky(Symmetric(I + K * D * K'))
            posterior_mean = D * K' * (C \ ps.Y)
            posterior_cov = Matrix(D - D * K' * (C \ (K * D)))
            L = cholesky(Symmetric(posterior_cov)).L
            rng = Xoshiro(9100 + 100 * rank + mask)
            N = 20_000
            draws = Matrix{Float64}(undef, l + m, N)
            original_K, original_Y = copy(K), copy(ps.Y)
            for j in 1:N
                draws[:, j] = BASSGWAS.draw_betas(ps.Y, ps.r, ps.c, ps.mc, l, m, K, rng)
            end
            @test K == original_K
            @test ps.Y == original_Y
            @test all(isfinite, draws)
            # Whitening exposes errors in every covariance direction, including
            # correlations; tolerances are six analytic Monte Carlo SEs.
            standardized = L \ (draws .- posterior_mean)
            mean_z = vec(sum(standardized; dims = 2)) / N
            centered = standardized .- mean_z
            cov_z = centered * centered' / (N - 1)
            @test all(abs.(mean_z) .<= 6 / sqrt(N))
            for i in 1:(l + m), j in 1:i
                target = Float64(i == j)
                se = sqrt((i == j ? 2.0 : 1.0) / (N - 1))
                @test abs(cov_z[i, j] - target) <= 6 * se
            end
        end
    end
end

@testset "Inverse-gamma residual variance conditional" begin
    for rank in (3, 8), mask in 0:15, (shape, scale) in ((2.5, 0.8), (6.0, 1.7))
        @testset "rank=$rank model=$mask shape=$shape scale=$scale" begin
            ps = numerical_fixture(4; rank, s = 1.0)
            gamma = inclusion_vector(mask, ps.p)
            X = ps.Xaug[:, [findall(==(1), gamma); ps.p + 1]]
            dense = dense_model_terms(ps, gamma)
            conditional = InverseGamma(shape + ps.n / 2, scale + dense.ssq / 2)
            actual_rng = Xoshiro(1200 + mask + rank)
            expected_rng = copy(actual_rng)
            for _ in 1:3
                expected_sigma = rand(expected_rng, conditional)
                Y = copy(ps.Y)
                sigma = BASSGWAS.draw_Ysigma!(
                    Y, ps.r, ps.c, ps.mc, shape, scale,
                    X, X' * X, ps.U' * X, ps.U, ps.S, actual_rng
                )
                @test isapprox(sigma, expected_sigma; rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isapprox(Y, ps.Y / sqrt(expected_sigma); rtol = NUM_RTOL, atol = NUM_ATOL)
                @test isfinite(sigma) && sigma > 0
                @test sign.(Y) == sign.(ps.Y)
            end
            @test rand(actual_rng) == rand(expected_rng)
        end
    end
end
