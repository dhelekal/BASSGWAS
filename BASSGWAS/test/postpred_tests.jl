# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

using Test
using BASSGWAS
using LinearAlgebra
using Random
using Statistics

function check_expansion(Um, Sm, U, S, observed, coefficients; seed = 37)
    ndraws = 20_000
    us = repeat(coefficients', ndraws, 1)
    rs = repeat([0.5, 2.0], ndraws ÷ 2)
    inputs = (Um, Sm, U, S, observed, us, rs)
    saved = deepcopy(inputs)
    expanded = expand_randeff(inputs..., Xoshiro(seed))
    @test size(expanded) == (ndraws, length(S))
    @test all(isfinite, expanded)
    @test inputs == saved

    # Independent dense conditional-Gaussian reference in observation space.
    V = U * Diagonal(sqrt.(S))
    K = V * V'
    y = Um * Diagonal(sqrt.(Sm)) * coefficients
    cross = K[:, observed]
    inverse = pinv(K[observed, observed])
    expected_mean = cross * inverse * y
    expected_covariance = Symmetric(K - cross * inverse * cross')
    effects = expanded * V'
    @test all(isapprox.(effects[:, observed], y'; atol = 1e-6))
    # Remove each draw's variance scaling before checking conditional moments.
    residual = (effects .- expected_mean') ./ sqrt.(rs)
    empirical_mean = vec(mean(residual; dims = 1))
    empirical_covariance = cov(residual; dims = 1)
    for i in axes(K, 1)
        @test abs(empirical_mean[i]) <= 7sqrt(max(0.0, expected_covariance[i,i]) / ndraws) + 1e-6
        for j in axes(K, 2)
            variance = max(0.0, expected_covariance[i,i] * expected_covariance[j,j] + expected_covariance[i,j]^2)
            @test abs(empirical_covariance[i,j] - expected_covariance[i,j]) <= 7sqrt(variance / (ndraws-1)) + 1e-6
        end
    end
    return expanded
end

@testset "Expansion supports singular observed covariances" begin
    I3 = Matrix{Float64}(I, 3, 3)
    # Reproduce zero-mode failure while retaining an uncertain unobserved effect.
    expanded = check_expansion(Matrix{Float64}(I, 2, 2), [2.0, 0.0], I3,
        [2.0, 0.0, 3.0], [1, 2], [0.7, 123.0])
    @test all(isapprox.(expanded[:, 1], 0.7; atol = 1e-6))
    # Full-basis null coefficients still have their unconstrained prior distribution.
    @test abs(var(expanded[:, 2] ./ sqrt.(repeat([0.5, 2.0], 10_000))) - 1) < 0.07
    check_expansion(ones(1,1), [0.0], I3, [0.0, 1.0, 2.0], [1], [123.0])
    check_expansion(Matrix{Float64}(I, 2, 2), zeros(2), Matrix{Float64}(I, 2, 2),
        zeros(2), [1, 2], [123.0, -45.0])
    check_expansion(zeros(1,0), Float64[], I3, [0.0, 1.0, 2.0], [1], Float64[])
    empty = expand_randeff(zeros(1,0), Float64[], zeros(3,0), Float64[],
        [1], zeros(4,0), ones(4), Xoshiro(37))
    @test size(empty) == (4,0)

    # A rotated singular covariance exercises non-coordinate null directions.
    U = [j == 1 ? 0.5 : sqrt(0.5)*cospi((i-0.5)*(j-1)/4) for i in 1:4, j in 1:4]
    S, observed = [2.0, 0.0, 1.0, 0.0], [1, 3, 4]
    eig = eigen(Symmetric((U * Diagonal(S) * U')[observed, observed]))
    Sm = copy(eig.values)
    Sm[1] = 0.0
    check_expansion(eig.vectors, Sm, U, S, observed, [123.0, 0.7, -0.3])
end

@testset "Positive modes and invalid eigenvalues" begin
    U = Matrix{Float64}(I, 2, 2)
    for s in (2.0, 1e-10)
        expanded = check_expansion(ones(1,1), [s], U, [s, 1.0], [1], [0.7])
        @test all(isapprox.(expanded[:, 1], 0.7; atol = 1e-6))
    end
    for bad in (-1.0, NaN, Inf, -Inf)
        @test_throws ArgumentError expand_randeff(ones(1,1), [bad], U, [1.0, 1.0],
            [1], ones(2,1), ones(2), Xoshiro(1))
        @test_throws ArgumentError expand_randeff(ones(1,1), [1.0], U, [1.0, bad],
            [1], ones(2,1), ones(2), Xoshiro(1))
    end
end
