# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

@testset "Rates from synthetic posterior odds" begin
    for p in 1:4, mask in 0:(2^p - 1), eps in (0.0, 0.6, 5.0),
        offset in (-20.0, 0.0, 20.0)
        gamma = inclusion_vector(mask, p)
        llrs = ([-2.5, 0.0, 1.7, 0.0] .+ offset)[1:p]
        # Normalize two model masses directly, without the production logistic.
        pairs = [(exp(-max(0.0, x)), exp(x - max(0.0, x))) for x in llrs]
        conds_ref = [a / (a + b) for (a, b) in pairs]
        pips_ref = [gamma[i] == 1 ? a / (a + b) : b / (a + b)
                    for (i, (a, b)) in enumerate(pairs)]
        expected = (pips_ref .+ eps / p) ./ (2 .* conds_ref)
        storage = fill(-99.0, p + 2)
        rates = view(storage, 2:(p + 1))
        conds, pips = fill(NaN, p), fill(NaN, p)
        BASSGWAS.comp_rates2!(rates, conds, pips, llrs,
                             findall(==(1), gamma), findall(==(0), gamma), eps)
        @test isapprox(conds, conds_ref; rtol=NUM_RTOL, atol=NUM_ATOL)
        @test isapprox(pips, pips_ref; rtol=NUM_RTOL, atol=NUM_ATOL)
        @test isapprox(rates, expected; rtol=NUM_RTOL, atol=NUM_ATOL)
        @test storage[[1, end]] == [-99.0, -99.0]
    end
end

@testset "Exact weighted-TGS transition matrix" begin
    for p in 1:4, eps in (0.0, 0.6, 5.0), xi in (0.4, 2.1),
        ll in (BASSGWAS.StdNormLogLik{Float64}(), BASSGWAS.TDistLogLik(3.5, 1.2, 8))
        @testset "p=$p eps=$eps xi=$xi likelihood=$(typeof(ll))" begin
            ps = numerical_fixture(p)
            sc = BASSGWAS.makeSCache(ps)
            exact = enumerate_posterior(ps, ll)
            nm = 2^p
            transition = zeros(nm, nm)
            totals = zeros(nm)
            pips = zeros(nm, p)
            for mask in 0:(nm - 1)
                state, _ = BASSGWAS.init_state2(ll, ps, sc, exact.gammas[mask + 1]; eps, xi_init=xi)
                ref = reference_conditionals(exact, mask, p, eps)
                @test isapprox(state.rates, [ref.rates; xi]; rtol=NUM_RTOL, atol=NUM_ATOL)
                row = mask + 1
                totals[row] = sum(state.rates)
                pips[row, :] = state.pips
                for i in 1:p
                    transition[row, toggle_index(mask, i)] = state.rates[i] / totals[row]
                end
                # Freeze continuous parameters: the auxiliary event is a self-loop.
                transition[row, row] = state.rates[end] / totals[row]
            end
            @test all(>=(0), transition)
            @test isapprox(vec(sum(transition; dims=2)), ones(nm); atol=NUM_ATOL)
            stationary = exact.probabilities .* totals
            stationary ./= sum(stationary)
            flow = Diagonal(stationary) * transition
            @test isapprox(flow, flow'; rtol=NUM_RTOL, atol=NUM_ATOL)
            @test isapprox(transition' * stationary, stationary; rtol=NUM_RTOL, atol=NUM_ATOL)
            # Use the actual transition matrix to recover its invariant law too.
            system = Matrix(transition' - I)
            system[end, :] .= 1.0
            rhs = zeros(nm)
            rhs[end] = 1.0
            solved = system \ rhs
            @test isapprox(solved, stationary; rtol=NUM_RTOL, atol=NUM_ATOL)
            posterior = solved ./ totals
            posterior ./= sum(posterior)
            @test isapprox(posterior, exact.probabilities; rtol=NUM_RTOL, atol=NUM_ATOL)
            inclusion = Float64.(reduce(hcat, exact.gammas)')
            marginal = inclusion' * exact.probabilities
            @test isapprox(inclusion' * posterior, marginal; rtol=NUM_RTOL, atol=NUM_ATOL)
            @test isapprox(pips' * posterior, marginal; rtol=NUM_RTOL, atol=NUM_ATOL)
        end
    end
end

# Control the uniform draw to test categorical intervals without sampling noise.
struct FixedUniform <: AbstractRNG
    value::Float64
end
Random.rand(rng::FixedUniform) = rng.value

@testset "Cumulative-rate move selection" begin
    ps = numerical_fixture(4)
    ll = BASSGWAS.StdNormLogLik{Float64}()
    state, _ = BASSGWAS.init_state2(ll, ps, BASSGWAS.makeSCache(ps), Int64[1, 0, 1, 0])
    lower = 0.0
    for i in eachindex(state.rates)
        upper = state.rsums[i]
        u = (lower + upper) / (2 * state.rsums[end])
        @test BASSGWAS.sample_next2(state, FixedUniform(u)) == i
        lower = upper
    end
    @test BASSGWAS.sample_next2(state, FixedUniform(0.0)) == 1
    @test BASSGWAS.sample_next2(state, FixedUniform(prevfloat(1.0))) == ps.p + 1
end
