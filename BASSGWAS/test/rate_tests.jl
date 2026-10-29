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
        pips_ref = [
            gamma[i] == 1 ? a / (a + b) : b / (a + b)
                for (i, (a, b)) in enumerate(pairs)
        ]
        expected = (pips_ref .+ eps / p) ./ (2 .* conds_ref)
        storage = fill(-99.0, p + 2)
        rates = view(storage, 2:(p + 1))
        conds, pips = fill(NaN, p), fill(NaN, p)
        BASSGWAS.comp_rates2!(
            rates, conds, pips, llrs,
            findall(==(1), gamma), findall(==(0), gamma), eps
        )
        @test isapprox(conds, conds_ref; rtol = NUM_RTOL, atol = NUM_ATOL)
        @test isapprox(pips, pips_ref; rtol = NUM_RTOL, atol = NUM_ATOL)
        @test isapprox(rates, log.(expected); rtol = NUM_RTOL, atol = NUM_ATOL)
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
                state, _ = BASSGWAS.init_state2(ll, ps, sc, exact.gammas[mask + 1]; eps, xi_init = xi)
                ref = reference_conditionals(exact, mask, p, eps)
                @test isapprox(exp.(state.logrates), [ref.rates; xi]; rtol = NUM_RTOL, atol = NUM_ATOL)
                row = mask + 1
                totals[row] = exp(state.logtot)
                pips[row, :] = state.pips
                for i in 1:p
                    transition[row, toggle_index(mask, i)] = (state.rsums[i] - (i == 1 ? 0.0 : state.rsums[i - 1])) / state.rsums[end]
                end
                # Freeze continuous parameters: the auxiliary event is a self-loop.
                transition[row, row] = (state.rsums[end] - state.rsums[end - 1]) / state.rsums[end]
            end
            @test all(>=(0), transition)
            @test isapprox(vec(sum(transition; dims = 2)), ones(nm); atol = NUM_ATOL)
            stationary = exact.probabilities .* totals
            stationary ./= sum(stationary)
            flow = Diagonal(stationary) * transition
            @test isapprox(flow, flow'; rtol = NUM_RTOL, atol = NUM_ATOL)
            @test isapprox(transition' * stationary, stationary; rtol = NUM_RTOL, atol = NUM_ATOL)
            # Use the actual transition matrix to recover its invariant law too.
            system = Matrix(transition' - I)
            system[end, :] .= 1.0
            rhs = zeros(nm)
            rhs[end] = 1.0
            solved = system \ rhs
            @test isapprox(solved, stationary; rtol = NUM_RTOL, atol = NUM_ATOL)
            posterior = solved ./ totals
            posterior ./= sum(posterior)
            @test isapprox(posterior, exact.probabilities; rtol = NUM_RTOL, atol = NUM_ATOL)
            inclusion = Float64.(reduce(hcat, exact.gammas)')
            marginal = inclusion' * exact.probabilities
            @test isapprox(inclusion' * posterior, marginal; rtol = NUM_RTOL, atol = NUM_ATOL)
            @test isapprox(pips' * posterior, marginal; rtol = NUM_RTOL, atol = NUM_ATOL)
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
    for i in eachindex(exp.(state.logrates))
        upper = state.rsums[i]
        u = (lower + upper) / (2 * state.rsums[end])
        @test BASSGWAS.sample_next2(state, FixedUniform(u)) == i
        lower = upper
    end
    @test BASSGWAS.sample_next2(state, FixedUniform(0.0)) == 1
    @test BASSGWAS.sample_next2(state, FixedUniform(prevfloat(1.0))) == ps.p + 1
end

# Independent probability definitions evaluated before Float64 can overflow.
function big_rates(odds, gamma, eps, xi)
    p = length(odds)
    c = [inv(1 + exp(BigFloat(x))) for x in odds]
    q = [gamma[i] == 1 ? c[i] : inv(1 + exp(-BigFloat(odds[i]))) for i in 1:p]
    return [[(q[i] + BigFloat(eps) / p) / (2c[i]) for i in 1:p]; BigFloat(xi)]
end

function synthetic_state(odds, gamma; eps = 5.0, xi = 2.0)
    p = length(odds)
    logs, conds, pips = zeros(p + 1), zeros(p), zeros(p)
    BASSGWAS.comp_rates2!(
        view(logs, 1:p), conds, pips, odds,
        findall(==(1), gamma), findall(==(0), gamma), eps
    )
    logs[end] = log(xi)
    m = maximum(logs)
    cumulative = cumsum(exp.(logs .- m))
    return BASSGWAS.wTGSState2(
        p, gamma, zeros(p), zeros(p), copy(odds),
        pips, conds, logs, cumulative, 0.0, xi, eps
    )
end

@testset "Extreme log rates and scaled cumulative intensities" begin
    for odds in ([800.0, 900.0], [-800.0, -900.0], [709.0, 709.0], [-900.0, 900.0], [0.0, 0.0]),
            gamma in ([0, 0], [1, 0], [1, 1]), eps in (0.0, 0.2, 2.0, 20.0, 1.0e-5), xi in (0.0, 2.0)
        state = synthetic_state(odds, gamma; eps, xi)
        reference = big_rates(odds, gamma, eps, xi)
        @test isapprox(state.logrates, log.(reference); atol = 2.0e-12)
        @test isapprox(state.logtot, log(sum(reference)); atol = 2.0e-12)
        @test isapprox(state.rsums, cumsum(reference / maximum(reference)); atol = 2.0e-12)
        @test isapprox(diff([0.0;state.rsums]) / state.rsums[end], reference / sum(reference); atol = 2.0e-12)
        @test issorted(state.rsums)
        @test 1 <= state.rsums[end] <= 3 + 1.0e-12
        refcdf = cumsum(reference / sum(reference))
        for i in eachindex(reference)
            lo = i == 1 ? big"0" : refcdf[i - 1]
            hi = refcdf[i]
            if hi - lo > 1.0e-10
                @test BASSGWAS.sample_next2(state, FixedUniform(Float64((hi + lo) / 2))) == i
            end
        end
        if xi == 0
            @test state.rsums[end] == state.rsums[end - 1]
            @test BASSGWAS.sample_next2(state, FixedUniform(prevfloat(1.0))) != 3
        end
        args = [getfield(state, name) for name in fieldnames(typeof(state))]
        @test BASSGWAS.wTGSState2{Float64}(args...).logtot == state.logtot
    end
    @test BASSGWAS.sample_next2((rsums = [0.0, 2.0, 2.0],), FixedUniform(0.0)) == 2
    @test BASSGWAS.sample_next2((rsums = [1.0, 1.0, 2.0],), FixedUniform(0.5)) == 3
    ps2 = numerical_fixture(2)
    ll2 = BASSGWAS.StdNormLogLik{Float64}()
    noaux, qc = BASSGWAS.init_state2(ll2, ps2, BASSGWAS.makeSCache(ps2), [0, 0]; xi_init = 0.0)
    @test noaux.logrates[end] == -Inf
    @test noaux.rsums[end] == noaux.rsums[end - 1]
    @test isfinite(noaux.logtot)
    ps = numerical_fixture(0)
    ll = BASSGWAS.StdNormLogLik{Float64}()
    for xi in (0.25, 3.0)
        state, _ = BASSGWAS.init_state2(ll, ps, BASSGWAS.makeSCache(ps), Int[]; xi_init = xi)
        @test state.logrates == [log(xi)]
        @test state.rsums == [1.0]
        @test state.logtot ≈ log(xi)
    end
    @test_throws DomainError BASSGWAS.init_state2(ll, ps, BASSGWAS.makeSCache(ps), Int[]; xi_init = 0.0)
end

@testset "Concurrent rate kernels and bounded allocations" begin
    for p in (1, 2, 3, 257, 1003)
        odds = [900sin(i) for i in 1:p]
        gamma = [Int(i % 5 == 0) for i in 1:p]
        expected = synthetic_state(odds, gamma)
        tasks = [Threads.@spawn synthetic_state(odds, gamma) for _ in 1:4]
        for actual in fetch.(tasks)
            @test actual.logrates ≈ expected.logrates
            @test actual.rsums ≈ expected.rsums
            @test actual.logtot ≈ expected.logtot
        end
        reference = big_rates(odds, gamma, 5.0, 2.0)
        @test isapprox(expected.logrates, log.(reference); atol = 2.0e-12)
    end
    p = 100_001
    logs, conds, pips, odds = zeros(p), zeros(p), zeros(p), fill(800.0, p)
    active = Int[]
    BASSGWAS.comp_rates2!(logs, conds, pips, odds, active, 1:p, 5.0)
    bytes = @allocated BASSGWAS.comp_rates2!(logs, conds, pips, odds, active, 1:p, 5.0)
    @test bytes < 8192 * (Threads.nthreads() + 1)
end

@testset "Extreme detailed balance and importance weights" begin
    logposterior = [0.0, 800.0, 801.0, 2.0]
    transition, logtotals = zeros(4, 4), zeros(4)
    for mask in 0:3
        odds = [logposterior[toggle_index(mask, i)] - logposterior[mask + 1] for i in 1:2]
        state = synthetic_state(odds, inclusion_vector(mask, 2))
        logtotals[mask + 1] = state.logtot
        probabilities = diff([0.0;state.rsums]) / state.rsums[end]
        for i in 1:2
            transition[mask + 1, toggle_index(mask, i)] = probabilities[i]
        end
        transition[mask + 1, mask + 1] = probabilities[end]
    end
    normalized(logs) = (w = exp.(BigFloat.(logs) .- maximum(logs)); w / sum(w))
    stationary = normalized(logposterior + logtotals)
    flow = Diagonal(stationary) * transition
    @test isapprox(flow, flow'; atol = big"1e-12")
    recovered = stationary .* exp.(-BigFloat.(logtotals)); recovered /= sum(recovered)
    @test isapprox(recovered, normalized(logposterior); atol = big"1e-12")
    ps = numerical_fixture(2; s = 1.0)
    samp = ProbitBVSSampler(ps.Xaug[:, 1:2], Int.(ps.Y .> 0), ps.U, ps.S, 1.0, 2.0; nadapt = 0, rng = Xoshiro(32))
    ref = synthetic_state([709.0, 709.0], [0, 0])
    samp.wTGSState.logtot = ref.logtot
    expected = inv(sum(big_rates([709.0, 709.0], [0, 0], 5.0, 2.0)))
    draw = get_state(samp)
    @test isapprox(draw.Z, expected; rtol = 1.0e-12, atol = 0)
    @test draw.Z > 0
    @test keys(draw) == (:γ, :pips, :icept, :beta, :u, :Y, :sigma, :r, :c, :Z)
end
