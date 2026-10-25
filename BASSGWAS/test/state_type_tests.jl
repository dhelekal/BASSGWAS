# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only

include("numerical_helpers.jl")

@testset "Typed covariance and response construction" begin
    for ns in (0,2)
        covariance = @inferred BASSGWAS.CovarianceState{Float64}(2,3//2,ns,4)
        response = @inferred BASSGWAS.ResponseState{Float64}(6,ns,4)
        @test covariance.r === 2.0 && covariance.s === 1.5
        @test covariance.sinv === inv(1.5)
        @test covariance.QX isa Matrix{Float64} && size(covariance.QX) == (ns,4)
        @test response.yu isa Vector{Float64} && length(response.yu) == ns
        @test length(response.residual) == 6 && length(response.fx) == 4
    end
    @test_throws ArgumentError BASSGWAS.CovarianceState{Float64}(0,1,2,4)
    @test_throws ArgumentError BASSGWAS.CovarianceState{Float64}(1,NaN,2,4)
    @test_throws ArgumentError BASSGWAS.CovarianceState{AbstractFloat}(1,1,2,4)
    @test_throws ArgumentError BASSGWAS.ResponseState{AbstractFloat}(6,2,4)
end

@testset "Concrete floating-point parameter storage" begin
    X = [1 2; 3 4; 5 6; 7 8]
    Y = [1.0,-1.0,0.5,-0.5]
    U = Matrix{Float64}(I,4,2)
    S = [0.5,1.5]
    # In particular, an integer mc must participate in scalar conversion.
    ps = @inferred BASSGWAS.ParState2(X,Y,U,S,1,2.0,3,4.0,1,10)
    @test ps isa BASSGWAS.ParState2{Float64}
    @test ps.Xaug == hcat(X,ones(4))
    @test ps.p === 2 && ps.n === 4
    @test ps.Y === Y && ps.U === U && ps.S === S
    @test all(name -> getfield(ps,name) isa Float64,(:alpha,:beta,:c,:r,:s,:mc))
    @test ps.mc === 10.0

    # Convert integer/rational inputs and non-dense wrappers at the boundary.
    converted = BASSGWAS.ParState2{Float64}(view(X,:,:),[1,-1,1,-1],
        view(U,:,:),[1//2,3//2],1,2,3,4,1,10)
    @test converted.Xaug isa Matrix{Float64}
    @test converted.Y isa Vector{Float64}
    @test converted.U isa Matrix{Float64}
    @test converted.S == S
    @test BASSGWAS.ParState2(X,[1,-1,1,-1],Int.(U),[1,2],1,2,3,4,1,10) isa BASSGWAS.ParState2{Float64}
    raw = BASSGWAS.ParState2(ps.Xaug,Y,U,S,2,4,1,2,3,4,1,10)
    @test raw.Xaug === ps.Xaug
    sc = BASSGWAS.makeSCache(converted)
    @test sc.data.Xaug === converted.Xaug
    @test sc.data.U === converted.U && sc.data.S === converted.S
    converted.Y = reverse(converted.Y)
    BASSGWAS.updateSCache!(sc,converted)
    @test sc.response.yx ≈ converted.Xaug'converted.Y

    @test_throws ArgumentError BASSGWAS.ParState2{AbstractFloat}(X,Y,U,S,1,2,3,4,1,10)
    @test_throws DimensionMismatch BASSGWAS.ParState2(X,Y[1:3],U,S,1,2,3,4,1,10)
    @test_throws DimensionMismatch BASSGWAS.ParState2(X,Y,U,[1.0],1,2,3,4,1,10)
    @test_throws DimensionMismatch BASSGWAS.ParState2(ps.Xaug,Y,U,S,3,4,1,2,3,4,1,10)
    for badS in ([-1.0,1.0],[NaN,1.0],[Inf,1.0])
        @test_throws ArgumentError BASSGWAS.ParState2(X,Y,U,badS,1,2,3,4,1,10)
    end
    for index in 1:6, bad in (0.0,-1.0,Inf,NaN)
        parameters = ones(6)
        parameters[index] = bad
        @test_throws ArgumentError BASSGWAS.ParState2(X,Y,U,S,parameters...)
    end
    for index in 1:3
        arrays = [Float64.(X),copy(Y),copy(U)]
        arrays[index][1] = NaN
        @test_throws ArgumentError BASSGWAS.ParState2(arrays...,S,1,2,3,4,1,10)
    end
end

@testset "Weighted-TGS storage and input validation" begin
    ps = numerical_fixture(2)
    sc = BASSGWAS.makeSCache(ps)
    ll = BASSGWAS.StdNormLogLik{Float64}()
    for gamma in (Int[0,1],Int8[0,1],Bool[0,1],view([0,1,0],1:2))
        state,qc = BASSGWAS.init_state2(ll,ps,sc,gamma;eps=0,xi_init=1)
        @test state isa BASSGWAS.wTGSState2{Float64}
        @test state.γ isa Vector{Int}
        @test state.γ !== gamma && state.γ == gamma
        @test state.eps === 0.0 && state.xi === 1.0
        @test all(name -> getfield(state,name) isa Vector{Float64},
                  (:dets,:ssq,:llrs,:pips,:conds,:rates,:rsums))
        @test qc isa BASSGWAS.QCache{Float64}
        @test isapprox(state.ll_curr,reference_loglik(ll,dense_model_terms(ps,gamma)...);
                       rtol=NUM_RTOL,atol=NUM_ATOL)
    end
    @test_throws DimensionMismatch BASSGWAS.init_state2(ll,ps,sc,[0])
    @test_throws ArgumentError BASSGWAS.init_state2(ll,ps,sc,[0,2])
    @test_throws ArgumentError BASSGWAS.init_state2(ll,ps,sc,[0,1];eps=-1)
    @test_throws ArgumentError BASSGWAS.init_state2(ll,ps,sc,[0,1];xi_init=Inf)

    state,_ = BASSGWAS.init_state2(ll,ps,sc,[0,1])
    args = [getfield(state,name) for name in fieldnames(typeof(state))]
    @test BASSGWAS.wTGSState2(args...) isa BASSGWAS.wTGSState2{Float64}
    @test_throws ArgumentError BASSGWAS.wTGSState2{AbstractFloat}(args...)
    for index in 3:9
        invalid = copy(args)
        invalid[index] = Float64[]
        @test_throws DimensionMismatch BASSGWAS.wTGSState2{Float64}(invalid...)
    end
    invalid = copy(args)
    invalid[2] = [0,2]
    @test_throws ArgumentError BASSGWAS.wTGSState2{Float64}(invalid...)

    # No selectable predictors: only the auxiliary move remains.
    ps0 = numerical_fixture(0)
    state0,_ = BASSGWAS.init_state2(ll,ps0,BASSGWAS.makeSCache(ps0),Int[])
    @test state0.rates == [1.0] && state0.rsums == [1.0]
end

@testset "Float64 sampler transitions and adaptation" begin
    ps = numerical_fixture(3;s=1.0)
    for sample_c in (false,true)
        sampler = BASSGWAS.ProbitBVSSampler(view(ps.Xaug,:,1:ps.p),Int.(ps.Y .> 0),
            view(ps.U,:,:),ps.S,1,2;sample_c,c=2,mcinit=10,eps=1,xi_init=1,
            nadapt=24,rng=Xoshiro(304))
        @test sampler.parState isa BASSGWAS.ParState2{Float64}
        @test sampler.wTGSState isa BASSGWAS.wTGSState2{Float64}
        @test sampler.adaptor isa BASSGWAS.RWMAdaptor{Float64}
        @test sampler.loglik isa BASSGWAS.TDistLogLik{Float64}
        for _ in 1:14
            BASSGWAS.next_step!(sampler)
        end
        @test sampler.adaptor.stage == 3
        @test all(isfinite,sampler.parState.Y)
        @test sign.(sampler.parState.Y) == sign.(ps.Y)
        @test isfinite(sampler.wTGSState.ll_curr)
        @test isfinite(sampler.adaptor.dt) && all(isfinite,sampler.adaptor.scales)
        @test sampler.parState.Y isa Vector{Float64}
        @test sample_c || sampler.parState.c ≈ 2.0
        posterior = BASSGWAS.get_state(sampler)
        @test posterior.beta isa Vector{Float64}
    end
end
