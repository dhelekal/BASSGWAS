# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only
include("numerical_helpers.jl")

@testset "Dependency-specific cache refreshes" begin
    ps = numerical_fixture(4)
    sc = BASSGWAS.makeSCache(ps)
    state,qc = BASSGWAS.init_state2(BASSGWAS.StdNormLogLik{Float64}(),ps,sc,Int64[1,0,1,0])
    cv,rs = sc.covariance,sc.response
    oldq,oldg,oldyu,oldyx = copy(cv.QX),copy(cv.gramdiag),copy(rs.yu),copy(rs.yx)
    cg,rg = cv.generation,rs.generation
    oldfactor = copy(qc.work.factor)
    ps.Y .+= 0.17
    BASSGWAS.updateSCache!(sc,ps)
    BASSGWAS.prepare_active!(qc,sc,ps.c,ps.mc)
    @test cv.generation == cg
    @test rs.generation == rg+1
    @test cv.QX == oldq && cv.gramdiag == oldg
    @test qc.work.factor == oldfactor
    @test rs.yu ≈ ps.U' * ps.Y
    @test rs.yx ≈ ps.Xaug' * ps.Y
    yu,yx = copy(rs.yu),copy(rs.yx)
    ps.r *= 2
    BASSGWAS.update_covariance!(sc,ps.r,ps.s)
    @test cv.generation == cg+1
    @test rs.generation == rg+1
    @test rs.yu == yu && rs.yx == yx
    BASSGWAS.prepare_active!(qc,sc,ps.c,ps.mc)
    @test qc.work.factor != oldfactor
    @test isapprox(qc.work.ssq,dense_model_terms(ps,state.γ).ssq;rtol=NUM_RTOL,atol=NUM_ATOL)
    G = copy(qc.covdata)
    ps.c *= 1.5
    BASSGWAS.prepare_active!(qc,sc,ps.c,ps.mc)
    @test qc.covdata == G
    @test isapprox(qc.work.det,dense_model_terms(ps,state.γ).det;rtol=NUM_RTOL,atol=NUM_ATOL)
    ps.mc *= 0.5
    BASSGWAS.prepare_active!(qc,sc,ps.c,ps.mc)
    @test qc.covdata == G
    @test isapprox(qc.work.det,dense_model_terms(ps,state.γ).det;rtol=NUM_RTOL,atol=NUM_ATOL)
    ps.s *= 0.7
    BASSGWAS.update_covariance!(sc,ps.r,ps.s)
    @test cv.generation == cg+2
    @test rs.generation == rg+1
    @test rs.yu == yu && rs.yx == yx
    BASSGWAS.prepare_active!(qc,sc,ps.c,ps.mc)
    @test isapprox(qc.work.ssq,dense_model_terms(ps,state.γ).ssq;rtol=NUM_RTOL,atol=NUM_ATOL)
    ps.Y = reverse(ps.Y)
    BASSGWAS.update_response!(sc,ps.Y)
    @test rs.yx ≈ ps.Xaug' * ps.Y
    @test_throws DimensionMismatch BASSGWAS.update_response!(sc,zeros(ps.n+1))
    @test_throws ArgumentError BASSGWAS.update_covariance!(sc,0.0,ps.s)
    @test_throws ArgumentError BASSGWAS.update_covariance!(sc,ps.r,NaN)
    ps.Xaug = copy(ps.Xaug)
    @test_throws ArgumentError BASSGWAS.updateSCache!(sc,ps)
end

@testset "Empty, zero and near-zero covariance modes" begin
    for rank in (0,1,3,8), design in (:correlated,:degenerate)
        ps = numerical_fixture(4;rank,design)
        if rank > 0
            ps.S[1] = 0
        end
        if rank > 1
            ps.S[2] = 1e-20
        end
        for s in (0.35,1.0,2.0)
            ps.s = s
            sc = BASSGWAS.makeSCache(ps)
            C = s*I+ps.r*ps.U*Diagonal(ps.S)*ps.U'
            @test isapprox(sc.cdet,logdet(Symmetric(C));rtol=NUM_RTOL,atol=NUM_ATOL)
            @test isapprox(sc.qterm,dot(ps.Y,C\ps.Y);rtol=NUM_RTOL,atol=NUM_ATOL)
            @test isapprox(sc.covariance.gramdiag,diag(ps.Xaug'*(C\ps.Xaug));rtol=NUM_RTOL,atol=NUM_ATOL)
            for ll in (BASSGWAS.StdNormLogLik{Float64}(),BASSGWAS.TDistLogLik(3.5,1.2,8))
                exact = enumerate_posterior(ps,ll)
                for mask in 0:15
                    state,qc = BASSGWAS.init_state2(ll,ps,sc,inclusion_vector(mask,4))
                    check_state(state,qc,ps,sc,ll,exact,mask)
                end
            end
        end
    end
    ps = numerical_fixture(4)
    ps.S[1] = -1.0
    @test_throws ArgumentError BASSGWAS.makeSCache(ps)
    ps.S[1] = Inf
    @test_throws ArgumentError BASSGWAS.makeSCache(ps)
    ps.S = ones(2)
    @test_throws DimensionMismatch BASSGWAS.makeSCache(ps)
end

@testset "Guarded weights and cancellation-sensitive forms" begin
    ps = numerical_fixture(4;rank=3)
    ps.r, ps.s = 1e16,0.7
    ps.Y = copy(ps.U[:,1])
    ps.Xaug[:,1] .= ps.Y
    sc = BASSGWAS.makeSCache(ps)
    expected = inv(ps.s+ps.r*ps.S[1])
    @test isapprox(sc.qterm,expected;rtol=1e-10)
    @test isapprox(sc.covariance.gramdiag[1],expected;rtol=1e-10)
    d,d2,v = zeros(1),zeros(1),zeros(1)
    ld = BASSGWAS._covariance_weights!(d,d2,v,[1e200],1e200,1.0,1)
    @test isapprox(ld,400log(10);rtol=1e-14)
    @test d2[1] == 1
    @test isfinite(ld)
    ld = BASSGWAS._covariance_weights!(d,d2,v,[1e-200],1e-200,1e-200,1)
    @test isapprox(d2[1],1.0;rtol=1e-12)
    @test isfinite(ld) && isfinite(v[1])
end

@testset "Proposals, conditional forms and accepted-state isolation" begin
    for rank in (0,3,8), mask in (0,5,15)
        ps = numerical_fixture(4;rank)
        sc = BASSGWAS.makeSCache(ps)
        state,qc = BASSGWAS.init_state2(BASSGWAS.StdNormLogLik{Float64}(),ps,sc,inclusion_vector(mask,4))
        accepted = (copy(sc.QX),copy(sc.response.fx),sc.qterm,sc.cdet,copy(qc.covdata),copy(qc.work.factor))
        r0,c0,s0 = ps.r,ps.c,ps.s
        for (r,c,s) in ((0.2,0.4,0.6),(1.3,2.4,1.0),(2.0,0.7,2.1))
            q,det = BASSGWAS.proposal_terms!(qc,sc,r,c,ps.mc;s)
            ps.r,ps.c,ps.s = r,c,s
            expected = dense_model_terms(ps,state.γ)
            @test isapprox(q,expected.ssq;rtol=NUM_RTOL,atol=NUM_ATOL)
            @test isapprox(det,expected.det;rtol=NUM_RTOL,atol=NUM_ATOL)
            Xa = BASSGWAS.get_X1(qc)
            M = s*I+r*ps.U*Diagonal(ps.S)*ps.U'+Xa*Diagonal([fill(c,qc.d-1);ps.mc])*Xa'
            Ynew = reverse(ps.Y) .+ 0.2
            q11,q1y = BASSGWAS.shift_terms!(qc,sc,Ynew)
            @test isapprox(q11,sum(M\ones(ps.n));rtol=NUM_RTOL,atol=NUM_ATOL)
            @test isapprox(q1y,sum(M\Ynew);rtol=NUM_RTOL,atol=NUM_ATOL)
            @test isapprox(BASSGWAS.response_ssq!(qc,sc,Ynew),dot(Ynew,M\Ynew);rtol=NUM_RTOL,atol=NUM_ATOL)
        end
        @test accepted == (sc.QX,sc.response.fx,sc.qterm,sc.cdet,qc.covdata,qc.work.factor)
        ps.r,ps.c,ps.s = r0,c0,s0
        q,det = BASSGWAS.proposal_terms!(qc,sc,ps.r,ps.c,ps.mc)
        @test isapprox(q,qc.work.ssq;rtol=NUM_RTOL,atol=NUM_ATOL)
        @test isapprox(det,qc.work.det;rtol=NUM_RTOL,atol=NUM_ATOL)
        expected_rng = Xoshiro(17)
        actual_rng = copy(expected_rng)
        expected_sigma = rand(expected_rng,InverseGamma(2+ps.n/2,0.8+dense_model_terms(ps,state.γ).ssq/2))
        Y = copy(ps.Y)
        sigma = BASSGWAS.draw_Ysigma!(Y,2.0,0.8,q,actual_rng)
        @test isapprox(sigma,expected_sigma;rtol=NUM_RTOL,atol=NUM_ATOL)
        @test rand(actual_rng) == rand(expected_rng)
    end
end

function workspace_fixture(p; n=32,k=8,a=10,seed=93)
    rng = Xoshiro(seed)
    U = Matrix(qr(randn(rng,n,k)).Q)[:,1:k]
    ps = BASSGWAS.ParState2(randn(rng,n,p),randn(rng,n),U,ones(k),0.7,2.0,1.3,0.8,1.0,4.0)
    sc = BASSGWAS.makeSCache(ps)
    gamma = zeros(Int64,p); gamma[1:min(a,p)] .= 1
    ll = BASSGWAS.StdNormLogLik{Float64}()
    state,qc = BASSGWAS.init_state2(ll,ps,sc,gamma)
    return ps,sc,state,qc,ll
end

@testset "Incremental products through capacity growth and covariance changes" begin
    ps,sc,state,qc,ll = workspace_fixture(24;a=0)
    rng = Xoshiro(7)
    # The prefix forces capacity growth; the random walk then exercises removal,
    # insertion at either end, response-only refresh and covariance invalidation.
    for (step,index) in enumerate([collect(1:24);rand(rng,1:24,250)])
        BASSGWAS.flip_gamma!(state,ps,qc,index)
        if step % 17 == 0
            ps.r *= 1.01
            ps.s *= 1.003
            BASSGWAS.update_covariance!(sc,ps.r,ps.s)
        elseif step % 13 == 0
            ps.Y .*= 0.99
            BASSGWAS.update_response!(sc,ps.Y)
        end
        BASSGWAS.update_state2!(state,ps,sc,qc,ll)
        C = ps.s*I+ps.r*ps.U*Diagonal(ps.S)*ps.U'
        @test isapprox(qc.covdata[1:qc.d,:],BASSGWAS.get_X1(qc)'*(C\ps.Xaug);rtol=NUM_RTOL,atol=NUM_ATOL)
        check_qcache(qc,ps,state.γ)
        expected = dense_model_terms(ps,state.γ)
        @test isapprox(qc.work.ssq,expected.ssq;rtol=NUM_RTOL,atol=NUM_ATOL)
        @test isapprox(qc.work.det,expected.det;rtol=NUM_RTOL,atol=NUM_ATOL)
    end
end

function workspace_allocations(p)
    ps,sc,state,qc,ll = workspace_fixture(p)
    # Reserve enough capacity for repeated additions/removals before measuring.
    BASSGWAS._reserve_active!(qc,qc.d+1)
    BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    ids1 = sort([qc.i1_set;p]); ids0 = collect(11:(p-1))
    original1,original0 = qc.i1_set,qc.i0_set
    add() = BASSGWAS.add_active_QC(qc,ps,p,ids1,ids0)
    remove() = BASSGWAS.remove_active_QC(qc,ps,p,original1,original0)
    for _ in 1:2
        add(); BASSGWAS.update_state2!(state,ps,sc,qc,ll)
        remove(); BASSGWAS.update_state2!(state,ps,sc,qc,ll)
        BASSGWAS.proposal_terms!(qc,sc,0.9,1.1,ps.mc)
    end
    refresh = @allocated BASSGWAS.updateSCache!(sc,ps)
    statebytes = @allocated BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    addbytes = @allocated add()
    BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    removebytes = @allocated remove()
    BASSGWAS.proposal_terms!(qc,sc,0.9,1.1,ps.mc)
    proposal = @allocated BASSGWAS.proposal_terms!(qc,sc,0.95,1.2,ps.mc)
    return (;refresh,statebytes,addbytes,removebytes,proposal)
end

@testset "Bounded warm-workspace allocations" begin
    for p in (2048,20000)
        bytes = workspace_allocations(p)
        # Bounds allow small factorization/view wrappers, but exclude even one
        # predictor-sized Float64 allocation, let alone a*k or a*P temporaries.
        for value in values(bytes)
            @test value < 8192
        end
    end
end

function sampler_smoke(seed)
    ps,_,_,_,_ = workspace_fixture(8;a=0,seed)
    sampler = BASSGWAS.ProbitBVSSampler(ps.Xaug[:,1:ps.p],Int.(ps.Y .> 0),ps.U,ps.S,0.7,2.0;
        nadapt=100,adapt_xi=false,rng=Xoshiro(seed))
    for _ in 1:35
        BASSGWAS.next_step!(sampler)
    end
    output = BASSGWAS.get_state(sampler)
    return (copy(output.γ),copy(output.pips),output.beta,output.Y,output.r,output.c)
end

@testset "Independent concurrent sampler instances" begin
    serial = [sampler_smoke(seed) for seed in (19,27)]
    tasks = [Threads.@spawn sampler_smoke(seed) for seed in (19,27)]
    @test fetch.(tasks) == serial
    @test all(all(isfinite,x[2]) && all(isfinite,x[3]) && all(isfinite,x[4]) for x in serial)
end

@testset "Respect caller-owned BLAS settings" begin
    original_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(2)
        ps,sc,state,qc,ll = workspace_fixture(4096;a=20)
        @test BLAS.get_num_threads() == 2
        before = copy(state.dets),copy(state.ssq)
        BLAS.set_num_threads(1)
        BASSGWAS.update_state2!(state,ps,sc,qc,ll)
        @test isapprox(state.dets,before[1];rtol=NUM_RTOL,atol=NUM_ATOL)
        @test isapprox(state.ssq,before[2];rtol=NUM_RTOL,atol=NUM_ATOL)
        @test hasproperty(sc,:YtU) && hasproperty(qc,:XtX)
    finally
        BLAS.set_num_threads(original_threads)
    end
end

@testset "Candidate failures return to the caller" begin
    ps,sc,state,qc,ll = workspace_fixture(4096;a=20)
    index = ps.p
    previous = sc.covariance.gramdiag[index]
    sc.covariance.gramdiag[index] = -1e100
    @test_throws PosDefException BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    sc.covariance.gramdiag[index] = previous
    BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    @test all(isfinite,state.dets) && all(isfinite,state.ssq)
end
