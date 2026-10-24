# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only
# Run with --project=BASSGWAS --threads=1 (or 4). Set BASSGWAS_BASELINE_SOURCE
# to an older src/BASSGWAS.jl to run exactly the same workload on that version.
using LinearAlgebra, Random, Printf
include(get(ENV,"BASSGWAS_BASELINE_SOURCE",joinpath(@__DIR__,"..","src","BASSGWAS.jl")))
using .BASSGWAS
BLAS.set_num_threads(1)

function measure(f; setup=()->nothing, samples=11)
    for _ in 1:3
        setup(); f()
    end
    times = Float64[]
    for _ in 1:samples
        setup()
        push!(times,@elapsed f())
    end
    setup()
    bytes = @allocated f()
    return (ms=sort!(times)[cld(samples,2)]*1000,bytes=bytes)
end

function benchmark_case(n,p,k,a)
    rng = Xoshiro(47)
    X,Y = randn(rng,n,p),randn(rng,n)
    U = Matrix(qr(randn(rng,n,k)).Q)[:,1:k]
    S = collect(range(0.1,5.0;length=k))
    ps = BASSGWAS.ParState2(X,Y,U,S,0.5,2.0,1.7,0.8,1.0,4.2)
    sc = BASSGWAS.makeSCache(ps)
    gamma = zeros(Int64,p); gamma[1:a] .= 1
    ll = BASSGWAS.TDistLogLik(3.5,1.2,n)
    state,qc = BASSGWAS.init_state2(ll,ps,sc,gamma)
    flip() = BASSGWAS.flip_gamma!(state,ps,qc,p)
    refresh_state() = BASSGWAS.update_state2!(state,ps,sc,qc,ll)
    function add_update()
        flip(); refresh_state()
    end
    function ensure_absent()
        if state.γ[p] == 1
            flip(); refresh_state()
        end
    end
    function ensure_present()
        if state.γ[p] == 0
            flip(); refresh_state()
        end
    end
    function changed_covariance()
        ps.r = ps.r == 0.8 ? 0.9 : 0.8
    end
    results = Pair{String,Any}[]
    push!(results,"response_refresh"=>measure(()->BASSGWAS.updateSCache!(sc,ps)))
    push!(results,"covariance_and_response"=>measure(()->BASSGWAS.updateSCache!(sc,ps);setup=changed_covariance))
    refresh_state()
    push!(results,"same_state_update"=>measure(refresh_state))
    push!(results,"add_and_update"=>measure(add_update;setup=ensure_absent))
    push!(results,"remove_and_update"=>measure(add_update;setup=ensure_present))
    ensure_absent()
    Xa = BASSGWAS.get_X1(qc); XtX = Xa'Xa; UtX = sc.UtX[:,qc.i1_set]
    function proposals()
        for j in 1:25
            r,c = 0.8+0.01j,1.7+0.01j
            if isdefined(BASSGWAS,:proposal_terms!)
                BASSGWAS.proposal_terms!(qc,sc,r,c,ps.mc)
            else
                BASSGWAS._comp_llterms(Y,r,c,ps.mc,Xa,XtX,UtX,sc.YtU,S)
            end
        end
    end
    push!(results,"25_proposals"=>measure(proposals))
    sampler = BASSGWAS.ProbitBVSSampler(X,Int.(Y .> 0),U,S,0.5,2.0;
        nadapt=100,adapt_xi=false,rng=Xoshiro(87))
    for i in 1:a
        BASSGWAS.flip_gamma!(sampler.wTGSState,sampler.parState,sampler.qcache,i)
    end
    BASSGWAS.update_state2!(sampler.wTGSState,sampler.parState,sampler.scache,sampler.qcache,sampler.loglik)
    function auxiliary_transition()
        BASSGWAS.gibbs_CG!(sampler)
        BASSGWAS.update_state2!(sampler.wTGSState,sampler.parState,sampler.scache,sampler.qcache,sampler.loglik)
    end
    push!(results,"auxiliary_transition"=>measure(auxiliary_transition;samples=7))
    println("n=$n p=$p k=$k active=$(a+1) threads=$(Threads.nthreads())")
    for (name,result) in results
        @printf("  %-25s %10.3f ms %12d bytes\n",name,result.ms,result.bytes)
    end
    return nothing
end

println("Julia ",VERSION,"; ",BLAS.get_config(),"; BLAS threads=",BLAS.get_num_threads())
cases = [(300,10000,100,10),(300,10000,300,40),(300,100000,100,10),(300,100000,300,40)]
if "--quick" in ARGS
    cases = cases[1:2]
end
for args in cases
    benchmark_case(args...)
    GC.gc()
end
