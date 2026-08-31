# Copyright (C) 2025–2026 David Helekal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

mutable struct ProbitBVSSampler{Tp, Tc, Ts, Tk, Tl, Ti, Tb, Tn, Ta, Tr, Tv}
    parState::Tp
    scache::Tc
    wTGSState::Ts
    qcache::Tk
    loglik::Tl
    T::Ti
    naux::Ti
    nadapt::Ti
    isadapted::Tb
    adapt_xi::Tb
    xi_targ::Tn
    sample_c::Tb
    adaptor::Ta
    shape::Tn
    scale::Tn
    rng::Tr
    perf::Tv
end

function ProbitBVSSampler(X::AbstractMatrix{Tx}, Y::AbstractVector{Ty}, U::AbstractMatrix{Tx}, S::AbstractVector{Tx},
    alpha, beta; c=6.0, sample_c=true, mcinit=100.0, eps=5.0, nadapt=10000, xi_init = 5.0, adapt_xi=true, xi_targ=0.25, nu=5.0, lambda=0.6, rng=Random.default_rng()) where {Tx<: Real, Ty<:Integer}
    
    alpha = Tx(alpha)
    beta = Tx(beta)
    nu = Tx(nu)
    lambda = Tx(lambda)
    eps = Tx(eps)
    p = size(X, 2)    
    n = size(Y, 1)
    
    γ0 = zeros(Int64, p)

    shape = nu/2.0
    scale = shape * lambda
    
    loglik = TDistLogLik(nu, lambda, n)

    signs = 1.0 * (Y .> 0) .+ -1.0 * (Y .<= 0)
    Yinit = abs.(randn(rng, n)) .* signs
    rinit = abs(1.0*randn(rng)) + 1e-8
    cinit = sample_c ? abs(1.0*randn(rng)) + 1e-8 : c

    sinit=1.0

    parState = ParState2(X, Yinit, U, S, alpha, beta, cinit, rinit, sinit, mcinit)
    cache = makeSCache(parState)
    tgsstate, qcache = init_state2(loglik, parState, cache, γ0; eps=eps, xi_init=xi_init)

    adaptor = RWMAdaptor(sample_c ? 2 : 1, 25*div(nadapt,12))

    updateSCache!(cache, parState)
    update_state2!(tgsstate, parState, cache, qcache, loglik)
    return ProbitBVSSampler(parState, cache, tgsstate, qcache, loglik, 0, 0, nadapt, !adapt_xi, adapt_xi, xi_targ, sample_c, adaptor, shape, scale, rng, zeros(2,5))
end

function next_step!(samp::ProbitBVSSampler)
    t = @elapsed if samp.T > 10
        activatePerf()
        next_flip = sample_next2(samp.wTGSState, samp.rng)
        if next_flip <= samp.parState.p
            t = @elapsed flip_gamma!(samp.wTGSState, samp.parState, samp.qcache, next_flip)
            perf!("flip", t)
            t = @elapsed update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
            perf!("upstate", t)
        else
            t = @elapsed gibbs_CG!(samp)
            perf!("gibbs", t)
            t = @elapsed update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
            perf!("upstate", t)
        end
        #Adapt xi
        if !samp.isadapted && samp.adapt_xi
            xi_targ = samp.xi_targ
            xi = samp.wTGSState.xi
            rs = samp.wTGSState.rsums[end]
            samp.wTGSState.xi = xi + (xi_targ - xi/rs)/(sqrt(samp.T))
        end
    else
        t = @elapsed gibbs_CG!(samp)
        perf!("gibbs", t)
        t = @elapsed update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
        perf!("upstate", t)

    end
    samp.T += 1
    if samp.T > samp.nadapt 
        samp.isadapted = true
    end
    perf!("nextstep", t)

    n_active = sum(samp.wTGSState.γ)
    n_active >= 100 &&
        @warn "Warning: The number of active variants has exceed 100. Number of active variants: $(n_active). If this warning appears repeatedly, the model is likely misspecified, or there is unmeasured confounding present!"
end

function gibbs_CG!(samp::ProbitBVSSampler)
    shiftmv = true 
    scalemv = true 
    scale1 = true
    
    rng = samp.rng
    ps = samp.parState
    sc = samp.scache
    qc = samp.qcache
    shape = samp.shape
    scale = samp.scale

    Y = ps.Y 
    r = ps.r
    c = ps.c
    mc = ps.mc

    X = get_X1(qc)
    XtX = X'X
    UtX = sc.UtX[:, qc.i1_set]
    YtU = sc.YtU
    U = sc.U
    S = sc.S
    V = sc.V

    rsqrt = sqrt(r)
    csqrt = sqrt(c)

    ll_curr = samp.wTGSState.ll_curr
    
    rprior = Exponential(-3.0/log(0.05))
    cprior = truncated(InverseGamma(6,12); lower = 0.0) 

    adaptor = samp.adaptor

    for i in 1:25

        dt = get_dt(adaptor)
        rwm_scales = get_scales(adaptor)

        #jitter to ensure pd; 1e-4 -> 1e-8 on variance scale
        rprop = abs(rsqrt + i*randn(rng)*dt*rwm_scales[1] - 1e-4) + 1e-4
        cprop = csqrt

        if samp.sample_c
            #jitter to ensure pd; 1e-4 -> 1e-8 on variance scale
            cprop = abs(csqrt + i*randn(rng)*dt*rwm_scales[2] - 1e-4) + 1e-4
        end

        prll_curr = logpdf(rprior,rsqrt) + logpdf(cprior,csqrt)
        prll_prop = logpdf(rprior,rprop) + logpdf(cprior,cprop)
        
        ssq, det = _comp_llterms(Y, rprop*rprop, cprop*cprop, mc, X, XtX, UtX, YtU, S)
        ll_prop = samp.loglik(det, ssq)

        qr = ll_prop + prll_prop - ll_curr - prll_curr
        u = log(rand(rng))
        
        accept = 0

        if u<=qr
            accept = 1
            rsqrt = rprop
            csqrt = cprop
            ll_curr = ll_prop
        end

        if !samp.isadapted
            adapt!(adaptor, [rsqrt, csqrt], accept)
        end
    end

    r_new = rsqrt * rsqrt
    c_new = csqrt * csqrt

    signs = sign.(Y)

    n = size(Y, 1)
    l = size(X, 2)
    m = size(U, 2)

    K = [X V]
    Y_old = copy(Y)
    
    if scalemv && scale1
        _ = draw_Ysigma!(Y_old, r_new, c_new, mc, shape, scale, X, XtX, UtX, U, S, rng)
    end

    t = @elapsed beta = draw_betas(Y_old, r_new, c_new, mc, l, m, K, rng)
    perf!("betas", t)

    #Y_new are Y means, to be reused
    Y_new = K*beta

    for i in eachindex(Y_new)
        y_mu = Y_new[i]
        if signs[i] >= 0.0
            Y_new[i] = rand(rng, truncated(Normal(y_mu, 1.0); lower=0.0))
        else
            Y_new[i] = rand(rng, truncated(Normal(y_mu, 1.0); upper=0.0))
        end
    end

    #PX
    #Shift move
    if shiftmv
        sig=10.0
        mu = rand(rng, Normal(0.0, sig))
        Y_new .+= mu
        I1 = ones(n)
        sqI1 = _comp_ssq(I1, r_new, c_new, mc, X, XtX, UtX, U, S)
        YI1 = _comp_ssq(I1, Y_new, r_new, c_new, mc, X, XtX, UtX, U, S)
        tauup = (1.0/sig*sig + sqI1)
        sigup = 1.0/tauup
        muup = YI1 * sigup
        lo = maximum(Y_new[findall(signs .< 0.0)])
        up = minimum(Y_new[findall(signs .> 0.0)])
        mu_new = rand(rng, truncated(Normal(muup, sqrt(sigup)), lower=lo, upper=up))
        Y_new .-= mu_new
    end

    #Scale move
    if scalemv
        sigma = rand(rng, InverseGamma(shape, scale))
        Y_new .*= sqrt(sigma)
    end

    if scalemv && !scale1
        _ = draw_Ysigma!(Y_new, r_new, c_new, mc, shape, scale, X, XtX, UtX, U, S, rng)
    end
    
    ps.Y = Y_new
    ps.c = c_new
    ps.r = r_new

    t = @elapsed updateSCache!(sc, ps)
    perf!("ggibs.sc", t)
end

#Draw sigma^2 | Y from a normal - inverse-gamma model and standardise Y := Y/sigma 
function draw_Ysigma!(Y, r, c, mc, shape, scale, X, XtX, UtX, U, S, rng)
    n = size(Y, 1)
    
    ssq = _comp_ssq(Y, r, c, mc, X, XtX, UtX, U, S)
            
    aup = shape + n/2.0
    bup = scale + ssq/2.0
            
    sigma = rand(rng, InverseGamma(aup, bup))
    Y .*= 1.0/sqrt(sigma)
    return sigma
end

#Draw beta | Y from a normal - normal model
function draw_betas(Y, r, c, mc, l, m, K, rng)
    
    Qinv = K'K

    #Predictor precision
    for i in 1:(l-1)
        Qinv[i,i] += 1.0/c
    end

    #Intercept precision
    Qinv[l,l] += 1.0/mc

    #Random effect precision
    for i in (l+1):(m+l)
        Qinv[i,i] += 1.0/r
    end

    Qf = cholesky(Qinv)
    beta_mu = Qf\(K'*Y)
    eta = rand(rng, Normal(0.0,1.0), l+m)
    beta = Qf.U \ eta + beta_mu 

    return beta
end

function get_state(samp::ProbitBVSSampler)

    ps = samp.parState
    sc = samp.scache
    qc = samp.qcache

    γ = samp.wTGSState.γ
    pips = samp.wTGSState.pips
    rates = samp.wTGSState.rates
    Z = 1.0/sum(rates)

    Y = copy(ps.Y)
    r = ps.r
    c = ps.c
    mc = ps.mc

    shape = samp.shape
    scale = samp.scale
    rng = samp.rng

    X = get_X1(qc)

    XtX = X'X
    UtX = sc.UtX[:, qc.i1_set]
    U = sc.U
    S = sc.S
    V = sc.V
   
    l = size(X, 2)
    m = size(V, 2)
    K = [X V]
    
    #
    sigma = draw_Ysigma!(Y, r, c, mc, shape, scale, X, XtX, UtX, U, S, rng)
    bs = draw_betas(Y, r, c, mc, l, m, K, rng)
    beta = zeros(size(pips,1))

    beta[qc.i1_set[1:(l-1)]] .= bs[1:(l-1)]
    icept = bs[l]
    u = bs[(l+1):end]

    return (γ=γ, pips=pips, icept=icept, beta=beta, u=u, Y=Y, sigma=sigma, r=r, c=c, Z=Z)
end

function is_adapted(samp::ProbitBVSSampler)
    return samp.isadapted
end

mutable struct RWMAdaptor
    stage_len::Int
    buffer_len::Int
    buffer_it::Int
    stage::Int
    steps::Int
    n_acc::Int
    n_var::Int
    dt::Float64
    scales::Array{Float64,1}
    sums::Array{Float64,1}
    sums_sq::Array{Float64,1}
end

function RWMAdaptor(n_var::Int, stage_len::Int)
    buffer_len = div(stage_len,2)
    RWMAdaptor(
        stage_len,
        buffer_len, 
        0,
        1, 
        0, 
        0,
        n_var, 
        0.1, 
        ones(Float64, n_var), 
        zeros(Float64, n_var),
        zeros(Float64, n_var))
end

function adapt!(adaptor::RWMAdaptor, state::Array{Float64,1}, acc::Int)
    adaptor.buffer_it += 1
    adaptor.n_acc += acc
    adaptor.steps += 1

    if adaptor.stage == 1 || adaptor.stage ==3
        accr = adaptor.n_acc / (adaptor.buffer_it)
        adaptor.dt = max(adaptor.dt + 1.0*(accr - 0.22)/sqrt(adaptor.steps), 0.001)
    else
        adaptor.sums_sq .+= state .* state
        adaptor.sums .+= state
    end

    if adaptor.buffer_it >= adaptor.buffer_len
        adaptor.n_acc = div(adaptor.n_acc, 2)
        adaptor.buffer_it = div(adaptor.buffer_it, 2)
    end

    if adaptor.steps >= adaptor.stage_len && adaptor.stage < 3
        if adaptor.stage == 2
            n = adaptor.steps
            n2 = n*n
            var = adaptor.sums_sq./n .- (adaptor.sums .* adaptor.sums)./n2
            adaptor.scales .= sqrt.(var) .+ 1e-6
            adaptor.scales ./= maximum(adaptor.scales)
            adaptor.scales .= max.(adaptor.scales, 1e-3)
        end

        adaptor.sums .= 0.0
        adaptor.sums_sq .= 0.0
        adaptor.steps = 0
        adaptor.n_acc = 0
        adaptor.buffer_it = 0
        adaptor.stage += 1
    end
end

function get_scales(adaptor::RWMAdaptor)
    return adaptor.scales
end

function get_dt(adaptor::RWMAdaptor)
    return adaptor.dt
end
