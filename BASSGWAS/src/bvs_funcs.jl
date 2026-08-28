mutable struct wTGSState2{Tn, Tg, Tv, Tr}
    p::Tn #number of covariates
    γ::Tg #inclusion state vector
    dets::Tv #likelihood determinants
    ssq::Tv #likelihood quadratic form
    llrs::Tv #log likelihood ratio
    pips::Tv #gamma PIPs 
    conds::Tv #gamma conditionals
    rates::Tv #move rates
    rsums::Tv #cummulative rate sums
    ll_curr::Tr #Current likelihood
    xi::Tr #Auxilliary move weight
    eps::Tr #wTGS weight
end

function flip_gamma!(state, ps, qcache, to_flip)
    newg = 1-state.γ[to_flip]
    state.γ[to_flip] = newg
    idx0 = findall(state.γ .== 0)
    idx1 = findall(state.γ .== 1)
    paug = state.p+1
    idx1_icept = [idx1; paug]
    if newg == 1
        add_active_QC(qcache, ps, to_flip, idx1_icept, idx0)
    else
        remove_active_QC(qcache, ps, to_flip, idx1_icept, idx0)
    end 
end

function init_state2(loglik::Aloglik, ps, scache, γ0::Array{Int64, 1}; eps = 5.0, xi_init = 1.0)
    p = length(γ0)
    alpha = ps.alpha
    beta = ps.beta
    c = ps.c
    s = ps.s
    mc = ps.mc

    idx0 = findall(γ0 .== 0)
    idx1 = findall(γ0 .== 1)
    γ = copy(γ0)

    T = eltype(scache.F)

    ssq = zeros(T, p)
    dets = zeros(T, p)
    pips = zeros(T, p)
    conds = zeros(T, p)
    llrs = zeros(T, p)
    rates = zeros(T, p+1)
    rsums = zeros(T, p+1)

    paug = scache.paug
    idx1_icept = [idx1; paug]
    qcache = QCache(ps, idx1_icept, idx0)

    det_curr, ssq_curr = comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)
    ll_curr = loglik(det_curr, ssq_curr)

    comp_llrs!(llrs, dets, ssq, ll_curr, idx1, idx0, p, loglik, alpha, beta)
    comp_rates2!(view(rates, 1:p), conds, pips, llrs, idx1, idx0, eps)
    
    rates[p+1] = xi_init
    cumsum!(rsums, rates) 
 
    state = wTGSState2(p, γ, dets, ssq, llrs, pips, conds, rates, rsums, ll_curr, xi_init, eps)
    return state, qcache
end

function update_state2!(state, ps, scache, qcache, loglik::Aloglik)

    @unpack p, γ, dets, ssq, llrs, pips, conds, rates, rsums, xi, eps = state

    alpha = ps.alpha
    beta = ps.beta
    c = ps.c
    s = ps.s
    mc = ps.mc

    idx0 = findall(γ .== 0)
    idx1 = findall(γ .== 1)

    det_curr, ssq_curr = comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)
    state.ll_curr = loglik(det_curr, ssq_curr)
    comp_llrs!(llrs, dets, ssq, state.ll_curr, idx1, idx0, p, loglik, alpha, beta)
    comp_rates2!(view(rates, 1:p), conds, pips, llrs, idx1, idx0, eps)
    
    rates[p+1] = xi
    _ = cumsum!(rsums, rates)
end

function sample_next2(state, rng)
    rsums = state.rsums

    rtot = rsums[end]
    r = rand(rng) * rtot 
    move_idx = findfirst(x -> x >= r, rsums)
    
    return move_idx
end

function comp_ll2(scache, qcache, ps, γ, loglik::Aloglik)
    s = ps.s
    c = ps.c
    mc = ps.mc

    sinv = 1.0/s
    paug = scache.paug

    i1 = [findall(γ .== 1); paug]
    
    cdet = scache.cdet
    qterm = scache.qterm
    d = length(i1)

    QX1 = scache.QX[:, i1]
    FX1 = scache.FX[:, i1]
    X1 = get_X1(qcache)

    A = sinv * X1'*X1 - QX1'QX1;
    A += UniformScaling(1.0/c)
    A[end, end] += (-1.0/c + 1.0/mc)

    AChol = cholesky(Hermitian(A))
    #Ainv = inv(AChol)
    ldiv!(AChol.U', FX1')

    #ssq_curr = -(FX1 * Ainv * FX1') + qterm
    ssq_curr = -dot(FX1, FX1) + qterm
    det_curr = logabsdet(AChol)[1]
    det_curr += cdet + (d-1)*log(c) + log(mc)

    loglik(det_curr, ssq_curr)
end

function comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)
    
    paug = scache.paug
    idx1_icept = [idx1; paug]
    sinv = 1.0/s
    d = length(idx1_icept)

    QX1 = scache.QX[:, idx1_icept]
    QX0 = _copy_excl(scache.QX, idx1_icept) 

    X1 = get_X1(qcache)
    X1tX0 = get_XtX(qcache)

    X1CinvX1 = sinv * X1'*X1
    X1CinvX1 .-= QX1'QX1

    A = X1CinvX1 + UniformScaling(1.0/c)
    A[end, end] += (-1.0/c + 1.0/mc)

    #A = X1CinvX1 + Diagonal((1.0/c)*ones(length(idx1_icept)))
    AChol = cholesky(Hermitian(A))
    Ainv = inv(AChol)

    FX1 = scache.FX[:,idx1_icept]
    FX0 = scache.FX[:,idx0]
    sqX0 = scache.sqX[idx0]

    qterm = scache.qterm
    ssq_curr = -(FX1 * Ainv * FX1') + qterm
    det_curr = logabsdet(AChol)[1]
    det_curr += scache.cdet + (d-1)*log(c) + log(mc)

    _inc_up2!(view(ssq, idx0), view(dets, idx0),
        c, AChol, QX1, QX0, FX1, FX0, X1tX0, sqX0, sinv)

    _inc_down2!(view(ssq, idx1), view(dets, idx1), 
        c, Ainv, X1CinvX1, FX1)

    ssq .+= qterm# + ssq_curr
    dets .+= det_curr

    return det_curr, ssq_curr
end


function _inc_up2!(ssq, dets, c, AChol, QX1, QX0, FX1, FX0, X1tX0, sqX0, sinv)
    ## Determinant Updates
    # x_i' Cinv x_i forall i

    n = length(ssq)
    d = size(FX1, 2) + 1

    U = AChol.U' \ FX1' # p x 1
    quad_term = -dot(U,U) 
    W = copy(FX0')

    AinvBs = copy(X1tX0)
    
    cinv = 1.0 / c
    a = cinv .+ sqX0 .* sinv

    nt = Threads.nthreads()
    blck_sz = ceil(Int64,n/nt)

    @batch for i in 1:blck_sz:n
        j = min(i+blck_sz-1,n)
        sp = i:j
        _a = view(a, sp)
        _W = view(W, sp)
        _AinvBs = view(AinvBs, :, sp)
        _QX0 = view(QX0, :, sp)
        _up_blck2!(_a, _W, _QX0, _AinvBs, U, QX1, AChol, sinv, c, quad_term, d)
    end

    #a .*= -1.0
    #a .+= (1.0/c) .+ sinv*sqX0 
    #W ./= a
    #W .+= quad_term

    #map!((x) -> abs(x), a, a)
    #map!((x) -> log(x), a, a)
    #a .+= p*log(c)
    dets .= a
    ssq .= W
end

function _up_blck2!(a, W, QX0, AinvBs, U, QX1, AChol, sinv, c, qterm, d)
    ## Determinant Updates
    # x_i' Cinv x_i forall i
    @turbo for j in axes(QX0,2)
        s = 0.0 
        for i in axes(QX0,1)
            s += QX0[i,j]*QX0[i,j]
        end
        a[j] -= s
    end

    gemm!('T', 'N', -1.0, QX1, QX0, sinv, AinvBs)
    ldiv!(AChol.U', AinvBs)

    @turbo for j in axes(AinvBs,2)
        s = 0.0 
        for i in axes(AinvBs,1)
            x = AinvBs[i,j]
            s += x*x
        end
        a[j] -= s
    end

    gemv!('T', 1.0, AinvBs, U, -1.0, W)

    @turbo for i in eachindex(a)
        w=W[i]
        W[i] = -w*w/a[i] + qterm
    end
    
    lc = log(c)
    @turbo for i in eachindex(a)
        x = log(abs(a[i]))
        a[i] = x + lc
    end

    nothing
end

function _inc_down2!(ssq, dets, c, Ainv, X1tCinvX1, FX1)
    p = size(ssq)[1]
    paug = size(Ainv)[2]

    if p > 0 
        nt = Threads.nthreads()
        blck_sz = ceil(Int64,p/nt)
        padj = paug-1
        chunks = collect(Iterators.partition(1:p, blck_sz))

        FX = FX1'
        A0s = [similar(Ainv, padj, padj) for _ in chunks]
        bs = [similar(Ainv, padj) for _ in chunks]
        ys = [similar(Ainv, padj) for _ in chunks]
        @batch for s in eachindex(chunks)
            chunk = chunks[s]
            for i in chunk
                #icm = 1:paug .!= i
                A0 = A0s[s]
                b = bs[s]
                y = ys[s]
                _copyto_loo!(A0, Ainv, i)
                @views _copyto_loo!(b, Ainv[:, i], i)
                dhat = Ainv[i, i]
                syr!('U', -1.0/dhat, b, A0)
                
                _copyto_loo!(b, FX, i)
                symv!('U', 1.0, A0, b, 0.0, y)

                ssq[i] = -dot(b,y)

                @views _copyto_loo!(b, X1tCinvX1[:, i], i)
                symv!('U', 1.0, A0, b, 0.0, y)

                d = X1tCinvX1[i, i]
                det_upd = 1.0/c + d - dot(b,y)
                dets[i] = -log(c)-log(det_upd)
            end
        end   
    end 
    nothing
end

function comp_rates2!(rates, conds, pips, llrs, idx1, idx0, eps)
    p = length(conds)
    w = eps/p

    for i in idx1
        llr = llrs[i]
        cprob = logistic(-llr)
        pip = cprob

        conds[i] = cprob
        pips[i] = pip
        rates[i] = 0.5*(pip+w)/cprob
    end

    for i in idx0
        llr = llrs[i]
        cprob = logistic(-llr)
        pip = logistic(llr)

        conds[i] = cprob
        pips[i] = pip
        rates[i] = 0.5*(pip+w)/cprob
    end
    nothing
end

function comp_llrs!(llrs, dets, ssq, ll_curr, idx1, idx0, p, loglik, alpha, beta)

    γ = length(idx1)
    prior_curr = _inc_prior(γ, p, alpha, beta)
    #inclusion prior ratios
    lpr_incl = _inc_prior(γ+1, p, alpha, beta) - prior_curr
    lpr_excl = γ > 0 ? _inc_prior(γ-1, p, alpha, beta) - prior_curr : -Inf64
    
    for i in axes(llrs, 1)
        ll = loglik(dets[i], ssq[i])
        llrs[i] = ll - ll_curr
    end

    #current ll 
    #ll_prev = ll_curr + prior_curr
    for i in idx1
        llrs[i] += lpr_excl
    end

    for i in idx0
        llrs[i] += lpr_incl
    end
    nothing
end

function _inc_prior(γ, p, alpha, beta)
    alpha_N = γ + alpha
    beta_N = p - γ + beta
    return logbeta(alpha_N, beta_N) - logbeta(alpha, beta)
    #q = alpha / (alpha+beta)
    #return log(q)*γ + log(1.0-q)*(p-γ)
end


function _update_state_test!(state, ps, scache, qcache, loglik::Aloglik; eps=5.0)

    @unpack p, γ, dets, ssq, llrs, pips, conds, rates, rsums = state

    n = ps.n 
    alpha = ps.alpha
    beta = ps.beta
    c = ps.c
    s = ps.s
    r = ps.r
    mc = ps.mc

    paug = scache.paug

    idx0 = findall(γ .== 0)
    idx1 = findall(γ .== 1)

    C = s*Diagonal(ones(n)) + r*scache.U*Diagonal(scache.S)*scache.U'
    X = ps.Xaug
    Y = ps.Y
    
    idx1_icept = [idx1; paug]
    Xγ = X[:, idx1_icept]
    det_curr, ssq_curr = _direct(C, c, mc, Xγ, Y)
    _dets, _ssq = _direct_comp(C, X, Y, c, mc, idx1, idx0, paug)
    state.ll_curr = loglik(det_curr, ssq_curr)
    dets .= _dets
    ssq .= _ssq

    comp_llrs!(llrs, dets, ssq, state.ll_curr, idx1, idx0, p, loglik, alpha, beta)
    comp_rates2!(view(rates, 1:p), conds, pips, llrs, idx1, idx0, eps)
    
    rates[p+1] = state.xi
    _ = cumsum!(rsums, rates)
    return(logabsdet(C))
end


function _direct_comp(C, X, Y, c, mc, idx1, idx0, paug)
    dets = ones(paug-1)
    ssq = ones(paug-1)

    idx1_icept = [idx1; paug]
    Xγ = X[:, idx1_icept]

    for i in idx0
        Xi =  cat(X[:, i], Xγ, dims=2)
        dets[i], ssq[i] = _direct(C, c, mc, Xi, Y)
    end

    for i in eachindex(idx1)
        j = idx1[i]
        Xi = Xγ[:, 1:end .!= i]
        dets[j], ssq[j] = _direct(C, c, mc, Xi, Y)
    end
    return dets, ssq
end

function _direct(C, c, mc, X, Y)
    n = size(X, 2)
    diagshift = c*ones(n)
    diagshift[end] = mc
    Q = cholesky(Hermitian(C + X*Diagonal(diagshift)*X'))
    det = logabsdet(Q)[1]
    qform = Y'*(Q\Y)
    return det, qform
end

function _ssq_direct(Y1, Y2, r, c, mc, X, U, S)
    m = size(X, 1)
    n = size(X, 2)
    s = 1.0
    C = s*Diagonal(ones(m)) + r*U*Diagonal(S)*U'
    
    diagshift = c*ones(n)
    diagshift[end] = mc

    Q= cholesky(Hermitian(C + X*Diagonal(diagshift)*X'))
    qform = Y1'*(Q\Y2)
    return qform
end

function _det_direct(r, c, mc, X, U, S)
    m = size(X, 1)
    n = size(X, 2)
    s = 1.0
    C = s*Diagonal(ones(m)) + r*U*Diagonal(S)*U'
    
    diagshift = c*ones(n)
    diagshift[end] = mc

    Q= cholesky(Hermitian(C + X*Diagonal(diagshift)*X'))
    return logabsdet(Q)[1]
end

#Calculation from chipman george mcculloch 2012, pg80
function _direct_ll(Y, r, c, mc, X, U, S, nu, lambda)

    V = U*Diagonal(sqrt.(S))
    K = [X V]

    n = size(X, 1)
    l = size(X, 2)
    m = size(V, 2)

    sigmasq = zeros(l+m)
    sigmasq[1:(l-1)] .= 1.0/c
    sigmasq[l] = 1.0/mc
    sigmasq[(l+1):(m+l)] .= 1.0/r

    Sd = Diagonal(sigmasq)
    Qinv = K'K + Sd
    Qf = cholesky(Qinv)

    ssq = Y'Y - Y'K*(Qf\(K'Y))
    det1 = 0.5*sum(log.(abs.(sigmasq)))
    det2 = -sum(log.(abs.(diag(Qf.U))))

    return det1+det2-0.5*(n+nu)*log(nu*lambda + ssq)
end

function _comp_ssq(Y1, Y2, r, c, mc, X, XtX, UtX, U, S)

    d = similar(S)

    for i in axes(d,1)
        d[i] = 1.0/sqrt(1.0/(S[i]*r) + 1.0)
    end

    D = Diagonal(d)

    Fl1 = U' * Y1
    Fl2 = U' * Y2
    lmul!(D, Fl1)
    lmul!(D, Fl2)

    DUtX = D*UtX

    qterm = dot(Y1, Y2) - dot(Fl1, Fl2)
    FX1 = -Fl1'*DUtX + (Y1'*X)
    FX2 = -Fl2'*DUtX + (Y2'*X)

    A = XtX - DUtX' * DUtX
    A += UniformScaling(1.0/c)
    A[end, end] += (-1.0/c + 1.0/mc)

    cholesky!(Hermitian(A))
    Atr = LowerTriangular(A')
    ldiv!(Atr, FX1')
    ldiv!(Atr, FX2')

    ssq = -dot(FX1, FX2) + qterm
    return ssq
end

function _comp_ssq(Y, r, c, mc, X, XtX, UtX, U, S)

    d = similar(S)

    for i in axes(d,1)
        d[i] = 1.0/sqrt(1.0/(S[i]*r) + 1.0)
    end

    D = Diagonal(d)

    Fl = U' * Y
    lmul!(D, Fl)
    DUtX = D*UtX

    qterm = dot(Y, Y) - dot(Fl, Fl)
    FX = -Fl'*DUtX + (Y'*X)

    A = XtX - DUtX' * DUtX
    A += UniformScaling(1.0/c)
    A[end, end] += (-1.0/c + 1.0/mc)

    cholesky!(Hermitian(A))
    Atr = LowerTriangular(A')
    ldiv!(Atr, FX')

    ssq = -dot(FX, FX) + qterm
    return ssq
end

function _comp_llterms(Y, r, c, mc, X, XtX, UtX, YtU, S)
    d = similar(S)

    n = size(d, 1)
    k = size(X, 2)

    for i in axes(d,1)
        d[i] = 1.0/sqrt(1.0/(S[i]*r) + 1.0)
    end

    D = Diagonal(d)
    Fl = D*YtU'
    D2 = D*D
    B = D2 * UtX 
    FX = -YtU*B + (Y'*X)

    qterm = dot(Y, Y) - dot(Fl, Fl)

    A = XtX - UtX' * B
    A += UniformScaling(1.0/c)
    A[end, end] += (-1.0/c + 1.0/mc)

    cholesky!(Hermitian(A))
    Atr = LowerTriangular(A')
    ldiv!(Atr, FX')

    ssq = -dot(FX, FX) + qterm
    
    cdet = -2.0*(logabsdet(D)[1])+ n*log(r)
    for i in eachindex(S)
         cdet += log(abs(S[i]))
    end

    det_curr = 2.0*logabsdet(Atr)[1]
    det_curr += cdet + (k-1)*log(c) + log(mc)

    return ssq, det_curr
end