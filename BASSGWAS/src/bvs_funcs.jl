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

"""
    wTGSState2

Mutable weighted tempered Gibbs sampling state for Bayesian variable
selection.

The state stores the current binary inclusion vector and cached posterior
quantities for every model reached by toggling one predictor. The intercept is
handled separately and is not represented in `γ`.
Numerical arrays and scalars share the concrete floating-point type `T`;
`p` and the binary inclusion vector use `Int`.

# Fields
- `p`: Number of selectable predictors.
- `γ`: Current binary predictor-inclusion vector.
- `dets`, `ssq`: Collapsed likelihood terms for one-toggle alternatives.
- `llrs`: Posterior log ratio of each toggled model to the current model.
- `pips`: Conditional predictor-inclusion probabilities.
- `conds`: Conditional probability of each predictor's current state.
- `rates`, `rsums`: Move intensities and their cumulative sums; the final
  entry represents the auxiliary continuous-state update.
- `ll_curr`: Collapsed log likelihood of the current model.
- `xi`: Auxiliary-update intensity.
- `eps`: Weighted-TGS regularization weight.
"""
mutable struct wTGSState2{T<:AbstractFloat}
    p::Int
    γ::Vector{Int}
    dets::Vector{T}
    ssq::Vector{T}
    llrs::Vector{T}
    pips::Vector{T}
    conds::Vector{T}
    rates::Vector{T}
    rsums::Vector{T}
    ll_curr::T
    xi::T
    eps::T

    function wTGSState2{T}(p::Integer, γ::AbstractVector{<:Integer},
                           dets, ssq, llrs, pips, conds, rates, rsums,
                           ll_curr, xi, eps) where {T<:AbstractFloat}
        isconcretetype(T) || throw(ArgumentError("state element type must be concrete"))
        p >= 0 && length(γ) == p &&
            all(x -> x isa AbstractVector && length(x) == p,(dets,ssq,llrs,pips,conds)) &&
            rates isa AbstractVector && rsums isa AbstractVector &&
            length(rates) == p+1 && length(rsums) == p+1 ||
            throw(DimensionMismatch("state arrays must match predictor and move counts"))
        all(x -> x == 0 || x == 1,γ) || throw(ArgumentError("inclusion values must be zero or one"))
        xi, eps = T(xi), T(eps)
        isfinite(xi) && xi >= 0 || throw(ArgumentError("xi must be finite and nonnegative"))
        isfinite(eps) && eps >= 0 || throw(ArgumentError("eps must be finite and nonnegative"))
        return new{T}(p,γ,dets,ssq,llrs,pips,conds,rates,rsums,ll_curr,xi,eps)
    end
end

function wTGSState2(p::Integer, γ::AbstractVector{<:Integer}, dets, ssq, llrs,
                    pips, conds, rates, rsums, ll_curr, xi, eps)
    T = _state_float_type(map(eltype,(dets,ssq,llrs,pips,conds,rates,rsums))...,
                         typeof(ll_curr),typeof(xi),typeof(eps))
    return wTGSState2{T}(p,γ,dets,ssq,llrs,pips,conds,rates,rsums,ll_curr,xi,eps)
end

"""
    flip_gamma!(state, ps, qcache, to_flip)

Toggle predictor `to_flip` in `state` and synchronize `qcache` with the new
active model.

Indices are one-based predictor-column indices. The intercept remains the
final active column in the augmented design. Both `state` and `qcache` are
mutated.
"""
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

"""
    init_state2(loglik, ps, scache, γ0; eps=5.0, xi_init=1.0)

Initialize a weighted-TGS model state and its predictor-toggle cache.

`γ0` is an integer or Boolean binary vector with one entry per predictor and
is copied into `Vector{Int}`. Numerical storage uses the parameter state's
floating-point type, which must match the cache and likelihood. The method evaluates the
current collapsed likelihood, every one-predictor alternative, and the initial
move intensities. It returns `(state, qcache)`.
"""
function init_state2(loglik::Aloglik{T}, ps::ParState2{T}, scache::SCache{D,T},
                     γ0::AbstractVector{<:Integer}; eps = 5.0, xi_init = 1.0) where {D,T}
    p = length(γ0)
    p == ps.p || throw(DimensionMismatch("inclusion vector must match predictor count"))
    all(x -> x == 0 || x == 1,γ0) || throw(ArgumentError("inclusion values must be zero or one"))
    eps, xi_init = T(eps), T(xi_init)
    isfinite(eps) && eps >= 0 || throw(ArgumentError("eps must be finite and nonnegative"))
    isfinite(xi_init) && xi_init >= 0 || throw(ArgumentError("xi_init must be finite and nonnegative"))
    alpha = ps.alpha
    beta = ps.beta
    c = ps.c
    s = ps.s
    mc = ps.mc

    idx0 = findall(γ0 .== 0)
    idx1 = findall(γ0 .== 1)
    γ = Vector{Int}(γ0)

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

    # The final rate is reserved for an auxiliary, non-predictor update.
    rates[p+1] = xi_init
    cumsum!(rsums, rates)

    state = wTGSState2{T}(p, γ, dets, ssq, llrs, pips, conds, rates, rsums, ll_curr, xi_init, eps)
    return state, qcache
end

"""
    update_state2!(state, ps, scache, qcache, loglik)

Recompute the current collapsed likelihood, one-toggle alternatives, and move
intensities after model or continuous-parameter changes.

Cached arrays in `state` and the active numerical workspace in `qcache` are updated in place.
"""
function update_state2!(state::wTGSState2{T}, ps::ParState2{T}, scache::SCache{D,T},
                        qcache::QCache{T}, loglik::Aloglik{T}) where {D,T}

    @unpack p, γ, dets, ssq, llrs, pips, conds, rates, rsums, xi, eps = state

    alpha = ps.alpha
    beta = ps.beta
    c = ps.c
    s = ps.s
    mc = ps.mc

    idx0 = qcache.i0_set
    idx1 = view(qcache.i1_set,1:(qcache.d-1))

    det_curr, ssq_curr = comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)
    state.ll_curr = loglik(det_curr, ssq_curr)
    comp_llrs!(llrs, dets, ssq, state.ll_curr, idx1, idx0, p, loglik, alpha, beta)
    comp_rates2!(view(rates, 1:p), conds, pips, llrs, idx1, idx0, eps)

    rates[p+1] = xi
    _ = cumsum!(rsums, rates)
end

"""
    sample_next2(state, rng)

Draw a one-based move index in proportion to the cumulative intensities in
`state.rsums`.

Indices `1:state.p` select predictor toggles, while `state.p + 1` selects the
auxiliary continuous-state update.
"""
function sample_next2(state, rng)
    rsums = state.rsums

    rtot = rsums[end]
    r = rand(rng) * rtot
    move_idx = findfirst(x -> x >= r, rsums)

    return move_idx
end

"""
    comp_ll2(scache, qcache, ps, γ, loglik)

Evaluate the collapsed log likelihood of inclusion state `γ` from the
low-rank covariance and active-design caches.

The intercept is always included with variance `ps.mc`; active predictor
effects use variance `ps.c`.
"""
function comp_ll2(scache::SCache{D,T}, qcache::QCache{T}, ps::ParState2{T},
                  γ, loglik::Aloglik{T}) where {D,T}
    ps.s == scache.covariance.s || throw(ArgumentError("refresh SCache after changing s"))
    work = prepare_active!(qcache,scache,ps.c,ps.mc)
    return loglik(work.det,work.ssq)
end

"""
    comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)

Evaluate all one-predictor alternatives with cached covariance-adjusted
products and a common active Cholesky. Candidate solves use bounded reusable
blocks; removal terms use the inverse diagonal and the shared response solve.
"""
function comp_upd_terms2!(dets, ssq, scache, qcache, c, s, mc, idx1, idx0)
    s == scache.covariance.s || throw(ArgumentError("refresh SCache after changing s"))
    work = prepare_active!(qcache,scache,c,mc)
    if !isempty(idx0)
        candidate_terms!(dets,ssq,qcache,scache,c)
    end
    lc = log(c)
    @inbounds for i in 1:(qcache.d-1)
        index = qcache.i1_set[i]
        invdiag = work.invdiag
        ssq[index] = work.ssq+work.beta[i]^2/invdiag[i]
        dets[index] = work.det+log(invdiag[i])-lc
    end
    return work.det,work.ssq
end


"""
    comp_rates2!(rates, conds, pips, llrs, idx1, idx0, eps)

Compute weighted-TGS predictor move intensities from one-toggle log posterior
ratios.

`conds` receives the conditional probability of each predictor's current
state, `pips` receives conditional inclusion probabilities, and `rates`
receives the regularized move intensities. All three arrays are mutated.
"""
function comp_rates2!(rates, conds, pips, llrs, idx1, idx0, eps)
    p = length(conds)
    p == 0 && return nothing
    w = eps/p

    nt = Threads.nthreads()
    blck_sz = ceil(Int64,p/nt)

    @batch for i in 1:blck_sz:p
        j = min(i+blck_sz-1,p)
        sp = i:j
        @turbo for k in sp
            llr = llrs[k]
            cprob = logistic(-llr)
            pip = logistic(llr)

            conds[k] = cprob
            pips[k] = pip
            rates[k] = 0.5*(pip+w)/cprob
        end
    end

    for i in idx1
        llr = llrs[i]
        cprob = logistic(-llr)
        pip = cprob

        conds[i] = cprob
        pips[i] = pip
        rates[i] = 0.5*(pip+w)/cprob
    end

    nothing
end

"""
    comp_llrs!(llrs, dets, ssq, ll_curr, idx1, idx0, p, loglik, alpha, beta)

Fill `llrs` with each one-toggle model's log posterior ratio relative to the
current model.

The calculation combines collapsed likelihood differences with the
Beta-Binomial prior ratio implied by `alpha` and `beta`.
"""
function comp_llrs!(llrs::AbstractVector{T}, dets::AbstractVector{T}, ssq::AbstractVector{T},
                    ll_curr::T, idx1, idx0, p, loglik::Aloglik{T}, alpha, beta) where {T}
    p == 0 && return nothing
    γ = length(idx1)
    prior_curr = _inc_prior(γ, p, alpha, beta)
    # Compute prior changes for adding or removing one included predictor.
    lpr_incl = γ < p ? T(_inc_prior(γ+1, p, alpha, beta) - prior_curr) : zero(T)
    lpr_excl = γ > 0 ? T(_inc_prior(γ-1, p, alpha, beta) - prior_curr) : T(-Inf)

    n=size(llrs,1)
    nt = Threads.nthreads()
    blck_sz = ceil(Int64,n/nt)
    @batch for i in 1:blck_sz:n
        j = min(i+blck_sz-1,n)
        sp = i:j
        @views ll_plus_a!(llrs[sp], dets[sp], ssq[sp], lpr_incl-ll_curr, loglik)
    end

    for i in idx1
        llrs[i] += lpr_excl-lpr_incl
    end

    nothing
end

"""
    _inc_prior(γ, p, alpha, beta)

Return the integrated Beta-Bernoulli log prior, up to a model-independent
constant, for a particular inclusion vector containing `γ` active entries
among `p` predictors.
"""
function _inc_prior(γ, p, alpha, beta)
    alpha_N = γ + alpha
    beta_N = p - γ + beta
    return logbeta(alpha_N, beta_N) - logbeta(alpha, beta)
end


"""
    _update_state_test!(state, ps, scache, qcache, loglik; eps=5.0)

Recompute a weighted-TGS state with explicit dense covariance calculations.

This test-only reference implementation mutates `state` and is used to verify
the cached Woodbury and rank-one update path. It returns `logabsdet(C)` for the
explicit covariance matrix.
"""
function _update_state_test!(state::wTGSState2{T}, ps::ParState2{T}, scache::SCache{D,T},
                             qcache::QCache{T}, loglik::Aloglik{T}; eps=5.0) where {D,T}

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


"""
    _direct_comp(C, X, Y, c, mc, idx1, idx0, paug)

Compute explicit collapsed determinant and quadratic-form terms for every
one-predictor alternative to the active model.

This dense reference routine returns `(dets, ssq)` in full predictor order.
"""
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

"""
    _direct(C, c, mc, X, Y)

Compute explicit collapsed log-determinant and quadratic-form terms for design
matrix `X`, response `Y`, and covariance `C`.

All columns except the final intercept use prior variance `c`; the intercept
uses `mc`. The function returns `(det, qform)`.
"""
function _direct(C, c, mc, X, Y)
    n = size(X, 2)
    diagshift = c*ones(n)
    diagshift[end] = mc
    Q = cholesky(Hermitian(C + X*Diagonal(diagshift)*X'))
    det = logabsdet(Q)[1]
    qform = Y'*(Q\Y)
    return det, qform
end

"""
    _ssq_direct(Y1, Y2, r, c, mc, X, U, S)

Evaluate the covariance-adjusted bilinear form between `Y1` and `Y2` by
explicitly factorizing the full covariance.

This reference implementation fixes the residual variance multiplier at one.
"""
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

"""
    _det_direct(r, c, mc, X, U, S)

Evaluate the collapsed model log determinant by explicitly factorizing the
full covariance, with residual variance multiplier fixed at one.
"""
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

"""
    _direct_ll(Y, r, c, mc, X, U, S, nu, lambda)

Evaluate the collapsed Student-t log likelihood through an explicit Gaussian
coefficient precision factorization.

The calculation is retained as a dense reference for the cached likelihood
implementation and fixes the residual variance multiplier at one.
"""
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

"""
    _comp_ssq(Y1, Y2, r, c, mc, X, XtX, UtX, U, S)

Evaluate the covariance-adjusted bilinear form between responses `Y1` and
`Y2` using a low-rank Woodbury representation.

`XtX` and `UtX` must correspond to the supplied active design `X`. Residual
variance is fixed at one.
"""
function _comp_ssq(Y1, Y2, r, c, mc, X, XtX, UtX, U, S)

    d = similar(S)

    _comp_d!(d, S, r, one(r))

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

"""
    _comp_ssq(Y, r, c, mc, X, XtX, UtX, U, S)

Evaluate the covariance-adjusted quadratic form of `Y` using a low-rank
Woodbury representation.

This specialization avoids duplicated work when both responses in the
bilinear form are identical. Residual variance is fixed at one.
"""
function _comp_ssq(Y, r, c, mc, X, XtX, UtX, U, S)

    d = similar(S)

    _comp_d!(d, S, r, one(r))

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

"""
    _comp_llterms(Y, r, c, mc, X, XtX, UtX, YtU, S)

Compute the collapsed response quadratic form and log determinant for proposed
random-effect and predictor variances.

The method reuses active-design and low-rank cross-products and returns
`(ssq, det)`. Residual variance is fixed at one.
"""
function _comp_llterms(Y, r, c, mc, X, XtX, UtX, YtU, S)
    d = similar(S)

    k = size(X, 2)

    _comp_d!(d, S, r, one(r))

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

    # Stand-alone compatibility path; the sampler uses proposal_terms! and
    # retains all of this workspace across proposals.
    cdet = _covariance_weights!(d, similar(d), similar(d), S, r, one(r), size(X,1))

    det_curr = 2.0*logabsdet(Atr)[1]
    det_curr += cdet + (k-1)*log(c) + log(mc)

    return ssq, det_curr
end
