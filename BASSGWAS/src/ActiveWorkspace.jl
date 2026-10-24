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
    ProposalWorkspace

Scratch for proposals and conditional response forms; never aliases accepted
cache arrays. Let `a` be the current active-column count including the
intercept. `allocated_model_size` is the number of active columns storage can
hold, including the intercept. Only the leading `a` entries or corresponding
matrix portions are used for active quantities.

# Fields

- `gram`: `allocated_model_size × allocated_model_size` buffer holding `X_A' * X_A` in its leading
  `a × a` portion, gathered once per active-model generation.
- `utx`: `ns × allocated_model_size` buffer holding `U' * X_A` in its first `a` columns.
- `weighted_design`: `ns × allocated_model_size` workspace for `Diagonal(d) * utx`.
- `factor`: `allocated_model_size × allocated_model_size` buffer whose leading lower triangle stores
  the Cholesky factor of the proposed active precision
  `A = X_A' * inv(C) * X_A + Diagonal([1/c, ..., 1/mc])`.
- `d`: Length-`ns` square-root Woodbury weights for the prepared proposal.
- `d2`: Length-`ns` squared Woodbury weights for the prepared proposal.
- `invden`: Length-`ns` vector with entries `1 / (s + r*S[i])`.
- `weighted_y`: Length-`ns` scratch for `d2 .* (U' * Y)` or the corresponding
  intercept projection during shift calculations.
- `fx`: Length-`allocated_model_size` response-form workspace: first holds
  `X_A' * inv(C) * Y`, then its solve through the lower Cholesky factor.
- `ones_fx`: Length-`allocated_model_size` workspace for the analogous intercept form
  `X_A' * inv(C) * ones(n)` and its lower-triangular solve.
- `yu`: Length-`ns` projection buffer `U' * Y` for newly supplied responses.
- `residual`: Length-`n` scratch for `Y - U * yu` in response quadratic forms.
- `design_generation`: Active-model generation of the cached `gram` and
  `utx`; `typemax(UInt)` denotes invalid or uninitialized storage.
- `factor_generation`: Active-model generation of the prepared factor;
  validity also requires matching `r`, `s`, `c` and `mc`. `typemax(UInt)`
  denotes an invalid or uninitialized factor.
- `r`: Random-effect variance multiplier of the prepared factor.
- `s`: Residual variance multiplier of the prepared factor.
- `c`: Selectable-predictor prior variance of the prepared factor.
- `mc`: Intercept prior variance of the prepared factor. Variance fields are
  initialized to `NaN` until a factor is prepared successfully.
- `cdet`: Proposed base-covariance log determinant `logdet(C)`.
- `det`: Collapsed model log determinant, including active predictor and
  intercept prior contributions.
"""
mutable struct ProposalWorkspace{T}
    gram::Matrix{T}
    utx::Matrix{T}
    weighted_design::Matrix{T}
    factor::Matrix{T}
    d::Vector{T}
    d2::Vector{T}
    invden::Vector{T}
    weighted_y::Vector{T}
    fx::Vector{T}
    ones_fx::Vector{T}
    yu::Vector{T}
    residual::Vector{T}
    design_generation::UInt
    factor_generation::UInt
    r::T
    s::T
    c::T
    mc::T
    cdet::T
    det::T
end

function ProposalWorkspace(::Type{T}, n, k, a) where T
    ProposalWorkspace(zeros(T,a,a),zeros(T,k,a),zeros(T,k,a),zeros(T,a,a),
        zeros(T,k),zeros(T,k),zeros(T,k),zeros(T,k),zeros(T,a),zeros(T,a),
        zeros(T,k),zeros(T,n),typemax(UInt),typemax(UInt),T(NaN),T(NaN),T(NaN),T(NaN),zero(T),zero(T))
end

"""
    ActiveWorkspace

Active Cholesky, response solves and bounded, worker-owned candidate scratch.
Let `a` include the active predictors and final intercept. Active matrix and
vector buffers reserve `allocated_model_size` columns or entries; their leading `a`
portions contain the current model's quantities.

# Fields

- `qactive`: `ns × allocated_model_size` gather buffer for active columns of `QX`, used
  when rebuilding all covariance-adjusted active products.
- `factor`: `allocated_model_size × allocated_model_size` buffer whose leading lower triangle stores
  `L`, the Cholesky factor of the accepted active precision `A`.
- `invlower`: `allocated_model_size × allocated_model_size` scratch holding `inv(L)` in its leading
  `a × a` portion to obtain the inverse-precision diagonal.
- `u`: Length-`allocated_model_size` buffer whose first `a` entries hold
  `L \\ (X_A' * inv(C) * Y)`.
- `beta`: Length-`allocated_model_size` buffer whose first `a` entries hold
  `inv(A) * X_A' * inv(C) * Y`.
- `invdiag`: Length-`allocated_model_size` buffer holding `diag(inv(A))` in its first `a`
  entries for predictor-removal updates.
- `blocks`: One `allocated_model_size × CACHE_BLOCK_COLUMNS` matrix per worker for
  candidate cross-products and their in-place lower-triangular solves.
- `invalid_candidates`: One failure slot per worker: zero on success, or the
  index of a candidate with a nonpositive Schur complement. Failures are
  raised on the calling task after workers finish.
- `factor_key`: Tuple `(model_generation, covariance_generation, c, mc)`
  identifying the accepted factor. Invalid keys use `typemax(UInt)` counters
  and `NaN` variances.
- `response_generation`: Response generation used by `u`, `beta` and `ssq`;
  `typemax(UInt)` marks invalid or uninitialized response solves.
- `det`: Accepted collapsed model log determinant, including covariance and
  coefficient-prior contributions.
- `ssq`: Accepted collapsed response quadratic form
  `Y' * inv(C) * Y - dot(u, u)`.
- `proposal`: Independent `ProposalWorkspace` for variance proposals and
  conditional response forms, preserving the accepted cache contents.
"""
mutable struct ActiveWorkspace{T}
    qactive::Matrix{T}
    factor::Matrix{T}
    invlower::Matrix{T}
    u::Vector{T}
    beta::Vector{T}
    invdiag::Vector{T}
    blocks::Vector{Matrix{T}}
    invalid_candidates::Vector{Int}
    factor_key::Tuple{UInt,UInt,T,T}
    response_generation::UInt
    det::T
    ssq::T
    proposal::ProposalWorkspace{T}
end

function ActiveWorkspace(::Type{T}, n, k, a) where T
    ActiveWorkspace(zeros(T,k,a),zeros(T,a,a),zeros(T,a,a),zeros(T,a),zeros(T,a),zeros(T,a),
        [zeros(T,a,CACHE_BLOCK_COLUMNS) for _ in 1:Threads.nthreads()], zeros(Int,Threads.nthreads()),
        (typemax(UInt),typemax(UInt),T(NaN),T(NaN)),typemax(UInt),zero(T),zero(T),
        ProposalWorkspace(T,n,k,a))
end

# Numerical kernels use strided, contiguous-column views of reserved storage.
# Index-vector gathers are copied into those buffers once per model.
function _active_product_block!(qc, sc, lo, hi)
    a = qc.d
    G = view(qc.covdata,1:a,lo:hi)
    copyto!(G, view(qc.xtxdata,1:a,lo:hi))
    if sc.ns > 0
        mul!(G, transpose(view(qc.work.qactive,:,1:a)), view(sc.QX,:,lo:hi),
            -one(sc.covariance.s), sc.covariance.sinv)
    else
        G .*= sc.covariance.sinv
    end
    return nothing
end

"""Synchronize G = X_active' C^-1 X, preserving valid rows across model flips."""
function sync_active_products!(qc, sc)
    cv = sc.covariance
    if qc.cov_source !== cv || qc.cov_generation != cv.generation
        @inbounds for j in 1:qc.d, i in 1:sc.ns
            qc.work.qactive[i,j] = sc.QX[i,qc.i1_set[j]]
        end
        P = sc.paug
        nb = cld(P,CACHE_BLOCK_COLUMNS)
        if Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && sc.ns*qc.d*P >= CACHE_THREAD_WORK
            @batch for b in 1:nb
                lo = (b-1)*CACHE_BLOCK_COLUMNS+1
                _active_product_block!(qc,sc,lo,min(lo+CACHE_BLOCK_COLUMNS-1,P))
            end
        else
            _active_product_block!(qc,sc,1,P)
        end
        fill!(qc.validrows,true)
        qc.cov_generation = cv.generation
        qc.cov_source = cv
        qc.work.factor_key = (typemax(UInt),typemax(UInt),typeof(cv.r)(NaN),typeof(cv.r)(NaN))
    else
        for i in 1:qc.d
            if !qc.validrows[i]
                row = view(qc.covdata,i,:)
                copyto!(row,view(qc.xtxdata,i,:))
                if sc.ns > 0
                    _column_dots!(row,sc.QX,view(sc.QX,:,qc.i1_set[i]),-one(cv.s),cv.sinv)
                else
                    row .*= cv.sinv
                end
                qc.validrows[i] = true
            end
        end
    end
    return nothing
end

"""Prepare accepted-model factorization and solves, keyed by their exact dependencies."""
function prepare_active!(qc, sc, c, mc)
    _positive_variance(c,"c")
    _positive_variance(mc,"mc")
    sync_active_products!(qc,sc)
    w, a = qc.work, qc.d
    key = (qc.generation,sc.covariance.generation,c,mc)
    F = view(w.factor,1:a,1:a)
    if w.factor_key != key
        # A failed factorization must never leave an old key on modified storage.
        w.factor_key = (typemax(UInt),typemax(UInt),typeof(w.det)(NaN),typeof(w.det)(NaN))
        @inbounds for j in 1:a, i in 1:a
            F[i,j] = qc.covdata[i,qc.i1_set[j]]
        end
        for i in 1:a
            F[i,i] += i == a ? inv(mc) : inv(c)
        end
        cholesky!(Symmetric(F,:L))
        w.det = sc.cdet + (a-1)*log(c) + log(mc)
        for i in 1:a
            w.det += 2log(F[i,i])
        end
        H = view(w.invlower,1:a,1:a)
        fill!(H,zero(eltype(H)))
        for i in 1:a
            H[i,i] = one(eltype(H))
        end
        ldiv!(LowerTriangular(F),H)
        @inbounds for j in 1:a
            h = zero(eltype(H))
            @simd for i in 1:a
                h += abs2(H[i,j])
            end
            w.invdiag[j] = h
        end
        w.factor_key = key
        w.response_generation = typemax(UInt)
    end
    if w.response_generation != sc.response.generation
        @inbounds for i in 1:a
            w.u[i] = sc.response.fx[qc.i1_set[i]]
        end
        u = view(w.u,1:a)
        ldiv!(LowerTriangular(F),u)
        w.ssq = sc.qterm - dot(u,u)
        copyto!(view(w.beta,1:a),u)
        ldiv!(UpperTriangular(transpose(F)),view(w.beta,1:a))
        w.response_generation = sc.response.generation
    end
    return w
end

function _candidate_block!(dets,ssq,qc,sc,c,lo,hi,buffer)
    w, a = qc.work, qc.d
    B = view(buffer,1:a,1:(hi-lo+1))
    copyto!(B,view(qc.covdata,1:a,lo:hi))
    ldiv!(LowerTriangular(view(w.factor,1:a,1:a)),B)
    cinv, lc = inv(c), log(c)
    @inbounds for j in axes(B,2)
        normb, uy = zero(c), zero(c)
        @simd for i in 1:a
            x = B[i,j]
            normb += x*x
            uy += x*w.u[i]
        end
        index = lo+j-1
        h = cinv + sc.covariance.gramdiag[index] - normb
        # Propagate failures to the calling task; never throw across a
        # Polyester worker callback.
        h > 0 || return index
        residual = sc.response.fx[index]-uy
        ssq[index] = w.ssq-residual^2/h
        dets[index] = w.det+log(h)+lc
    end
    return 0
end

function candidate_terms!(dets,ssq,qc,sc,c)
    p, w = length(dets), qc.work
    nb = cld(p,CACHE_BLOCK_COLUMNS)
    fill!(w.invalid_candidates,0)
    # Each worker owns one buffer, independent of task migration/thread IDs.
    nt = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && qc.d*p >= CACHE_THREAD_WORK ? min(length(w.blocks),nb) : 1
    if nt > 1
        @batch for worker in 1:nt
            for b in worker:nt:nb
                lo = (b-1)*CACHE_BLOCK_COLUMNS+1
                bad = _candidate_block!(dets,ssq,qc,sc,c,lo,min(lo+CACHE_BLOCK_COLUMNS-1,p),w.blocks[worker])
                if bad != 0
                    w.invalid_candidates[worker] = bad
                    break
                end
            end
        end
    else
        for b in 1:nb
            lo = (b-1)*CACHE_BLOCK_COLUMNS+1
            bad = _candidate_block!(dets,ssq,qc,sc,c,lo,min(lo+CACHE_BLOCK_COLUMNS-1,p),w.blocks[1])
            bad == 0 || throw(PosDefException(bad))
        end
    end
    for bad in w.invalid_candidates
        bad == 0 || throw(PosDefException(bad))
    end
    return nothing
end

function prepare_proposal!(qc,sc,r,s,c,mc)
    _positive_variance(c,"c")
    _positive_variance(mc,"mc")
    w, a = qc.work.proposal, qc.d
    if w.design_generation != qc.generation
        @inbounds for j in 1:a
            index = qc.i1_set[j]
            for i in 1:a
                w.gram[i,j] = qc.xtxdata[i,index]
            end
            for i in 1:sc.ns
                w.utx[i,j] = sc.UtX[i,index]
            end
        end
        w.design_generation = qc.generation
        w.factor_generation = typemax(UInt)
    end
    if w.factor_generation == qc.generation && (r,s,c,mc) == (w.r,w.s,w.c,w.mc)
        return w
    end
    w.factor_generation = typemax(UInt)
    w.cdet = _covariance_weights!(w.d,w.d2,w.invden,sc.S,r,s,sc.n)
    B = view(w.weighted_design,:,1:a)
    @inbounds for j in 1:a
        @simd for i in 1:sc.ns
            B[i,j] = w.d[i]*w.utx[i,j]
        end
    end
    F = view(w.factor,1:a,1:a)
    copyto!(F,view(w.gram,1:a,1:a))
    if sc.ns > 0
        mul!(F,transpose(B),B,-one(s),inv(s))
    else
        F ./= s
    end
    for i in 1:a
        F[i,i] += i == a ? inv(mc) : inv(c)
    end
    cholesky!(Symmetric(F,:L))
    w.det = w.cdet + (a-1)*log(c) + log(mc)
    for i in 1:a
        w.det += 2log(F[i,i])
    end
    w.r,w.s,w.c,w.mc = r,s,c,mc
    w.factor_generation = qc.generation
    return w
end

"""Evaluate a proposal using fixed response projections without changing accepted caches."""
function proposal_terms!(qc,sc,r,c,mc; s=sc.covariance.s)
    w = prepare_proposal!(qc,sc,r,s,c,mc)
    a, rs = qc.d, sc.response
    @inbounds @simd for i in 1:sc.ns
        w.weighted_y[i] = w.d2[i]*rs.yu[i]
    end
    f = view(w.fx,1:a)
    @inbounds for i in 1:a
        f[i] = rs.yx[qc.i1_set[i]]
    end
    if sc.ns > 0
        mul!(f,transpose(view(w.utx,:,1:a)),w.weighted_y,-one(s),inv(s))
    else
        f ./= s
    end
    ldiv!(LowerTriangular(view(w.factor,1:a,1:a)),f)
    q = rs.perp2/s
    for i in 1:sc.ns
        q += w.invden[i]*rs.yu[i]^2
    end
    return q-dot(f,f),w.det
end

"""
Return (1'M^-1 1, 1'M^-1 Y) for the accepted proposal factorization M.
Intercept projections come from fixed design products. Only Y is projected.
"""
function shift_terms!(qc,sc,Y)
    w, a = qc.work.proposal,qc.d
    f, g = view(w.fx,1:a), view(w.ones_fx,1:a)
    if sc.ns > 0
        mul!(w.yu,transpose(sc.U),Y)
    end
    mul!(f,transpose(get_X1(qc)),Y)
    @inbounds for i in 1:a
        g[i] = qc.xtxdata[a,qc.i1_set[i]]
    end
    q11, q1y = sc.n/w.s, sum(Y)/w.s
    @inbounds for i in 1:sc.ns
        ui = sc.UtX[i,sc.paug]
        q11 -= w.d2[i]*ui^2
        q1y -= w.d2[i]*ui*w.yu[i]
        w.weighted_y[i] = w.d2[i]*w.yu[i]
    end
    if sc.ns > 0
        mul!(f,transpose(view(w.utx,:,1:a)),w.weighted_y,-one(w.s),inv(w.s))
        @inbounds for i in 1:sc.ns
            w.weighted_y[i] = w.d2[i]*sc.UtX[i,sc.paug]
        end
        mul!(g,transpose(view(w.utx,:,1:a)),w.weighted_y,-one(w.s),inv(w.s))
    else
        f ./= w.s
        g ./= w.s
    end
    L = LowerTriangular(view(w.factor,1:a,1:a))
    ldiv!(L,f)
    ldiv!(L,g)
    return q11-dot(g,g),q1y-dot(g,f)
end

"""Quadratic form of a new response under the already prepared proposal factor."""
function response_ssq!(qc,sc,Y)
    w, a = qc.work.proposal,qc.d
    f = view(w.fx,1:a)
    mul!(f,transpose(get_X1(qc)),Y)
    copyto!(w.residual,Y)
    if sc.ns > 0
        mul!(w.yu,transpose(sc.U),Y)
        mul!(w.residual,sc.U,w.yu,-one(w.s),one(w.s))
        @inbounds @simd for i in 1:sc.ns
            w.weighted_y[i] = w.d2[i]*w.yu[i]
        end
        mul!(f,transpose(view(w.utx,:,1:a)),w.weighted_y,-one(w.s),inv(w.s))
    else
        f ./= w.s
    end
    q = dot(w.residual,w.residual)/w.s
    for i in 1:sc.ns
        q += w.invden[i]*w.yu[i]^2
    end
    ldiv!(LowerTriangular(view(w.factor,1:a,1:a)),f)
    return q-dot(f,f)
end
