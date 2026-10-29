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

function ProposalWorkspace(::Type{T}, n, k, a) where {T}
    nan_t, zero_t = T(NaN), zero(T)
    return ProposalWorkspace(
        zeros(T, a, a), zeros(T, k, a), zeros(T, k, a), zeros(T, a, a),
        zeros(T, k), zeros(T, k), zeros(T, k), zeros(T, k), zeros(T, a), zeros(T, a),
        zeros(T, k), zeros(T, n), typemax(UInt), typemax(UInt), nan_t, nan_t, nan_t, nan_t, zero_t, zero_t
    )
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
    factor_key::Tuple{UInt, UInt, T, T}
    response_generation::UInt
    det::T
    ssq::T
    proposal::ProposalWorkspace{T}
end

function ActiveWorkspace(::Type{T}, n, k, a) where {T}
    nan_t, zero_t = T(NaN), zero(T)
    return ActiveWorkspace(
        zeros(T, k, a), zeros(T, a, a), zeros(T, a, a), zeros(T, a), zeros(T, a), zeros(T, a),
        [zeros(T, a, CACHE_BLOCK_COLUMNS) for _ in 1:Threads.nthreads()],
        (typemax(UInt), typemax(UInt), nan_t, nan_t), typemax(UInt), zero_t, zero_t,
        ProposalWorkspace(T, n, k, a)
    )
end

# Numerical kernels use strided, contiguous-column views of reserved storage.
# Index-vector gathers are copied into those buffers once per model.
function _active_product_block!(data, lo, hi, worker)
    (; qc, sc) = data
    d = qc.d
    covdata = view(qc.covdata, 1:d, lo:hi)
    copyto!(covdata, view(qc.xtxdata, 1:d, lo:hi))

    if sc.ns > 0
        mul!(
            covdata, transpose(view(qc.work.qactive, :, 1:d)), view(sc.QX, :, lo:hi),
            -one(sc.covariance.s), sc.covariance.sinv
        )
    else
        covdata .*= sc.covariance.sinv
    end

    return 0
end

"""Synchronize G = X_active' C^-1 X, preserving valid rows across model flips."""
function sync_active_products!(qc, sc)
    covariance = sc.covariance
    nan_t = typeof(covariance.r)(NaN)
    if qc.cov_source !== covariance || qc.cov_generation != covariance.generation

        paug = sc.paug

        @inbounds for j in 1:qc.d, i in 1:sc.ns
            qc.work.qactive[i, j] = sc.QX[i, qc.i1_set[j]]
        end

        threaded = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && sc.ns * qc.d * paug >= CACHE_THREAD_WORK
        foreach_block!(
            _active_product_block!, (; qc, sc), paug;
            blocksize = threaded ? CACHE_BLOCK_COLUMNS : max(paug, 1),
            workers = threaded ? Threads.nthreads() : 1
        )

        fill!(qc.validrows, true)
        qc.cov_generation = covariance.generation
        qc.cov_source = covariance
        qc.work.factor_key = (typemax(UInt), typemax(UInt), nan_t, nan_t)
    else
        for i in 1:qc.d
            if !qc.validrows[i]
                covdata = view(qc.covdata, i, :)
                copyto!(covdata, view(qc.xtxdata, i, :))
                if sc.ns > 0
                    _column_dots!(covdata, sc.QX, view(sc.QX, :, qc.i1_set[i]), -one(covariance.s), covariance.sinv)
                else
                    covdata .*= covariance.sinv
                end
                qc.validrows[i] = true
            end
        end
    end
    return nothing
end

"""Prepare accepted-model factorization and solves, keyed by their exact dependencies."""
function prepare_active!(qc, sc, c, mc)
    _positive_variance(c, "c")
    _positive_variance(mc, "mc")
    sync_active_products!(qc, sc)
    work, d = qc.work, qc.d
    T = eltype(work.invlower)
    nan_t, zero_t = T(NaN), zero(T)
    factor_key = (qc.generation, sc.covariance.generation, c, mc)
    factor = view(work.factor, 1:d, 1:d)

    if work.factor_key != factor_key
        # A failed factorization must never leave an old factor_key on modified storage.
        work.factor_key = (typemax(UInt), typemax(UInt), nan_t, nan_t)

        @inbounds for j in 1:d
            copyto!(view(factor, :, j), view(qc.covdata, 1:d, qc.i1_set[j]))
        end

        for i in 1:d
            factor[i, i] += i == d ? inv(mc) : inv(c)
        end

        cholesky!(Symmetric(factor, :L))
        work.det = sc.cdet + (d - 1) * log(c) + log(mc)

        for i in 1:d
            work.det += 2log(factor[i, i])
        end

        invlower = view(work.invlower, 1:d, 1:d)
        fill!(invlower, zero_t)

        for i in 1:d
            invlower[i, i] = one(T)
        end

        ldiv!(LowerTriangular(factor), invlower)

        @inbounds for j in 1:d
            h = zero_t

            @turbo for i in 1:d
                h += abs2(invlower[i, j])
            end

            work.invdiag[j] = h
        end

        work.factor_key = factor_key
        work.response_generation = typemax(UInt)
    end

    if work.response_generation != sc.response.generation
        u = view(work.u, 1:d)
        copyto!(u, view(sc.response.fx, qc.i1_set))
        ldiv!(LowerTriangular(factor), u)
        work.ssq = sc.qterm - dot(u, u)
        copyto!(view(work.beta, 1:d), u)
        ldiv!(UpperTriangular(transpose(factor)), view(work.beta, 1:d))
        work.response_generation = sc.response.generation
    end

    return work
end

function _candidate_block!(data, lo, hi, worker)
    (; dets, ssq, qc, sc, c) = data
    work, d = qc.work, qc.d
    zero_t = zero(c)
    # Scratch belongs to logical workers, independent of task migration/thread IDs.
    buffer = work.blocks[worker]
    block = view(buffer, 1:d, 1:(hi - lo + 1))
    copyto!(block, view(qc.covdata, 1:d, lo:hi))
    ldiv!(LowerTriangular(view(work.factor, 1:d, 1:d)), block)
    cinv, lc = inv(c), log(c)

    @inbounds for j in axes(block, 2)
        normb, uy = zero_t, zero_t

        @turbo for i in 1:d
            x = block[i, j]
            normb += x * x
            uy += x * work.u[i]
        end

        index = lo + j - 1
        h = cinv + sc.covariance.gramdiag[index] - normb
        # Propagate failures to the calling task; never throw across a
        # Polyester worker callback.
        h > 0 || return index
        residual = sc.response.fx[index] - uy
        ssq[index] = work.ssq - residual^2 / h
        dets[index] = work.det + log(h) + lc
    end

    return 0
end

function candidate_terms!(dets, ssq, qc, sc, c)
    p = length(dets)
    threaded = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && qc.d * p >= CACHE_THREAD_WORK

    bad = foreach_block!(
        _candidate_block!, (; dets, ssq, qc, sc, c), p;
        blocksize = CACHE_BLOCK_COLUMNS,
        workers = threaded ? length(qc.work.blocks) : 1
    )
    bad == 0 || throw(PosDefException(bad))

    return nothing
end

function prepare_proposal!(qc, sc, r, s, c, mc)
    _positive_variance(c, "c")
    _positive_variance(mc, "mc")
    proposal, d = qc.work.proposal, qc.d

    if proposal.design_generation != qc.generation
        @inbounds for j in 1:d
            index = qc.i1_set[j]
            copyto!(view(proposal.gram, 1:d, j), view(qc.xtxdata, 1:d, index))
            copyto!(view(proposal.utx, :, j), view(sc.UtX, :, index))
        end

        proposal.design_generation = qc.generation
        proposal.factor_generation = typemax(UInt)
    end

    if proposal.factor_generation == qc.generation && (r, s, c, mc) == (proposal.r, proposal.s, proposal.c, proposal.mc)
        return proposal
    end

    proposal.factor_generation = typemax(UInt)
    proposal.cdet = _covariance_weights!(proposal.d, proposal.d2, proposal.invden, sc.S, r, s, sc.n)
    weighted_design = view(proposal.weighted_design, :, 1:d)

    @inbounds for j in 1:d
        @turbo for i in 1:sc.ns
            weighted_design[i, j] = proposal.d[i] * proposal.utx[i, j]
        end
    end

    factor = view(proposal.factor, 1:d, 1:d)
    copyto!(factor, view(proposal.gram, 1:d, 1:d))

    if sc.ns > 0
        mul!(factor, transpose(weighted_design), weighted_design, -one(s), inv(s))
    else
        factor ./= s
    end

    for i in 1:d
        factor[i, i] += i == d ? inv(mc) : inv(c)
    end

    cholesky!(Symmetric(factor, :L))
    proposal.det = proposal.cdet + (d - 1) * log(c) + log(mc)

    for i in 1:d
        proposal.det += 2log(factor[i, i])
    end

    proposal.r, proposal.s, proposal.c, proposal.mc = r, s, c, mc
    proposal.factor_generation = qc.generation

    return proposal
end

"""Evaluate a proposal using fixed response projections without changing accepted caches."""
function proposal_terms!(qc, sc, r, c, mc; s = sc.covariance.s)
    proposal = prepare_proposal!(qc, sc, r, s, c, mc)
    d, response = qc.d, sc.response

    @inbounds @turbo for i in 1:sc.ns
        proposal.weighted_y[i] = proposal.d2[i] * response.yu[i]
    end

    fx = view(proposal.fx, 1:d)
    copyto!(fx, view(response.yx, qc.i1_set))

    if sc.ns > 0
        mul!(fx, transpose(view(proposal.utx, :, 1:d)), proposal.weighted_y, -one(s), inv(s))
    else
        fx ./= s
    end

    ldiv!(LowerTriangular(view(proposal.factor, 1:d, 1:d)), fx)
    q = response.perp2 / s

    for i in 1:sc.ns
        q += proposal.invden[i] * response.yu[i]^2
    end

    return q - dot(fx, fx), proposal.det
end

"""
Return (1'M^-1 1, 1'M^-1 Y) for the accepted proposal factorization M.
Intercept projections come from fixed design products. Only Y is projected.
"""
function shift_terms!(qc, sc, Y)
    proposal, d = qc.work.proposal, qc.d
    one_t = one(proposal.s)
    fx, ones_fx = view(proposal.fx, 1:d), view(proposal.ones_fx, 1:d)

    if sc.ns > 0
        mul!(proposal.yu, transpose(sc.U), Y)
    end

    mul!(fx, transpose(get_X1(qc)), Y)
    copyto!(ones_fx, view(qc.xtxdata, d, qc.i1_set))
    q11, q1y = sc.n / proposal.s, sum(Y) / proposal.s

    @inbounds for i in 1:sc.ns
        ui = sc.UtX[i, sc.paug]
        q11 -= proposal.d2[i] * ui^2
        q1y -= proposal.d2[i] * ui * proposal.yu[i]
        proposal.weighted_y[i] = proposal.d2[i] * proposal.yu[i]
    end

    if sc.ns > 0
        mul!(fx, transpose(view(proposal.utx, :, 1:d)), proposal.weighted_y, -one_t, inv(proposal.s))
        @inbounds for i in 1:sc.ns
            proposal.weighted_y[i] = proposal.d2[i] * sc.UtX[i, sc.paug]
        end
        mul!(ones_fx, transpose(view(proposal.utx, :, 1:d)), proposal.weighted_y, -one_t, inv(proposal.s))
    else
        fx ./= proposal.s
        ones_fx ./= proposal.s
    end

    factor = LowerTriangular(view(proposal.factor, 1:d, 1:d))
    ldiv!(factor, fx)
    ldiv!(factor, ones_fx)

    return q11 - dot(ones_fx, ones_fx), q1y - dot(ones_fx, fx)
end

"""Quadratic form of a new response under the already prepared proposal factor."""
function response_ssq!(qc, sc, Y)
    proposal, d = qc.work.proposal, qc.d
    one_t = one(proposal.s)
    fx = view(proposal.fx, 1:d)
    mul!(fx, transpose(get_X1(qc)), Y)
    copyto!(proposal.residual, Y)

    if sc.ns > 0
        mul!(proposal.yu, transpose(sc.U), Y)
        mul!(proposal.residual, sc.U, proposal.yu, -one_t, one_t)
        @inbounds @turbo for i in 1:sc.ns
            proposal.weighted_y[i] = proposal.d2[i] * proposal.yu[i]
        end
        mul!(fx, transpose(view(proposal.utx, :, 1:d)), proposal.weighted_y, -one_t, inv(proposal.s))
    else
        fx ./= proposal.s
    end

    q = dot(proposal.residual, proposal.residual) / proposal.s

    for i in 1:sc.ns
        q += proposal.invden[i] * proposal.yu[i]^2
    end

    ldiv!(LowerTriangular(view(proposal.factor, 1:d, 1:d)), fx)

    return q - dot(fx, fx)
end
