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

# The large arrays in FixedCovarianceData are read-only for the cache lifetime.
struct FixedCovarianceData{TX,TU,TS,T}
    Xaug::TX
    U::TU
    S::TS
    V::Matrix{T}
    UtX::Matrix{T}
    sqX::Vector{T}
    n::Int
    paug::Int
    ns::Int
end

mutable struct CovarianceState{T}
    r::T
    s::T
    sinv::T
    d::Vector{T}
    d2::Vector{T}
    invden::Vector{T}
    QX::Matrix{T}
    gramdiag::Vector{T}
    cdet::T
    generation::UInt
end

mutable struct ResponseState{T}
    yu::Vector{T}
    yx::Vector{T}
    yty::T
    perp2::T
    weighted::Vector{T}
    residual::Vector{T}
    fx::Vector{T}
    qterm::T
    generation::UInt
end

"""
    SCache

Model-independent low-rank covariance cache. `data` contains fixed design and
orthonormal eigenbasis data, `covariance` depends only on positive `r, s`, and
`response` contains projections of the current single response. Fixed input
arrays are shared read-only; reconstruct the cache if they change. Updates
are not safe concurrently on the same cache; independent samplers own their
mutable storage.

Legacy read-only properties (`U`, `S`, `V`, `UtX`, `YtU`, `YtX`, `FX`, `QX`,
`d`, `sqX`, `cdet`, `qterm` and dimensions) remain available. Response row
properties are transpose views of ordinary vectors. `F` is no longer stored.
"""
struct SCache{D,T}
    data::D
    covariance::CovarianceState{T}
    response::ResponseState{T}
end

Base.propertynames(::SCache, private::Bool=false) =
    (:data, :covariance, :response, :S, :U, :V, :UtX, :sqX, :n, :paug, :ns,
     :d, :QX, :cdet, :YtU, :YtX, :FX, :qterm)

@inline function Base.getproperty(sc::SCache, name::Symbol)
    if name in (:S, :U, :V, :UtX, :sqX, :n, :paug, :ns)
        return getproperty(getfield(sc, :data), name)
    elseif name in (:d, :QX, :cdet)
        return getproperty(getfield(sc, :covariance), name)
    elseif name === :YtU
        return transpose(getfield(sc, :response).yu)
    elseif name === :YtX
        return transpose(getfield(sc, :response).yx)
    elseif name === :FX
        return transpose(getfield(sc, :response).fx)
    elseif name === :qterm
        return getfield(sc, :response).qterm
    end
    return getfield(sc, name)
end

@inline function _positive_variance(x, name)
    isfinite(x) && x > 0 || throw(ArgumentError("$name must be finite and positive"))
    return nothing
end

# Use log space only outside the safe direct-ratio range. This handles zero
# modes and avoids overflow in r*S or r*S/s for otherwise finite inputs.
function _covariance_weights!(d, d2, invden, S, r, s, n)
    _positive_variance(r, "r")
    _positive_variance(s, "s")
    sinv = inv(s)
    cdet = n * log(s)
    for i in eachindex(S)
        t = r * S[i] / s
        if isfinite(t) && (t > 0 || S[i] == 0)
            w = t / (1 + t)
            ld = log1p(t)
            invden[i] = sinv / (1 + t)
        else
            z = log(r) + log(S[i]) - log(s)
            ld = log1pexp(z)
            w = logistic(z)
            invden[i] = exp(-log(s) - ld)
        end
        d2[i] = w * sinv
        d[i] = sqrt(d2[i])
        cdet += ld
    end
    return cdet
end

# Compatibility for the stand-alone numerical helpers.
function _comp_d!(d, S, r, s)
    _positive_variance(r, "r")
    _positive_variance(s, "s")
    for i in eachindex(d, S)
        z = S[i] == 0 ? -Inf : log(r) + log(S[i]) - log(s)
        d[i] = sqrt(logistic(z) / s)
    end
    return nothing
end

const CACHE_BLOCK_COLUMNS = 256
const CACHE_THREAD_WORK = 32768

# Column dots are memory-bound for large predictor sets. Each worker reads
# contiguous columns and owns disjoint outputs, including strided active rows.
function _column_dots_block!(out,A,x,scale_product,scale_previous,lo,hi)
    @inbounds for j in lo:hi
        q = zero(eltype(out))
        @simd for i in axes(A,1)
            q += A[i,j]*x[i]
        end
        out[j] = iszero(scale_previous) ? scale_product*q : scale_product*q+scale_previous*out[j]
    end
    return nothing
end

function _column_dots!(out,A,x,scale_product,scale_previous)
    P = size(A,2)
    blocks = cld(P,CACHE_BLOCK_COLUMNS)
    if Threads.nthreads() > 1 && length(A) >= CACHE_THREAD_WORK
        @batch for b in 1:blocks
            lo = (b-1)*CACHE_BLOCK_COLUMNS+1
            _column_dots_block!(out,A,x,scale_product,scale_previous,lo,min(lo+CACHE_BLOCK_COLUMNS-1,P))
        end
    else
        _column_dots_block!(out,A,x,scale_product,scale_previous,1,P)
    end
    return nothing
end

function _scaled_columns!(cv, data, lo, hi)
    Q, B, d = cv.QX, data.UtX, cv.d
    @inbounds for j in lo:hi
        q = zero(eltype(Q))
        @simd for i in axes(Q, 1)
            v = d[i] * B[i,j]
            Q[i,j] = v
            q += v*v
        end
        g = cv.sinv * data.sqX[j] - q
        # Avoid catastrophic cancellation when the column is nearly entirely
        # explained by covariance modes with very large variance.
        if g <= 64eps(eltype(Q)) * cv.sinv * data.sqX[j]
            g = zero(g)
            for row in axes(data.Xaug, 1)
                v = data.Xaug[row,j]
                for i in axes(B, 1)
                    v -= data.U[row,i] * B[i,j]
                end
                g += cv.sinv * v*v
            end
            for i in axes(B, 1)
                g += cv.invden[i] * B[i,j]^2
            end
        end
        cv.gramdiag[j] = g
    end
    return nothing
end

function _refresh_covariance!(sc, r, s)
    cv, data = sc.covariance, sc.data
    _positive_variance(r, "r")
    _positive_variance(s, "s")
    r, s = convert(typeof(cv.r), r), convert(typeof(cv.s), s)
    if cv.generation != 0 && r == cv.r && s == cv.s
        return false
    end
    cv.cdet = _covariance_weights!(cv.d, cv.d2, cv.invden, data.S, r, s, data.n)
    cv.r, cv.s, cv.sinv = r, s, inv(s)
    blocks = cld(data.paug, CACHE_BLOCK_COLUMNS)
    if Threads.nthreads() > 1 && data.ns * data.paug >= CACHE_THREAD_WORK
        @batch for b in 1:blocks
            lo = (b-1)*CACHE_BLOCK_COLUMNS + 1
            _scaled_columns!(cv, data, lo, min(lo+CACHE_BLOCK_COLUMNS-1, data.paug))
        end
    else
        _scaled_columns!(cv, data, 1, data.paug)
    end
    cv.generation += UInt(1)
    return true
end

function _refresh_response_terms!(sc)
    cv, rs, data = sc.covariance, sc.response, sc.data
    @inbounds @simd for i in eachindex(rs.weighted)
        rs.weighted[i] = cv.d2[i] * rs.yu[i]
    end
    copyto!(rs.fx, rs.yx)
    if data.ns > 0
        _column_dots!(rs.fx, data.UtX, rs.weighted, -one(cv.s), cv.sinv)
    else
        rs.fx .*= cv.sinv
    end
    # Orthogonal-complement form avoids subtracting two nearly equal norms.
    q = cv.sinv * rs.perp2
    for i in eachindex(rs.yu)
        q += cv.invden[i] * rs.yu[i]^2
    end
    rs.qterm = q
    return nothing
end

"""Refresh response projections and dependent terms, including in-place Y changes."""
function update_response!(sc::SCache, Y)
    data, rs = sc.data, sc.response
    Y isa AbstractVector && length(Y) == data.n || throw(DimensionMismatch("response must be a vector of the original length"))
    _column_dots!(rs.yx, data.Xaug, Y, one(eltype(rs.yx)), zero(eltype(rs.yx)))
    copyto!(rs.residual, Y)
    if data.ns > 0
        mul!(rs.yu, transpose(data.U), Y)
        mul!(rs.residual, data.U, rs.yu, -one(eltype(rs.yu)), one(eltype(rs.yu)))
    end
    rs.yty = dot(Y, Y)
    rs.perp2 = dot(rs.residual, rs.residual)
    rs.generation += UInt(1)
    _refresh_response_terms!(sc)
    return nothing
end

"""Refresh covariance-dependent terms without repeating response projections."""
function update_covariance!(sc::SCache, r, s)
    _refresh_covariance!(sc, r, s) && _refresh_response_terms!(sc)
    return nothing
end

"""
    makeSCache(ps)

Construct a cache for fixed Xaug and an orthonormal eigenbasis U. Eigenvalues
must be finite and nonnegative; an empty basis is supported. No orthogonality
check is performed in the hot path.
"""
function makeSCache(ps::ParState2)
    X, Y, U, S = ps.Xaug, ps.Y, ps.U, ps.S
    n, P = size(X)
    n > 0 && P > 0 && Y isa AbstractVector && size(U,1) == n && length(Y) == n && size(U,2) == length(S) &&
        length(S) <= n && ps.n == n && ps.p+1 == P ||
        throw(DimensionMismatch("design, response and eigenbasis dimensions disagree"))
    all(x -> isfinite(x) && x >= 0, S) || throw(ArgumentError("eigenvalues must be finite and nonnegative"))
    _positive_variance(ps.r, "r")
    _positive_variance(ps.s, "s")
    T = promote_type(float(eltype(X)), float(eltype(Y)), float(eltype(U)), float(eltype(S)), typeof(ps.r), typeof(ps.s))
    k = length(S)
    V = Matrix{T}(undef,n,k)
    for j in 1:k, i in 1:n
        V[i,j] = U[i,j] * sqrt(S[j])
    end
    B = Matrix{T}(undef,k,P)
    k > 0 && mul!(B, transpose(U), X)
    sqX = zeros(T,P)
    for j in 1:P
        v = zero(T)
        @inbounds @simd for i in 1:n
            v += abs2(X[i,j])
        end
        sqX[j] = v
    end
    data = FixedCovarianceData(X,U,S,V,B,sqX,n,P,k)
    cv = CovarianceState(T(ps.r),T(ps.s),inv(T(ps.s)),zeros(T,k),zeros(T,k),zeros(T,k),
        zeros(T,k,P),zeros(T,P),zero(T),UInt(0))
    rs = ResponseState(zeros(T,k),zeros(T,P),zero(T),zero(T),zeros(T,k),
        zeros(T,n),zeros(T,P),zero(T),UInt(0))
    sc = SCache(data,cv,rs)
    updateSCache!(sc,ps)
    return sc
end

"""
    updateSCache!(sc, ps)

Compatibility refresh. Always refresh the response (which may have changed
in place); refresh covariance-only arrays only when r or s changes. Replacing
fixed arrays requires a new cache.
"""
function updateSCache!(sc::SCache, ps)
    ps.Xaug === sc.data.Xaug && ps.U === sc.data.U && ps.S === sc.data.S ||
        throw(ArgumentError("fixed data changed; construct a new SCache"))
    _refresh_covariance!(sc, ps.r, ps.s)
    update_response!(sc, ps.Y)
    return nothing
end
