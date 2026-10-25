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
    FixedCovarianceData

Fixed design and covariance eigenbasis data, read-only for the cache lifetime.
All arrays use the common element type `T` and dense storage.
Let `n` be the observation count, `paug` the augmented predictor count, and
`ns` the retained covariance rank. The intercept is the final design column.

# Fields
- `Xaug`: Shared `n × paug` design matrix, including the intercept.
- `U`: Shared `n × ns` matrix of orthonormal covariance eigenvectors.
- `S`: Shared length-`ns` vector of nonnegative covariance eigenvalues.
- `V`: Cached `n × ns` covariance square-root factor `U * Diagonal(sqrt.(S))`.
- `UtX`: Cached `ns × paug` projection `U' * Xaug`.
- `sqX`: Length-`paug` vector of squared Euclidean design-column norms.
- `n`: Number of observations.
- `paug`: Number of augmented predictors, including the intercept.
- `ns`: Number of retained covariance modes; may be zero.
"""
struct FixedCovarianceData{T}
    Xaug::Matrix{T}
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
    UtX::Matrix{T}
    sqX::Vector{T}
    n::Int
    paug::Int
    ns::Int
end

"""
    CovarianceState

Quantities depending on the base covariance `C = s*I + r*U*Diagonal(S)*U'`.
They are refreshed when either positive variance multiplier changes.

# Fields
- `r`: Random-effect variance multiplier used by the cached quantities.
- `s`: Residual variance multiplier used by the cached quantities.
- `sinv`: Cached reciprocal `1/s`.
- `d`: Length-`ns` vector of nonnegative square roots of the Woodbury weights.
- `d2`: Squared weights, `r*S[i] / (s * (s + r*S[i]))` for each mode.
- `invden`: Length-`ns` vector with entries `1 / (s + r*S[i])`.
- `QX`: `ns × paug` matrix `Diagonal(d) * UtX`.
- `gramdiag`: Length-`paug` diagonal of `Xaug' * inv(C) * Xaug`.
- `cdet`: Base-covariance log determinant `logdet(C)`.
- `generation`: Covariance refresh counter used to invalidate downstream
  products and factorizations; zero denotes uninitialized state.
"""
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

"""
    CovarianceState{T}(r, s, ns, paug)

Convert variance parameters to `T` and allocate covariance buffers for `ns`
modes and `paug` augmented predictors. Generation zero marks the derived
quantities as uninitialized until the first covariance refresh.
"""
function CovarianceState{T}(r, s, ns::Integer, paug::Integer) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError("covariance element type must be concrete"))
    r, s = T(r), T(s)
    _positive_variance(r,"r")
    _positive_variance(s,"s")
    return CovarianceState{T}(r,s,inv(s),zeros(T,ns),zeros(T,ns),zeros(T,ns),
                              zeros(T,ns,paug),zeros(T,paug),zero(T),UInt(0))
end

"""
    ResponseState

Single-response projections and covariance-adjusted forms. Projections change
when the response changes; `weighted`, `fx` and `qterm` also change when the
base covariance changes.

# Fields
- `yu`: Length-`ns` projection `U' * Y`.
- `yx`: Length-`paug` cross-product `Xaug' * Y`.
- `yty`: Squared response norm `dot(Y, Y)`.
- `perp2`: Squared norm of the orthogonal residual `Y - U * yu`.
- `weighted`: Length-`ns` workspace holding `d2 .* yu`.
- `residual`: Length-`n` buffer holding `Y - U * yu` after response refresh.
- `fx`: Length-`paug` vector `Xaug' * inv(C) * Y`.
- `qterm`: Base-covariance response quadratic form `Y' * inv(C) * Y`.
- `generation`: Response refresh counter used to invalidate active response
  solves; covariance-only refreshes do not increment it.
"""
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
    ResponseState{T}(n, ns, paug)

Allocate response buffers and initialize scalar storage using `T`, for `n`
observations, `ns` covariance modes and `paug` augmented predictors. Generation
zero marks the response quantities as uninitialized until the first refresh.
"""
function ResponseState{T}(n::Integer, ns::Integer, paug::Integer) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError("response element type must be concrete"))
    return ResponseState{T}(zeros(T,ns),zeros(T,paug),zero(T),zero(T),zeros(T,ns),
                            zeros(T,n),zeros(T,paug),zero(T),UInt(0))
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
properties are transpose views of ordinary vectors. 

# Fields
- `data`: Fixed design/eigenbasis inputs, cross-products and dimensions in a
  `FixedCovarianceData` object.
- `covariance`: `CovarianceState` containing variance-dependent weights,
  adjusted design products and the covariance generation counter.
- `response`: `ResponseState` containing current response projections,
  adjusted forms, reusable buffers and the response generation counter.
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

# Column dots are memory-bound for large predictor sets. Each worker reads
# contiguous columns and owns disjoint outputs, including strided active rows.
function _column_dots_block!(out,A,x,scale_product,scale_previous,lo,hi)
    mul!(view(out,lo:hi), transpose(view(A,:,lo:hi)), x,
        scale_product, scale_previous)
    
    return nothing
end

function _column_dots!(out,A,x,scale_product,scale_previous)
    
    P = size(A,2)
    blocks = cld(P,CACHE_BLOCK_COLUMNS)
    
    if Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && length(A) >= CACHE_THREAD_WORK
        # Keep the original array types for mul!: Polyester otherwise wraps
        # captured arrays in PtrArray, changing matrix-vector dispatch.
        arrays = (;out,A,x)
        @batch for b in 1:blocks
            lo = (b-1)*CACHE_BLOCK_COLUMNS+1
            _column_dots_block!(arrays.out,arrays.A,arrays.x,scale_product,scale_previous,
                                lo,min(lo+CACHE_BLOCK_COLUMNS-1,P))
        end
    else
        _column_dots_block!(out,A,x,scale_product,scale_previous,1,P)
    end

    return nothing
end

function _scaled_columns!(covariance, data, lo, hi)
    
    QX, UtX, d = covariance.QX, data.UtX, covariance.d
    
    @inbounds for j in lo:hi
        q = zero(eltype(QX))
        
        @simd for i in axes(QX, 1)
            v = d[i] * UtX[i,j]
            QX[i,j] = v
            q += v*v
        end

        g = covariance.sinv * data.sqX[j] - q
        # Avoid catastrophic cancellation when the column is nearly entirely
        # explained by covariance modes with very large variance.
        if g <= 64eps(eltype(QX)) * covariance.sinv * data.sqX[j]
            g = zero(g)
            for row in axes(data.Xaug, 1)
                v = data.Xaug[row,j]
                for i in axes(UtX, 1)
                    v -= data.U[row,i] * UtX[i,j]
                end
                g += covariance.sinv * v*v
            end
            for i in axes(UtX, 1)
                g += covariance.invden[i] * UtX[i,j]^2
            end
        end
        covariance.gramdiag[j] = g
    end

    return nothing
end

function _refresh_covariance!(sc, r, s)
    
    covariance, data = sc.covariance, sc.data
    _positive_variance(r, "r")
    _positive_variance(s, "s")
    r, s = convert(typeof(covariance.r), r), convert(typeof(covariance.s), s)
    
    if covariance.generation != 0 && r == covariance.r && s == covariance.s
        return false
    end

    covariance.cdet = _covariance_weights!(covariance.d, covariance.d2, covariance.invden, data.S, r, s, data.n)
    covariance.r, covariance.s, covariance.sinv = r, s, inv(s)
    blocks = cld(data.paug, CACHE_BLOCK_COLUMNS)

    if Threads.nthreads() > 1 && data.ns * data.paug >= CACHE_THREAD_WORK
        @batch for b in 1:blocks
            lo = (b-1)*CACHE_BLOCK_COLUMNS + 1
            _scaled_columns!(covariance, data, lo, min(lo+CACHE_BLOCK_COLUMNS-1, data.paug))
        end
    else
        _scaled_columns!(covariance, data, 1, data.paug)
    end

    covariance.generation += UInt(1)
    return true
end

function _refresh_response_terms!(sc)
    covariance, response, data = sc.covariance, sc.response, sc.data
    @inbounds @simd for i in eachindex(response.weighted)
        response.weighted[i] = covariance.d2[i] * response.yu[i]
    end
    copyto!(response.fx, response.yx)
    if data.ns > 0
        _column_dots!(response.fx, data.UtX, response.weighted, -one(covariance.s), covariance.sinv)
    else
        response.fx .*= covariance.sinv
    end
    # Orthogonal-complement form avoids subtracting two nearly equal norms.
    q = covariance.sinv * response.perp2
    for i in eachindex(response.yu)
        q += covariance.invden[i] * response.yu[i]^2
    end
    response.qterm = q
    return nothing
end

"""Refresh response projections and dependent terms, including in-place Y changes."""
function update_response!(sc::SCache, Y)
    data, response = sc.data, sc.response
    Y isa AbstractVector && length(Y) == data.n || throw(DimensionMismatch("response must be a vector of the original length"))
    _column_dots!(response.yx, data.Xaug, Y, one(eltype(response.yx)), zero(eltype(response.yx)))
    copyto!(response.residual, Y)
    if data.ns > 0
        mul!(response.yu, transpose(data.U), Y)
        mul!(response.residual, data.U, response.yu, -one(eltype(response.yu)), one(eltype(response.yu)))
    end
    response.yty = dot(Y, Y)
    response.perp2 = dot(response.residual, response.residual)
    response.generation += UInt(1)
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
function makeSCache(ps::ParState2{T}) where {T}
   
    Xaug, Y, U, S = ps.Xaug, ps.Y, ps.U, ps.S 
    n, paug = size(Xaug)

    n > 0 && paug > 0 && Y isa AbstractVector && size(U,1) == n && length(Y) == n && size(U,2) == length(S) &&
        length(S) <= n && ps.n == n && ps.p+1 == paug ||
        throw(DimensionMismatch("design, response and eigenbasis dimensions disagree"))
    
        all(x -> isfinite(x) && x >= 0, S) || throw(ArgumentError("eigenvalues must be finite and nonnegative"))
    
    _positive_variance(ps.r, "r")
    _positive_variance(ps.s, "s")

    ns = length(S)
    V = Matrix{T}(undef,n,ns)

    for j in 1:ns, i in 1:n
        V[i,j] = U[i,j] * sqrt(S[j])
    end

    UtX = Matrix{T}(undef,ns,paug)
    ns > 0 && mul!(UtX, transpose(U), Xaug)
    sqX = zeros(T,paug)

    for j in 1:paug
        v = zero(T)
        @inbounds @simd for i in 1:n
            v += abs2(Xaug[i,j])
        end
        sqX[j] = v
    end

    data = FixedCovarianceData(Xaug,U,S,V,UtX,sqX,n,paug,ns)
    covariance = CovarianceState{T}(ps.r,ps.s,ns,paug)
    response = ResponseState{T}(n,ns,paug)
    sc = SCache(data,covariance,response)

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
