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
    ParState2

Mutable model parameters and data shared by the sampler's numerical caches.
All numerical arrays and continuous parameters use the concrete floating-point
type `T`; counts use `Int`. Matching dense inputs are shared with the caller.

# Fields
- `Xaug`: Predictor matrix with an intercept column appended last.
- `Y`: Current latent-response vector.
- `U`, `S`: Eigenvectors and eigenvalues of the random-effect covariance.
- `p`, `n`: Numbers of predictors and observations, respectively.
- `alpha`, `beta`: Beta-Binomial inclusion-prior hyperparameters.
- `c`: Predictor-effect variance.
- `r`: Random-effect variance multiplier.
- `s`: Residual variance multiplier.
- `mc`: Intercept variance.
"""
mutable struct ParState2{T<:AbstractFloat}
    Xaug::Matrix{T}
    Y::Vector{T}
    U::Matrix{T}
    S::Vector{T}
    p::Int
    n::Int
    alpha::T
    beta::T
    c::T
    r::T
    s::T
    mc::T

    function ParState2{T}(Xaug, Y, U, S, p::Integer, n::Integer,
                          alpha, beta, c, r, s, mc) where {T<:AbstractFloat}
        isconcretetype(T) || throw(ArgumentError("state element type must be concrete"))
        Xaug, Y = convert(Matrix{T}, Xaug), convert(Vector{T}, Y)
        U, S = convert(Matrix{T}, U), convert(Vector{T}, S)
        alpha, beta, c, r, s, mc = T.((alpha, beta, c, r, s, mc))
        n > 0 && p >= 0 && size(Xaug) == (n,p+1) && length(Y) == n &&
            size(U) == (n,length(S)) && length(S) <= n ||
            throw(DimensionMismatch("design, response, eigenbasis and counts must agree"))
        all(isfinite,Xaug) && all(isfinite,Y) && all(isfinite,U) ||
            throw(ArgumentError("design, response and eigenvectors must be finite"))
        all(x -> isfinite(x) && x >= 0,S) ||
            throw(ArgumentError("eigenvalues must be finite and nonnegative"))
        for (name,value) in (("alpha",alpha),("beta",beta),("c",c),("r",r),("s",s),("mc",mc))
            isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
        end
        return new{T}(Xaug,Y,U,S,p,n,alpha,beta,c,r,s,mc)
    end
end

# Promote first so integer inputs do not unnecessarily widen floating inputs.
function _state_float_type(types...)
    T = float(promote_type(types...))
    T <: AbstractFloat && isconcretetype(T) ||
        throw(ArgumentError("state inputs must promote to a concrete floating-point type"))
    return T
end

"""
    ParState2(X, Y, U, S, alpha, beta, c, r, s, mc)
    ParState2{T}(X, Y, U, S, alpha, beta, c, r, s, mc)

Construct sampler parameters and append a column of ones to `X` for the
intercept.

`X` has one row per observation, `Y` has one element per observation, and the
rows of `U` correspond to the same observations. Infer `T` from all array
element types and scalar parameters, or select it explicitly. Inputs are
converted to dense storage once; already matching `Y`, `U` and `S` are shared.
"""
function ParState2(X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real},
                   U::AbstractMatrix{<:Real}, S::AbstractVector{<:Real},
                   alpha::Real, beta::Real, c::Real, r::Real, s::Real, mc::Real)
    T = _state_float_type(eltype(X),eltype(Y),eltype(U),eltype(S),
                         typeof(alpha),typeof(beta),typeof(c),typeof(r),typeof(s),typeof(mc))
    return ParState2{T}(X,Y,U,S,alpha,beta,c,r,s,mc)
end

function ParState2{T}(X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real},
                      U::AbstractMatrix{<:Real}, S::AbstractVector{<:Real},
                      alpha::Real, beta::Real, c::Real, r::Real, s::Real, mc::Real) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError("state element type must be concrete"))
    n, p = size(X)
    Xaug = Matrix{T}(undef,n,p+1)
    copyto!(view(Xaug,:,1:p),X)
    fill!(view(Xaug,:,p+1),one(T))
    return ParState2{T}(Xaug,Y,U,S,p,n,alpha,beta,c,r,s,mc)
end

# Preserve the constructor for an already augmented design.
function ParState2(Xaug, Y, U, S, p::Integer, n::Integer, alpha, beta, c, r, s, mc)
    T = _state_float_type(eltype(Xaug),eltype(Y),eltype(U),eltype(S),
                         typeof(alpha),typeof(beta),typeof(c),typeof(r),typeof(s),typeof(mc))
    return ParState2{T}(Xaug,Y,U,S,p,n,alpha,beta,c,r,s,mc)
end
