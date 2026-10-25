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
    Aloglik{T<:AbstractFloat}

Abstract interface for collapsed log-likelihood evaluators.

Concrete subtypes accept a log-determinant term and a quadratic form of type
`T` and return the corresponding log likelihood of type `T`, up to an additive
constant. Consumers share this type with their parameter state and caches.
"""
abstract type Aloglik{T<:AbstractFloat} end

"""
    TDistLogLik{T} <: Aloglik{T}

Collapsed Student-t log likelihood obtained after integrating over a residual
variance with scaled-inverse-chi-squared hyperparameters.

# Fields
- `nu`: Prior degrees of freedom.
- `lambda`: Prior scale.
- `n`: Number of observations.
"""
struct TDistLogLik{T<:AbstractFloat} <: Aloglik{T}
    nu::T
    lambda::T
    n::T
end

"""
    TDistLogLik(nu, lambda, n)

Construct a collapsed Student-t likelihood after promoting its three
parameters to a common floating-point type. Use `TDistLogLik{T}` to select
the type explicitly.
"""
function TDistLogLik(nu::Real,lambda::Real,n::Real)
    T = _state_float_type(typeof(nu),typeof(lambda),typeof(n))
    return TDistLogLik{T}(nu,lambda,n)
end

"""
    (x::TDistLogLik)(det, ssq)

Evaluate the collapsed log likelihood for log-determinant `det` and quadratic
form `ssq`, omitting constants shared by all models.
"""
@inline (x::TDistLogLik{T})(det::T, ssq::T) where{T} = -(one(T)/2) * det - (one(T)/2) * (x.nu+x.n) * log(x.nu*x.lambda + ssq)

"""
    ll_plus_a!(ll, det, ssq, a, x::TDistLogLik)

Fill `ll` with the elementwise collapsed Student-t log likelihood for `det`
and `ssq`, adding the constant `a` to every result.
"""
function ll_plus_a!(ll::AbstractVector{T}, det::AbstractVector{T}, ssq::AbstractVector{T}, a::T, x::TDistLogLik{T}) where{T<:AbstractFloat}
    nu=x.nu
    n=x.n
    lambda=x.lambda
    @turbo for i in axes(ll,1)
        ll[i] = -(one(T)/2) * det[i] - (one(T)/2) * (nu+n) * log(nu*lambda + ssq[i]) + a
    end
end

"""
    StdNormLogLik{T} <: Aloglik{T}

Collapsed standard-normal log likelihood for determinant and quadratic-form
terms of element type `T`.
"""
struct StdNormLogLik{T<:AbstractFloat} <: Aloglik{T} end

"""
    (x::StdNormLogLik)(det, ssq)

Evaluate the standard-normal log likelihood for log-determinant `det` and
quadratic form `ssq`, omitting constants shared by all models.
"""
(x::StdNormLogLik{T})(det::T, ssq::T) where{T} = -(one(T)/2) * det - (one(T)/2) * ssq

"""
    ll_plus_a!(ll, det, ssq, a, x::StdNormLogLik)

Fill `ll` with the elementwise standard-normal log likelihood for `det` and
`ssq`, adding the constant `a` to every result.
"""
function ll_plus_a!(ll::AbstractVector{T}, det::AbstractVector{T}, ssq::AbstractVector{T}, a::T, x::StdNormLogLik{T}) where{T<:AbstractFloat}
    @turbo for i in axes(ll,1)
        ll[i] = -(one(T)/2) * det[i] - (one(T)/2) * ssq[i] + a
    end
end
