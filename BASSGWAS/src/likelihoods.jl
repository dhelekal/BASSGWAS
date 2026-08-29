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

abstract type Aloglik end

struct TDistLogLik{T<:Real} <: Aloglik
    nu::T
    lambda::T
    n::T
end

function TDistLogLik(nu,lambda,n)
    nu, lambda, n = promote(nu, lambda, n)
    TDistLogLik(nu,lambda,n)
end

(x::TDistLogLik{T})(det::T, ssq::T) where{T} = -0.5 * det - 0.5 * (x.nu+x.n) * log(x.nu*x.lambda + ssq) 

struct StdNormLogLik{T<:Real} <: Aloglik end
(x::StdNormLogLik{T})(det::T, ssq::T) where{T} = -0.5 * det - 0.5 * ssq 
