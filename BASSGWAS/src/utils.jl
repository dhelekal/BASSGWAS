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
    _vpacnormsq!(v, a, X, Y)

Accumulate `a * sum(X[:, i] .* Y[:, i])` into each element `v[i]`.

`X` and `Y` must have identical dimensions, and `v` must have one element per
matrix column. The update is performed in place and the function returns
`nothing`.
"""
function _vpacnormsq!(v, a, X, Y)
    size(X) != size(Y) && throw(ArgumentError("Array dimensions must match"))
    ncol = size(X,2)
    size(v) != (ncol, ) &&  throw(ArgumentError("Vector length, $(size(v)), must match column count, $((ncol, ))"))
    
    for i in axes(X,2), j in axes(X,1)
        @inbounds v[i] += a * X[j,i] * Y[j,i]
    end
end

"""
    _copyto_loo!(dest::AbstractMatrix, A::AbstractMatrix, i)

Copy `A` into `dest` while omitting row and column `i`.

`dest` must be one row and one column smaller than `A`. Indices use Julia's
one-based convention, and `dest` is mutated in place.
"""
function _copyto_loo!(dest::AbstractMatrix, A::AbstractMatrix, i)
    m = size(A,1) 
    n = size(A,2)
    size(dest) != (m-1, n-1) && 
        throw(ArgumentError("Incorrect destination matrix size"))

    j = i-1

    @views dest[1:j, 1:j] .= A[1:j, 1:j]
    @views dest[(j+1):end, 1:j] .= A[(i+1):end, 1:j]

    @views dest[1:j, (j+1):end] .= A[1:j, (i+1):end]
    @views dest[(j+1):end, (j+1):end] .= A[(i+1):end, (i+1):end]

end

"""
    _copyto_loo!(dest::AbstractVector, A::AbstractVector, i)

Copy `A` into `dest` while omitting element `i`.

`dest` must contain one fewer element than `A`. Indices use Julia's one-based
convention, and `dest` is mutated in place.
"""
function _copyto_loo!(dest::AbstractVector, A::AbstractVector, i)
    m = size(A,1) 
    size(dest,1) != (m-1) && 
        throw(ArgumentError("Incorrect destination vector size. Expected $(m-1). Found $(size(dest))."))
 
    j = i-1

    @views dest[1:j] .= A[1:j]
    @views dest[(j+1):end] .= A[(i+1):end]
end
