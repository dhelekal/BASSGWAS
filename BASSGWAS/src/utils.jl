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

#in place 'v' plus 'a' times squared column sums
function _vpacnormsq!(v, a, X, Y)
    size(X) != size(Y) && throw(ArgumentError("Array dimensions must match"))
    ncol = size(X,2)
    size(v) != (ncol, ) &&  throw(ArgumentError("Vector length, $(size(v)), must match column count, $((ncol, ))"))
    
    for i in axes(X,2), j in axes(X,1)
        @inbounds v[i] += a * X[j,i] * Y[j,i]
    end
end

#Compute X[:, i1]'X[:,i0] assuming that i1 << i0
function _X1tX0_disjoint(X, i1, i0)
    X1 = X[:, i1]'
    P = similar(X, length(i1), length(i0))
    j = 1
    k = 1
    for i in i1
        sp = max(i-j, 0)
        @views mul!(P[:, k:(sp+k-1)], X1, X[:, j:(i-1)])
        k = k+sp
        j = i+1
    end

    imax = i0[end]
    if imax > i1[end]
        sp = max(imax-j+1, 0)
        @views mul!(P[:, k:(sp+k-1)], X1, X[:, j:imax])
    end
    return P
end

#Compute X[:, i1]'X[:,i0] assuming that i1 << i0 and store it in A
function _X1tX0_disjoint!(A, X, i1, i0)
    X1 = X[:, i1]'
    P = similar(X, length(i1), length(i0))
    j = 1
    k = 1
    for i in i1
        sp = max(i-j, 0)
        @views mul!(A[:, k:(sp+k-1)], X1, X[:, j:(i-1)])
        k = k+sp
        j = i+1
    end

    imax = i0[end]
    if imax > i1[end]
        sp = max(imax-j+1, 0)
        @views mul!(A[:, k:(sp+k-1)], X1, X[:, j:imax])
    end
end

#Compute X1'X[:,i0] assuming where i0 is the complement of i1
#Assuming i1 << i0
function _X1tX0_partial(X1, X, i1, i0)
    size(A) != (length(i1), length(i0)) && 
        throw(ArgumentError("Dimensions of A must match index lengths"))
    j = 1
    k = 1
    for i in i1
        sp = max(i-j, 0)
        @views mul!(P[:, k:(sp+k-1)], X1', X[:, j:(i-1)])
        k = k+sp
        j = i+1
    end

    imax = i0[end]
    if imax > i1[end]
        sp = max(imax-j+1, 0)
        @views mul!(P[:, k:(sp+k-1)], X1', X[:, j:imax])
    end
    return P
end

function _X1tX0_partial!(A, X1, X, i1, i0)
    j = 1
    k = 1
    for i in i1
        sp = max(i-j, 0)
        @views mul!(A[:, k:(sp+k-1)], X1', X[:, j:(i-1)])
        k = k+sp
        j = i+1
    end

    imax = i0[end]
    if imax > i1[end]
        sp = max(imax-j+1, 0)
        @views mul!(A[:, k:(sp+k-1)], X1', X[:, j:imax])
    end
end

function _copy_excl(A::AbstractMatrix, idx)
    m = size(A,1) 
    n = size(A,2)
    l = length(idx)
    out = similar(A, m, n-l)
    j = 1
    k = 1
    for i in idx
        sp = max(i-j, 0)
        copyto!(out, 1:m, k:(sp+k-1), A, 1:m, j:(i-1))
        k = k+sp
        j = i+1
    end

    if n > idx[end]
        sp = max(n-j+1, 0)
        copyto!(out, 1:m, k:(sp+k-1), A, 1:m, j:n)
    end
    return out
end

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

function _copyto_loo!(dest::AbstractVector, A::AbstractVector, i)
    m = size(A,1) 
    size(dest,1) != (m-1) && 
        throw(ArgumentError("Incorrect destination vector size. Expected $(m-1). Found $(size(dest))."))
 
    j = i-1

    @views dest[1:j] .= A[1:j]
    @views dest[(j+1):end] .= A[(i+1):end]
end
