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

_shared_block_data(data, worker) = data

function _foreach_block_worker!(context, worker, nb, nw, n, blocksize)::Int
    data = context.worker_data(context.data,worker)
    first_block = fld((worker-1)*nb,nw)+1
    last_block = fld(worker*nb,nw)
    for b in first_block:last_block
        lo = (b-1)*blocksize+1
        status = context.task!(data,lo,min(b*blocksize,n))::Int
        status == 0 || return status
    end
    return 0
end

"""
    foreach_block!(task!, data, n; blocksize, workers=Threads.nthreads(),
                   worker_data=_shared_block_data) -> Int

Apply `task!(data, lo, hi)` to contiguous blocks of `1:n`, with inclusive
bounds. Each logical worker owns a balanced contiguous range of blocks and
receives `worker_data(data, worker)` once. The default shares `data` unchanged.

Tasks return an `Int`: zero on success, otherwise a failure code. A failure
stops that worker; other workers finish independently. Return the first
failure in logical worker order, or zero if all tasks succeed. Expected
failures must use status codes rather than throw across Polyester callbacks.

The runner allocates its own per-worker status vector for parallel execution;
the serial path needs no status storage. Empty input invokes no callbacks.
`n` must be nonnegative; `blocksize` and `workers` must be positive integers.
"""
function foreach_block!(task!::F, data::D, n::Integer;
                        blocksize::Integer, workers::Integer=Threads.nthreads(),
                        worker_data::W=_shared_block_data)::Int where {F,D,W}
    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
    workers > 0 || throw(ArgumentError("workers must be positive"))
    n == 0 && return 0
    nb = cld(n,blocksize)
    nw = min(workers,Threads.nthreads(),nb)
    # Bundle captures to preserve the original array types for BLAS dispatch.
    context = (;task!,data,worker_data)
    nw == 1 && return _foreach_block_worker!(context,1,nb,nw,n,blocksize)

    statuses = zeros(Int,nw)
    @batch for worker in 1:nw
        statuses[worker] = _foreach_block_worker!(context,worker,nb,nw,n,blocksize)
    end
    for status in statuses
        status == 0 || return status
    end
    return 0
end

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
