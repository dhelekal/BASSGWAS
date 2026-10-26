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

@inline function _positive_variance(x, name)
    isfinite(x) && x > 0 || throw(ArgumentError("$name must be finite and positive"))
    return nothing
end

# Promote first so integer inputs do not unnecessarily widen floating inputs.
function _state_float_type(types...)
    T = float(promote_type(types...))
    T <: AbstractFloat && isconcretetype(T) ||
        throw(ArgumentError("state inputs must promote to a concrete floating-point type"))
    return T
end

function _foreach_block_worker!(context, worker, nb, nw, n, blocksize)::Int
    
    first_block = fld((worker-1)*nb,nw)+1
    last_block = fld(worker*nb,nw)
    
    for b in first_block:last_block
        lo = (b-1)*blocksize+1
        status = context.task!(context.data,lo,min(b*blocksize,n),worker)::Int
        status == 0 || return status
    end

    return 0
end

"""
    foreach_block!(task!, data, n; blocksize, workers=Threads.nthreads()) -> Int

Apply `task!(data, lo, hi, worker)` to contiguous blocks of `1:n`, with
inclusive bounds and `data` passed unchanged. Each logical worker owns a
balanced contiguous range of blocks. Its index is stable across those blocks
and ranges from one to the number of participating workers, independently of
`Threads.threadid()`. Serial execution passes `worker = 1`.

Tasks return an `Int`: zero on success, otherwise a failure code. A failure
stops that worker; other workers finish independently. Return the first
failure in logical worker order, or zero if all tasks succeed. Expected
failures must use status codes rather than throw across Polyester callbacks.

The runner allocates its own per-worker status vector for parallel execution;
the serial path needs no status storage. Empty input invokes no callbacks.
`n` must be nonnegative; `blocksize` and `workers` must be positive integers.
"""
function foreach_block!(task!::F, data::D, n::Integer;
                        blocksize::Integer, workers::Integer=Threads.nthreads())::Int where {F,D}

    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
    workers > 0 || throw(ArgumentError("workers must be positive"))

    n == 0 && return 0
    nb = cld(n,blocksize)
    nw = min(workers,Threads.nthreads(),nb)

    # Bundle captures to preserve the original array types for BLAS dispatch.
    context = (;task!,data)
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

# Column dots are memory-bound for large predictor sets. Each worker reads
# contiguous columns and owns disjoint outputs, including strided active rows.
function _column_dots_block!(data,lo,hi,worker)
    (;out,A,x,scale_product,scale_previous) = data
    mul!(view(out,lo:hi), transpose(view(A,:,lo:hi)), x,
        scale_product, scale_previous)

    return 0
end

function _column_dots!(out,A,x,scale_product,scale_previous)

    P = size(A,2)

    threaded = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1 && length(A) >= CACHE_THREAD_WORK

    foreach_block!(_column_dots_block!, (;out,A,x,scale_product,scale_previous), P;
        blocksize=threaded ? CACHE_BLOCK_COLUMNS : max(P,1),
        workers=threaded ? Threads.nthreads() : 1)

    return nothing
end
