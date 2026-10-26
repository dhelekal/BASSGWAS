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
    QCache(ps, active, inactive)

Active design, raw cross-products and covariance-adjusted cross-products
with storage reserved for an allocated model size. Active indices are sorted and end with the intercept. `X1`
and `XtX` compatibility properties are views of the used portions of backing
storage; they must not be retained across growth in allocated model size or model flips.
`work` owns accepted-model and separate proposal workspaces.
`allocated_model_size` is the number of active columns storage can hold,
including the intercept; `d` is the current model size.

# Fields
- `i1_set`: Sorted active intercept-augmented-design column indices, with the intercept
  last; its length is `d`.
- `i0_set`: Sorted inactive predictor indices; excludes the intercept.
- `xdata`: `n × allocated_model_size` backing matrix whose first `d` columns hold the active
  design `X_A`; exposed as the `X1` view.
- `xtxdata`: `allocated_model_size × paug` backing matrix whose first `d` rows hold
  `X_A' * Xaug`; exposed as the `XtX` view.
- `covdata`: `allocated_model_size × paug` backing matrix whose valid active rows hold
  `X_A' * inv(C) * Xaug` for the recorded covariance source and generation.
- `validrows`: Per-allocated-row validity flags for `covdata`; only the first
  `d` entries are relevant. Inserted rows remain invalid until synchronized.
- `d`: Number of active columns, including the intercept.
- `p`: Number of inactive selectable predictors, equal to `length(i0_set)`.
- `generation`: Active-model change counter, incremented on additions and
  removals to invalidate active factorizations and proposal design gathers.
- `cov_generation`: Covariance generation used to compute the adjusted rows;
  zero before their first synchronization.
- `cov_source`: Identity of the `CovarianceState` supplying those rows, or
  `nothing` before synchronization; distinguishes caches with equal counters.
- `work`: Owned `ActiveWorkspace` with factorization, response-solve,
  candidate-block and independent proposal storage.
"""
mutable struct QCache{T}
    i1_set::Vector{Int}
    i0_set::Vector{Int}
    xdata::Matrix{T}
    xtxdata::Matrix{T}
    covdata::Matrix{T}
    validrows::Vector{Bool}
    d::Int
    p::Int
    generation::UInt
    cov_generation::UInt
    cov_source::Union{Nothing, CovarianceState{T}}
    work::ActiveWorkspace{T}
end

Base.propertynames(qc::QCache, private::Bool = false) = (fieldnames(typeof(qc))..., :X1, :XtX)

@inline function Base.getproperty(qc::QCache, name::Symbol)
    if name === :X1
        return view(getfield(qc, :xdata), :, 1:getfield(qc, :d))
    elseif name === :XtX
        return view(getfield(qc, :xtxdata), 1:getfield(qc, :d), :)
    end
    return getfield(qc, name)
end

get_X1(qc::QCache) = qc.X1
get_XtX(qc::QCache) = qc.XtX

function QCache(ps::ParState2{T}, i1_set, i0_set) where {T}
    d, paug, n, ns = length(i1_set), size(ps.Xaug, 2), size(ps.Xaug, 1), length(ps.S)
    allocated_model_size = max(4, d)
    qc = QCache(
        collect(Int, i1_set), collect(Int, i0_set), zeros(T, n, allocated_model_size),
        zeros(T, allocated_model_size, paug), zeros(T, allocated_model_size, paug), fill(false, allocated_model_size), d, length(i0_set),
        UInt(1), UInt(0), nothing, ActiveWorkspace(T, n, ns, allocated_model_size)
    )

    for j in 1:d
        copyto!(view(qc.xdata, :, j), view(ps.Xaug, :, i1_set[j]))
    end

    mul!(get_XtX(qc), transpose(get_X1(qc)), ps.Xaug)

    return qc
end

function _reserve_active!(qc, needed)
    allocated_model_size = size(qc.xdata, 2)
    needed <= allocated_model_size && return nothing
    allocated_model_size = max(needed, 2allocated_model_size)
    T = eltype(qc.xdata)
    xdata = zeros(T, size(qc.xdata, 1), allocated_model_size)
    xtxdata = zeros(T, allocated_model_size, size(qc.xtxdata, 2))
    covdata = similar(xtxdata)

    copyto!(view(xdata, :, 1:qc.d), get_X1(qc))
    copyto!(view(xtxdata, 1:qc.d, :), get_XtX(qc))
    copyto!(view(covdata, 1:qc.d, :), view(qc.covdata, 1:qc.d, :))

    resize!(qc.validrows, allocated_model_size)
    fill!(view(qc.validrows, (qc.d + 1):allocated_model_size), false)

    qc.xdata, qc.xtxdata, qc.covdata = xdata, xtxdata, covdata
    qc.work = ActiveWorkspace(T, size(xdata, 1), size(qc.work.qactive, 1), allocated_model_size)
    return nothing
end

"""Insert one raw cross-product row; defer its covariance product until synchronization."""
function add_active_QC(qc, ps, index, i1_set_new, i0_set_new)

    old = qc.d
    _reserve_active!(qc, old + 1)
    pos = searchsortedfirst(i1_set_new, index)
    #Move columns past insertion position by one

    @inbounds for j in axes(qc.xtxdata, 2), i in old:-1:pos
        qc.xtxdata[i + 1, j] = qc.xtxdata[i, j]
        qc.covdata[i + 1, j] = qc.covdata[i, j]
    end

    for i in old:-1:pos
        copyto!(view(qc.xdata, :, i + 1), view(qc.xdata, :, i))
        qc.validrows[i + 1] = qc.validrows[i]
    end

    #Insert new column
    copyto!(view(qc.xdata, :, pos), view(ps.Xaug, :, index))

    #Update Column dot products
    _column_dots!(
        view(qc.xtxdata, pos, :),
        ps.Xaug,
        view(ps.Xaug, :, index),
        one(eltype(qc.xtxdata)),
        zero(eltype(qc.xtxdata))
    )

    qc.validrows[pos] = false
    qc.d, qc.p = old + 1, length(i0_set_new)
    qc.i1_set, qc.i0_set = i1_set_new, i0_set_new
    qc.generation += UInt(1)
    return nothing
end

"""Remove one row without recalculating either raw or covariance-adjusted products."""
function remove_active_QC(qc, ps, index, i1_set_new, i0_set_new)
    pos, old = searchsortedfirst(qc.i1_set, index), qc.d

    @inbounds for j in axes(qc.xtxdata, 2), i in pos:(old - 1)
        qc.xtxdata[i, j] = qc.xtxdata[i + 1, j]
        qc.covdata[i, j] = qc.covdata[i + 1, j]
    end

    for i in pos:(old - 1)
        copyto!(view(qc.xdata, :, i), view(qc.xdata, :, i + 1))
        qc.validrows[i] = qc.validrows[i + 1]
    end

    qc.d, qc.p = old - 1, length(i0_set_new)
    qc.i1_set, qc.i0_set = i1_set_new, i0_set_new
    qc.generation += UInt(1)

    return nothing
end
