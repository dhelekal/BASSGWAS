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

Capacity-managed active design, raw cross-products and covariance-adjusted
cross-products. Active indices are sorted and end with the intercept. `X1`
and `XtX` compatibility properties are views of the used portions of backing
storage; they must not be retained across capacity growth or model flips.
`work` owns accepted-model and separate proposal workspaces.
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
    cov_source::Union{Nothing,CovarianceState{T}}
    work::ActiveWorkspace{T}
end

Base.propertynames(qc::QCache, private::Bool=false) = (fieldnames(typeof(qc))..., :X1, :XtX)

@inline function Base.getproperty(qc::QCache, name::Symbol)
    if name === :X1
        return view(getfield(qc,:xdata),:,1:getfield(qc,:d))
    elseif name === :XtX
        return view(getfield(qc,:xtxdata),1:getfield(qc,:d),:)
    end
    return getfield(qc,name)
end

get_X1(qc::QCache) = qc.X1
get_XtX(qc::QCache) = qc.XtX

function QCache(ps, i1_set, i0_set)
    T = promote_type(float(eltype(ps.Xaug)),float(eltype(ps.U)),float(eltype(ps.S)),float(eltype(ps.Y)),typeof(ps.r),typeof(ps.s))
    a, P, n, k = length(i1_set),size(ps.Xaug,2),size(ps.Xaug,1),length(ps.S)
    capacity = max(4,a)
    qc = QCache(collect(Int,i1_set),collect(Int,i0_set),zeros(T,n,capacity),
        zeros(T,capacity,P),zeros(T,capacity,P),fill(false,capacity),a,length(i0_set),
        UInt(1),UInt(0),nothing,ActiveWorkspace(T,n,k,capacity))
    for j in 1:a
        copyto!(view(qc.xdata,:,j),view(ps.Xaug,:,i1_set[j]))
    end
    mul!(get_XtX(qc),transpose(get_X1(qc)),ps.Xaug)
    return qc
end

function _reserve_active!(qc, needed)
    capacity = size(qc.xdata,2)
    needed <= capacity && return nothing
    capacity = max(needed,2capacity)
    T = eltype(qc.xdata)
    x = zeros(T,size(qc.xdata,1),capacity)
    xtx = zeros(T,capacity,size(qc.xtxdata,2))
    cov = similar(xtx)
    copyto!(view(x,:,1:qc.d),get_X1(qc))
    copyto!(view(xtx,1:qc.d,:),get_XtX(qc))
    copyto!(view(cov,1:qc.d,:),view(qc.covdata,1:qc.d,:))
    resize!(qc.validrows,capacity)
    fill!(view(qc.validrows,(qc.d+1):capacity),false)
    qc.xdata, qc.xtxdata, qc.covdata = x,xtx,cov
    qc.work = ActiveWorkspace(T,size(x,1),size(qc.work.qactive,1),capacity)
    return nothing
end

"""Insert one raw cross-product row; defer its covariance product until synchronization."""
function add_active_QC(qc, ps, index, i1_set_new, i0_set_new)
    old = qc.d
    _reserve_active!(qc,old+1)
    pos = searchsortedfirst(i1_set_new,index)
    @inbounds for j in axes(qc.xtxdata,2), i in old:-1:pos
        qc.xtxdata[i+1,j] = qc.xtxdata[i,j]
        qc.covdata[i+1,j] = qc.covdata[i,j]
    end
    for i in old:-1:pos
        copyto!(view(qc.xdata,:,i+1),view(qc.xdata,:,i))
        qc.validrows[i+1] = qc.validrows[i]
    end
    copyto!(view(qc.xdata,:,pos),view(ps.Xaug,:,index))
    _column_dots!(view(qc.xtxdata,pos,:),ps.Xaug,view(ps.Xaug,:,index),one(eltype(qc.xtxdata)),zero(eltype(qc.xtxdata)))
    qc.validrows[pos] = false
    qc.d, qc.p = old+1,length(i0_set_new)
    qc.i1_set, qc.i0_set = i1_set_new,i0_set_new
    qc.generation += UInt(1)
    return nothing
end

"""Remove one row without recalculating either raw or covariance-adjusted products."""
function remove_active_QC(qc, ps, index, i1_set_new, i0_set_new)
    pos, old = searchsortedfirst(qc.i1_set,index),qc.d
    @inbounds for j in axes(qc.xtxdata,2), i in pos:(old-1)
        qc.xtxdata[i,j] = qc.xtxdata[i+1,j]
        qc.covdata[i,j] = qc.covdata[i+1,j]
    end
    for i in pos:(old-1)
        copyto!(view(qc.xdata,:,i),view(qc.xdata,:,i+1))
        qc.validrows[i] = qc.validrows[i+1]
    end
    qc.d, qc.p = old-1,length(i0_set_new)
    qc.i1_set, qc.i0_set = i1_set_new,i0_set_new
    qc.generation += UInt(1)
    return nothing
end
