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

# Cache structure for rank 1 updates to the matrix X1'X0
mutable struct QCache{Ti, Tm, Tn}
    i1_set::Ti #active indices
    i0_set::Ti #inactive indices
    X1::Tm
    XtX::Tm 
    XtXb::Tm

    d::Tn #number of active indices
    p::Tn #number of inactive indices
end

function QCache(ps, i1_set, i0_set)
    d = length(i1_set)
    p = length(i0_set)

    X1=ps.Xaug[:, i1_set]
    X=ps.Xaug
    XtX = X1'X

    QCache(i1_set, i0_set, X1, XtX, copy(XtX), d, p)
end

function get_XtX(qc::QCache)
    return qc.XtX
end

function get_X1(qc::QCache)
    return qc.X1
end

function get_X(qc::QCache)
    return qc.X
end

function add_active_QC(qc, ps, index, i1_set_new, i0_set_new)
    d = qc.d
    p = qc.p
    d_new = d+1
    p_new = p-1

    i1_new = searchsortedfirst(i1_set_new, index)
    
    X = ps.Xaug
    qc.X1 = X[:, i1_set_new]
    
    m = size(qc.XtX, 2)

    qc.XtXb = qc.XtX
    qc.XtX = similar(qc.XtXb, d_new, m)

    n1 = size(qc.XtX, 1)
    n2 = size(qc.XtXb, 1)

    copyto!(qc.XtX, 1:(i1_new-1), 1:m, qc.XtXb,  1:(i1_new-1), 1:m)
    @views mul!(qc.XtX[i1_new, :], X', X[:, index])
    copyto!(qc.XtX, (i1_new+1):n1, 1:m, qc.XtXb, (i1_new):n2, 1:m)

    qc.d=d_new
    qc.p=p_new
    qc.i1_set=i1_set_new
    qc.i0_set=i0_set_new
    nothing;
end

function remove_active_QC(qc, ps, index, i1_set_new, i0_set_new)
    d = qc.d
    p = qc.p
    d_new = d-1
    p_new = p+1

    i1_old = searchsortedfirst(qc.i1_set, index)
    
    X = ps.Xaug
    qc.X1 = X[:, i1_set_new]
    
    m = size(qc.XtX, 2)

    qc.XtXb = qc.XtX
    qc.XtX = similar(qc.XtXb, d_new, m)

    n1 = size(qc.XtX, 1)
    n2 = size(qc.XtXb, 1)

    copyto!(qc.XtX, 1:(i1_old-1), 1:m,  qc.XtXb, 1:(i1_old-1), 1:m)
    copyto!(qc.XtX, i1_old:n1,1:m, qc.XtXb, (i1_old+1):n2, 1:m)

    qc.d=d_new
    qc.p=p_new
    qc.i1_set=i1_set_new
    qc.i0_set=i0_set_new
    nothing;
end
