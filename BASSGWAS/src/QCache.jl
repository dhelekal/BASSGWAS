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
    QCache

Cache the active design matrix and cross-products used by predictor-toggle
updates.

The intercept is included in `i1_set` and remains the final active column.

# Fields
- `i1_set`: Sorted indices of active augmented-design columns.
- `i0_set`: Sorted indices of inactive predictor columns.
- `X1`: Active columns of the augmented design matrix.
- `XtX`: Cross-product of `X1` with every augmented-design column.
- `XtXb`: Reusable buffer holding the previous cross-product.
- `d`: Number of active columns, including the intercept.
- `p`: Number of inactive predictors.
"""
mutable struct QCache{Ti, Tm, Tn}
    i1_set::Ti
    i0_set::Ti
    X1::Tm
    XtX::Tm 
    XtXb::Tm

    d::Tn
    p::Tn
end

"""
    QCache(ps, i1_set, i0_set)

Construct a predictor-toggle cache from `ps.Xaug` and the supplied active and
inactive one-based column indices.

`i1_set` must include the intercept index. The cache owns newly allocated
matrix data but retains the supplied index collections.
"""
function QCache(ps, i1_set, i0_set)
    d = length(i1_set)
    p = length(i0_set)

    X1=ps.Xaug[:, i1_set]
    X=ps.Xaug
    XtX = X1'X

    QCache(i1_set, i0_set, X1, XtX, copy(XtX), d, p)
end

"""
    get_XtX(qc::QCache)

Return the cached cross-product between active and augmented-design columns.

The returned matrix aliases storage owned by `qc`.
"""
function get_XtX(qc::QCache)
    return qc.XtX
end

"""
    get_X1(qc::QCache)

Return the cached active design matrix, including the final intercept column.

The returned matrix aliases storage owned by `qc`.
"""
function get_X1(qc::QCache)
    return qc.X1
end

"""
    add_active_QC(qc, ps, index, i1_set_new, i0_set_new)

Update `qc` after activating predictor column `index`.

The supplied index sets describe the post-toggle model. The method inserts one
new cross-product row in sorted active-index order, refreshes `X1`, mutates all
cache metadata, and returns `nothing`.
"""
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
    @views Octavian.matmul!(qc.XtX[i1_new, :], X', X[:, index])
    copyto!(qc.XtX, (i1_new+1):n1, 1:m, qc.XtXb, (i1_new):n2, 1:m)

    qc.d=d_new
    qc.p=p_new
    qc.i1_set=i1_set_new
    qc.i0_set=i0_set_new
    nothing;
end

"""
    remove_active_QC(qc, ps, index, i1_set_new, i0_set_new)

Update `qc` after deactivating predictor column `index`.

The supplied index sets describe the post-toggle model. The method removes the
corresponding cross-product row, refreshes `X1`, mutates all cache metadata,
and returns `nothing`.
"""
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
