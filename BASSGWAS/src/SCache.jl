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
    _comp_d!(d, S, r, s)

Fill `d` with the diagonal Woodbury factors for covariance eigenvalues `S`
and positive variance multipliers `r` and `s`.
"""
function _comp_d!(d, S, r, s)
    @turbo for i in axes(d,1)
        d[i] = 1.0/s/sqrt(1.0/(S[i]*r) + 1.0/s)
    end
end

"""
    makeSCache(ps::ParState2)

Precompute low-rank covariance and response terms for the parameter state
`ps`.

The cache represents operations involving `r * U * Diagonal(S) * U' + s * I`
without materializing its dense inverse. Predictor and response dimensions
must agree with those stored in `ps`.
"""
function makeSCache(ps::ParState2{Tx, Ty, Tc, Tn, Tr}) where {Tx, Ty, Tc, Tn, Tr}
    @unpack Xaug, Y, U, S, p, n, alpha, beta, c, r, s = ps

    ncol = size(Y, 1)
    ns = size(S, 1)
    
    V = U*Diagonal(sqrt.(S))
    UtX = U' * Xaug
    YtU = Y' * U
    YtX = Y' * Xaug
    sqX = zeros(p+1);
    _vpacnormsq!(sqX, 1.0, Xaug, Xaug);

    d = similar(S)
    _comp_d!(d, S, r, s)
    D = Diagonal(d)

    F = 1.0/s * Y' - YtU*D*D'U'
    QX = D*UtX
    FX = -YtU*D*QX
    FX .+= 1.0/s*YtX

    qterm = dot(Y,F)
    cdet = -2.0*(logabsdet(D)[1]+ns*log(s)) + sum(log.(abs.(S))) + ns*log(r) + log(s)*ncol

    return SCache(S, U, V, UtX, YtU, YtX, sqX, d, F, QX, FX, cdet, qterm, p+1, ncol, ns)
end

"""
    updateSCache!(cache, ps)

Refresh response- and variance-dependent entries of `cache` from `ps`.

The design matrix and covariance eigensystem are assumed unchanged. All
updates are performed in place.
"""
function updateSCache!(cache, ps)
    @unpack Xaug, Y, p, n, alpha, beta, c, r, s = ps
    d = cache.d 
    S = cache.S
    U = cache.U
    UtX=cache.UtX
    YtU=cache.YtU
    YtX=cache.YtX

    ns = cache.ns
    n = cache.n

    _comp_d!(d, S, r, s)
    D = Diagonal(d)
    D2 = D*D

    sinv = 1.0/s

    Octavian.matmul!(YtX, Y', Xaug)
    Octavian.matmul!(YtU, Y', U)

    YD = YtU * D2
    copy!(cache.F, Y')
    Octavian.matmul!(cache.F', U, YD', -1.0, sinv)
    
    copy!(cache.FX, YtX)
    Octavian.matmul!(cache.FX', UtX', YD', -1.0, sinv)

    # `lmul!` lacks a three-argument form, so copy before scaling in place.
    copy!(cache.QX, UtX)
    lmul!(D, cache.QX)

    cache.cdet = -2.0*(logabsdet(D)[1]+ns*log(s))+ ns*log(r) + log(s)*n
    for i in eachindex(S)
         cache.cdet += log(abs(S[i]))
    end

    cache.qterm = dot(Y,cache.F)
end

"""
    SCache

Cache low-rank covariance terms used in collapsed likelihoods and rank-one
predictor updates.

`U` and `S` are assumed to be a possibly truncated eigendecomposition of a
positive-semidefinite covariance matrix, with positive `r` and `s` supplied
through the associated parameter state.

# Fields
- `S`, `U`: Covariance eigenvalues and eigenvectors.
- `V`: Covariance square-root factor `U * Diagonal(sqrt.(S))`.
- `UtX`, `YtU`, `YtX`: Cached design and response cross-products.
- `sqX`: Squared Euclidean norm of each augmented-design column.
- `d`: Diagonal Woodbury factors.
- `F`: Product of the transposed response with the inverse base covariance.
- `QX`: Scaled eigenvector-design cross-product.
- `FX`: Product of `F` with the augmented design.
- `cdet`: Cached covariance log-determinant contribution.
- `qterm`: Cached response quadratic-form contribution.
- `paug`, `n`, `ns`: Augmented predictor, response, and retained-rank sizes.
"""
mutable struct SCache{Tv, Tm, Ta, Tr, Tn}
    S::Tv
    U::Tm
    V::Tm
    UtX::Tm    
    YtU::Ta
    YtX::Ta
    sqX::Tv
    
    d::Tv
    F::Ta
    QX::Tm
    
    FX::Ta
    
    cdet::Tr
    qterm::Tr
    paug::Tn
    n::Tn
    ns::Tn
end
