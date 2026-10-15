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
    upd_mlik_arr(mlik, p_success, p_fail, idx)

Extend a Monte Carlo outcome-likelihood table with both binary outcomes for
candidate `idx`.

If `mlik` has one row per posterior draw and `k` outcome-pattern columns, the
returned array has `2k` columns. The first half multiplies by success
probabilities and the second half by failure probabilities. Inputs are not
mutated.
"""
function upd_mlik_arr(mlik, p_success, p_fail, idx)
    n_it = size(mlik, 1)
    k = size(mlik, 2)

    @views y1 = p_success[:, idx]
    @views y0 = p_fail[:, idx]

    mliknew = similar(mlik, n_it, 2*k)

    copyto!(mliknew, 1:n_it, 1:k, mlik, 1:n_it, 1:k)
    @turbo @fastmath for i in 1:k, j in 1:n_it
        @inbounds mliknew[j, i] *= y1[j]
    end

    copyto!(mliknew, 1:n_it, (k+1):2*k, mlik, 1:n_it, 1:k)
    @turbo @fastmath for i in (k+1):2*k, j in 1:n_it
        @inbounds mliknew[j, i] *= y0[j]
    end

    mliknew
end

"""
    comp_mlik(mlik_arr, Z)

Compute the sum of `p * log(p)` over outcome patterns represented by
`mlik_arr`, using weighted posterior draws `Z` to estimate each marginal
pattern probability.
"""
function comp_mlik(mlik_arr, Z)
    n_it = size(mlik_arr, 1)
    n_t = size(mlik_arr, 2)
    psums = similar(mlik_arr, n_t)
    @turbo for i in 1:n_t
        ps = zero(eltype(psums))
        for j in 1:n_it
            @inbounds ps += mlik_arr[j, i] * Z[j]
        end
        psums[i] = ps
    end
    for i in 1:n_t
        @inbounds psums[i] = xlogx(psums[i])
    end
    sum(psums)
end

"""
    comp_mlik_next(tmp, mlik_arr, m1Z)

Compute the marginal `p * log(p)` sum after tentatively adding one binary
design point.

`tmp` is reusable output storage and is overwritten. `m1Z` contains the two
importance-weighted likelihood factors for the candidate.
"""
function comp_mlik_next(tmp, mlik_arr, m1Z)
    # Marginalize posterior draws to estimate each outcome-pattern probability.
    mul!(tmp, mlik_arr', m1Z)
    sl = 0.0
    for i in eachindex(tmp)
        @inbounds sl += xlogx(tmp[i])
    end
    sl
end

"""
    CIDEIG

Immutable Rao–Blackwellized expected-information-gain estimator for
conditionally independent outcomes, implementing `EIGest`.

# Fields
- `p_success`, `p_fail`: Posterior-draw probabilities for each remaining target.
- `mlik_arr`: Joint outcome likelihoods for the currently selected design.
- `cond_ent`: Per-target expected conditional `p * log(p)` contributions.
- `Z`: Weighted-TGS importance weights for posterior draws.
- `mlik1Z`: Precomputed weighted success and failure factors for each target.
- `comb_ent`: Accumulated conditional contribution for the selected design.
- `eig`: Expected information gain of the selected design.
"""
struct CIDEIG{Tx} <: EIGest
    p_success::AbstractMatrix{Tx}
    p_fail::AbstractMatrix{Tx}
    mlik_arr::AbstractMatrix{Tx}
    cond_ent::AbstractVector{Tx}
    Z::AbstractVector{Tx}
    mlik1Z::Array{Tx,3}
    comb_ent::Tx
    eig::Tx
end

"""
    CIDEIG(idx_rem, Ufull, Sfull, X, icept, betas, us, Z)

Initialize expected-information-gain state for the candidate targets in
`idx_rem`.

`idx_rem` contains one-based target indices into the prediction matrices.
`icept`, `betas`, `us`, and `Z` contain aligned posterior draws. Predictive
success and failure probabilities and all one-candidate likelihood factors are
precomputed for subsequent greedy selection.
"""
function CIDEIG(idx_rem::Array{Ty, 1}, Ufull::AbstractMatrix{Tx},
        Sfull::AbstractVector{Tx}, X::AbstractMatrix{Tx},
        icept::AbstractVector{Tx}, betas::AbstractMatrix{Tx},
        us::AbstractMatrix{Tx}, Z::AbstractVector{Tx}) where {Tx<: Real, Ty<:Int}
    lp1, lp0 = postpred(Ufull, Sfull, X, icept, betas, us)

    p1 = exp.(lp1[:,idx_rem])
    p0 = exp.(lp0[:,idx_rem])

    n_it = size(p1, 1)
    n_Y = size(p1, 2)

    mlik_arr = similar(p1, n_it, 1)
    mlik_arr .= 1.0
    mlik1Z = similar(p1, n_it, 2, n_Y)

    for i in 1:n_Y
        @views mlik1Z[:, :, i] = upd_mlik_arr(mlik_arr, p1, p0, i)
        @views mlik1Z[:, :, i] .*= Z
    end

    cond_ent = similar(p1, n_Y)
    for i in 1:n_Y
        s = 0.0
        for j in 1:n_it
            s += Z[j]*(xlogx(p1[j,i]) + xlogx(p0[j,i]))
        end
        cond_ent[i] = s
    end

    CIDEIG(p1, p0, mlik_arr, cond_ent, Z, mlik1Z, 0.0, 0.0)
end

"""
    eig(ec::CIDEIG, idx_set)

Return the total expected information gain after adding each candidate position in
`idx_set` to the design represented by `ec`.

Candidate positions index the columns stored in `ec`, rather than the original
target identifiers supplied to `CIDEIG`.
"""
function eig(ec::CIDEIG, idx_set::Vector{T}) where T<:Integer
    k = size(ec.mlik_arr, 2)
    n_y = size(idx_set,1)

    mlikvals = similar(ec.mlik_arr, n_y)
    clikvals = similar(ec.mlik_arr, n_y)

    nt = Threads.nthreads()
    blck_sz = ceil(Int64,n_y/nt)
    chunks = collect(Iterators.partition(1:n_y, blck_sz))
    tmps = [similar(ec.mlik_arr, k, 2) for _ in chunks]

    @batch for s in eachindex(chunks)
        chunk = chunks[s]
        for i in chunk
            idx = idx_set[i]
            @views mlikvals[i] = comp_mlik_next(tmps[s], ec.mlik_arr, ec.mlik1Z[:,:, idx])
            @views clikvals[i] = ec.comb_ent + ec.cond_ent[idx]
        end
    end

    return clikvals - mlikvals
end

"""
    update_eig(ec::CIDEIG, idx)

Return a new EIG state with candidate position `idx` appended to the selected
design.

The joint likelihood table and cumulative objective are expanded without
mutating `ec`.
"""
function update_eig(ec::CIDEIG, idx::T) where T<:Integer
    mlik_arr_new = upd_mlik_arr(ec.mlik_arr, ec.p_success, ec.p_fail, idx)
    comb_ent_new = ec.comb_ent + ec.cond_ent[idx]
    mliksum = comp_mlik(mlik_arr_new, ec.Z)

    CIDEIG(ec.p_success, ec.p_fail, mlik_arr_new, ec.cond_ent, ec.Z, ec.mlik1Z, comb_ent_new,comb_ent_new-mliksum)
end
