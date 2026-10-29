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
    optimize_design(ec::EIGest, idx_rem, batch_sz;
                    randdes=false, idx_fixed=Int[])

Select a batch of distinct experimental targets by greedy expected information
gain.

`idx_rem` contains distinct one-based target indices eligible for selection.
`Ufull`, `Sfull`, and `X` describe every prediction target; `icept`, `betas`,
and `us` contain aligned posterior draws; and `Z` contains their weighted-TGS
importance weights. `batch_sz` must be a nonnegative integer no greater than
the number of candidates. A zero-size batch returns an empty batch and zero EIG.

When `idx_fixed` is nonempty, its target indices are forced into the beginning
of the batch in the supplied order; each fixed index must occur in `idx_rem`
and fixed indices must be distinct. Their count must not exceed `batch_sz`.
Invalid selection constraints raise `ArgumentError` before EIG computation.
Remaining targets maximize incremental
expected information gain unless `randdes=true`, in which case they are chosen
uniformly using Julia's default RNG. Candidates are removed after selection,
so the result contains no repeated target.

Return `(batch, obj)`, where `batch` contains original target indices and `obj`
is the cumulative expected information gain after the final selection. Inputs
are not mutated.
"""
function optimize_design(ec::EIGest, idx_rem, batch_sz; randdes = false, idx_fixed = Int[])
    n_Y = size(idx_rem, 1)
    batch_sz isa Integer && 0 <= batch_sz <= n_Y ||
        throw(ArgumentError("batch_sz must be a nonnegative integer no greater than the number of candidates ($n_Y)"))
    all(i -> i isa Integer && 1 <= i <= size(X, 1), idx_rem) ||
        throw(ArgumentError("candidate indices must be integers between 1 and $(size(X, 1))"))
    allunique(idx_rem) || throw(ArgumentError("candidate indices must be distinct"))
    n_fixed = length(idx_fixed)
    n_fixed <= batch_sz || throw(ArgumentError("the number of fixed targets ($n_fixed) must not exceed batch_sz ($batch_sz)"))
    allunique(idx_fixed) || throw(ArgumentError("fixed target indices must be distinct"))
    eligible = Set(idx_rem)
    all(i -> i isa Integer && i in eligible, idx_fixed) ||
        throw(ArgumentError("every fixed target index must be an eligible candidate in idx_rem"))
    batch_sz == 0 && return (batch = idx_rem[Int[]], obj = 0.0)

    idx_set = collect(1:n_Y)

    batch = similar(idx_set, batch_sz)
    e = 0.0
    for i in 1:batch_sz
        obj = eig(ec, idx_set)

        maxi = 0
        if i <= n_fixed
            maxi = findfirst(idx_fixed[i] .== idx_rem[idx_set])
        elseif randdes
            maxi = rand(1:size(idx_set, 1))
        else
            maxi = findmax(obj)[2]
        end

        maxind = idx_set[maxi]
        batch[i] = maxind
        e = obj[maxi]

        # Remove the target so a batch cannot contain biological replicates.
        deleteat!(idx_set, maxi)
        ec = update_eig(ec, maxind)
    end
    return (batch = idx_rem[batch], obj = e)
end


"""
    comp_eig(ec::EIGest, candidate_positions::Vector{T}) where T<:Integer

Evaluate the total expected information gain after adding a fixed design to
`ec`, processing `candidate_positions` in the supplied order.

Positions refer to candidates stored in the estimator, not original target
identifiers. Scoring and state updates dispatch through `eig` and `update_eig`.
Return the final scalar EIG, or `0.0` for an empty design.
"""
function comp_eig(ec::EIGest, candidate_positions::Vector{T}) where {T <: Integer}
    n_Y = size(candidate_positions, 1)
    e = 0.0
    for i in 1:n_Y
        obj = eig(ec, candidate_positions)
        e = obj[i]
        ec = update_eig(ec, candidate_positions[i])
    end
    return e
end
