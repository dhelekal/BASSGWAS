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

`idx_rem` contains the one-based target indices eligible for selection.
`ec` is an initialized estimator whose candidate positions map to `idx_rem`
in the same order used during construction. `batch_sz` must not exceed the
number of candidates.

When `idx_fixed` is nonempty, its target indices are forced into the beginning
of the batch in the supplied order; each fixed index must occur in `idx_rem`
and fixed indices must be distinct. Remaining targets maximize incremental
expected information gain unless `randdes=true`, in which case they are chosen
uniformly using Julia's default RNG. Candidates are removed after selection,
so the result contains no repeated target.

Return `(batch, obj)`, where `batch` contains original target indices and `obj`
is the cumulative expected information gain after the final selection. Inputs
are not mutated.
"""
function optimize_design(ec::EIGest, idx_rem, batch_sz; randdes = false, idx_fixed = Int[])
    n_Y = size(idx_rem, 1)
    idx_set = collect(1:n_Y)

    batch = similar(idx_set, batch_sz)
    n_fixed = length(idx_fixed)
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
