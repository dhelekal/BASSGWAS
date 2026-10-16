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

#MLMC EIG estimator constructed using M posterior draws and N candidate observations
struct MLMCEig{T<:AbstractFloat} <: EIGest
    lp_success::Matrix{T} #posterior draws of log prob of y=1
    lp_fail::Matrix{T} #posterior draws of log prob of y=0
    outcomes::Matrix{T} #MxN posterior outcome predictions
    Z::Vector{T} #posterior importance weights normalized to sum to M
end

#Compute an MLMC inner expectation for a given level l, i.e. E[P_{l} - P_{l-1}] using antithetic couplings
#outcome - the outcome realization to be used in the inner expectation
#design_idx - the index of design variables corresponding to outcomes
#draw_idx - the index of posterior draws allocated to this expectation i.e. the index of rows in ec. The number must be equal to 2^l
#return E[P_{l} - P_{l-1}] 
function mlmc_inner(outcome::AbstractVector, design_idx::AbstractVector, draw_idx::AbstractVector,  level::Integer, ec::MLMCEig)

end

#For a given number of levels and empirical estimates of variances V_l, compute the number of inner draws per level l, M_l, using the heuristic M_l \propto sqrt(V_l/C_l)
#Note that C_l is the cost of an inner expectation at level l, equal to 2^l + 1 as one draw gets consumed for the outer realization
#The allocated draws must sum not exceed the total number of posterior draws
#return a vector with entries M_l
function allocate_draws(levels::Integer, variances::AbstractVector, ec::MLMCEig)

end

#Given a draw allocation vector M_l, partition the draws in ec into non-overlapping partitions.
#Each partition specifies the posterior rows to be used to compute an mlmc inner expectation
#The total number of partitions must be equal to sum(allocations)
#Return a collection mapping (level, realization) -> draw_idx
function partition_draws(allocations::AbstractVector, ec::MLMCEig)

end


"""Return total MLMC EIG for each candidate addition, without changing `ec`."""
function eig(ec::MLMCEig{T}, idx_set::Vector{I}) where {T,I<:Integer}

end

"""Return a new state with candidate `idx` selected and its MLMC diagnostics."""
function update_eig(ec::MLMCEig, idx::I) where I<:Integer

end
