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
    EIGest

Abstract expected-information-gain estimator state. Subtypes implement `eig`
to score candidate additions and `update_eig` to return the updated state.
Candidate positions refer to the ordering established by the estimator.
"""
abstract type EIGest end

"""
    eig(ec::EIGest, idx::Vector{T}) where T<:Integer

Return one total EIG value per candidate position in `idx`, in the supplied
order, for the current design extended by that candidate. These are total
objectives, not incremental gains. Leave the supplied state unchanged.
"""
function eig(eig::EIGest, idx::Vector{T}) where {T <: Integer}
    error("Unimplemented EIG estimator")
end

"""
    update_eig(ec::EIGest, idx::T) where T<:Integer

Return an estimator state with candidate position `idx` added to the current
design, leaving the supplied state unchanged.
"""
function update_eig(eig::EIGest, idx::T) where {T <: Integer}
    error("Unimplemented EIG estimator")
end
