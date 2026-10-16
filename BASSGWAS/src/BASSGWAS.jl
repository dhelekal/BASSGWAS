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
    BASSGWAS

Bayesian adaptive sampling and variable selection for binary responses.

The package combines a weighted tempered Gibbs sampler for Bayesian variable
selection under a probit mixed model with posterior-predictive utilities and a
greedy expected-information-gain criterion for choosing experimental designs.
Low-rank eigendecompositions represent correlated random effects throughout
the sampling and prediction workflows.
"""
module BASSGWAS

using LinearAlgebra
using LinearAlgebra.BLAS
using SpecialFunctions
using Distributions
using LogExpFunctions
using LoopVectorization
using Octavian
using Polyester
using Random
using StatsBase
using UnPack

# Internal cache-kernel tuning parameters.
const CACHE_BLOCK_COLUMNS = 256
const CACHE_THREAD_WORK = 32768

include("utils.jl")
include("ParState.jl")
include("SCache.jl")
include("ActiveWorkspace.jl")
include("QCache.jl")
include("likelihoods.jl")
include("bvs_funcs.jl")
export wTGSState2
include("ProbitSampler.jl")
export ProbitBVSSampler, next_step!, get_state, is_adapted
include("postpred.jl")
export postpred, expand_randeff
include("eig.jl")
include("raoblackwell_eig.jl")
include("mlmc_eig.jl")
export EIGest, CIDEIG, MLMCEig, eig, update_eig
include("eig_opt.jl")
export optimize_design
end
