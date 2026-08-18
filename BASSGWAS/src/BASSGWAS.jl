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

include("utils.jl")
include("ParState.jl")
include("SCache.jl")
include("QCache.jl")
include("likelihoods.jl")
include("bvs_funcs.jl")
export wTGSState2 
include("ProbitSampler.jl")
export ProbitBVSSampler, next_step!, get_state, is_adapted
include("postpred.jl")
export postpred, expand_randeff
include("eig_opt.jl")
export optimize_design
end # module BASSGWAS
