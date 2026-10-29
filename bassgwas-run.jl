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

# Main command-line workflow: prepare input data, adapt xi with a short chain,
# sample parallel chains, save posterior draws, and select the next strain batch
# by expected information gain (EIG) when a positive batch size is requested.
using LinearAlgebra
using MCMCChains
using DataFrames
using CSV
using Distributed
using UnPack
using ArgParse
using Optim
using Distributions

# --inclnext is a headerless text file with one literal isolate identifier per line.
function read_fixed_indices(path, sample_ids)
    names = string.(sample_ids)
    indices = Int[]
    for name in readlines(path)
        index = findfirst(==(name), names)
        isnothing(index) && throw(ArgumentError("Unknown fixed isolate: $(repr(name))"))
        push!(indices, index)
    end
    return indices
end

# Define input files, sampling controls, model-size prior targets, and design options.
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--pfile"
        required = true
        arg_type = String
        help = "Predictor matrix file"
        "--ufile"
        required = true
        arg_type = String
        help = "Population structure matrix file"
        "--sfile"
        required = true
        arg_type = String
        help = "Population structure loadings file"
        "--obsfile"
        required = true
        arg_type = String
        help = "Observations file"
        "--odir"
        required = true
        arg_type = String
        help = "Output directory"
        "--batchsz"
        required = true
        arg_type = Int
        help = "Design batch size"
        "--inclnext"
        arg_type = String
        help = "File of sample names to include in the next batch"
        "--seed"
        arg_type = Int
        default = 0
        help = "Rng seed"
        "--ndraws"
        arg_type = Int
        default = 1000
        help = "Number of samples retained per chain"
        "--nthin"
        arg_type = Int
        default = 500
        help = "Thinning factor per sample"
        "--nchains"
        arg_type = Int
        default = 1
        help = "Number of parallel chains to use"
        "--p_mean"
        arg_type = Float64
        default = 5.0
        help = "model size prior mean conditional on n>0"
        "--pr_0"
        arg_type = Float64
        default = 0.25
        help = "model size prior probability of n=0"
        "--randdes"
        help = "An option to construct a random design instead of an optimized one"
        action = :store_true
    end

    s.description = "Run BASSGWAS sampling and (optionally) optimise the next experimental design."

    return parse_args(s)
end

# Pool draws across chains while preserving alignment between sampled quantities.
# Input axes are draw, parameter, chain; output rows group draws by chain.
# Scalar quantities arrive as draw-by-chain matrices and become vectors.
function flatten_chains(x)
    if length(size(x)) == 2
        x_flat = reshape(x, :)
    elseif length(size(x)) == 3
        x_flat = reshape(permutedims(x, (1, 3, 2)), :, size(x, 2))
    else
        throw(ErrorException("x must be a 2 or 3 dimensional array"))
    end
    return x_flat
end

# Restore pooled rows to draw-by-parameter-by-chain storage.
function expand_chains(x, nchains, ndraws)
    return permutedims(reshape(x, ndraws, nchains, :), (1, 3, 2))
end

# Read the command-line configuration used by every workflow stage.
parsed = parse_commandline()

pfile = parsed["pfile"]
ufile = parsed["ufile"]
sfile = parsed["sfile"]
obsfile = parsed["obsfile"]
outdir = parsed["odir"]
batch_sz = parsed["batchsz"]
seed = parsed["seed"]
randdes = parsed["randdes"]

ndraws = parsed["ndraws"]
nthin = parsed["nthin"]
nchains = parsed["nchains"]
pmean = parsed["p_mean"]
pr0 = parsed["pr_0"]
incl_next = parsed["inclnext"] !== nothing

# Save the input paths and run settings in receipt.txt for reference.
receiptfile = outdir * "/receipt.txt"
open(receiptfile, "w") do f

    println(f, "")
    println(f, "--- Adaptive sampling for genome wide prediction ---")
    println(f, "")
    println(f, " -- Input Files --")
    println(f, "  - Predictor matrix: $(pfile)")
    println(f, "  - U matrix:         $(ufile)")
    println(f, "  - S matrix:         $(sfile)")
    println(f, "  - Observation file: $(obsfile)")
    println(f, "")
    println(f, " Output Directory: $(outdir)")
    println(f, "")
    println(f, " -- Model Parameters --")
    println(f, "  - Prior model size conditional mean $(pmean)")
    println(f, "  - Prior model size null probability $(pr0)")
    println(f, "")
    println(f, " -- Run Specification --")
    println(f, "  - Using seed:       $(seed)")
    println(f, "  - Draws per chain:  $(ndraws)")
    println(f, "  - Number of chains: $(nchains)")
    println(f, "  - Thinning factor:  $(nthin)")
    println(f, "")
    println(f, " -- Design Parameters --")
    println(f, "  - Batch size:               $(batch_sz)")
    println(f, "  - Fixed samples in design?: $(incl_next)")
    println(f, "    Fixed sample file:        $(incl_next ? parsed["inclnext"] : "")")
    println(f, "  - Random design:            $(randdes)")
    println(f, "")

end

# Start one worker process per chain and load the sampler on every process.
println("Adding worker processes ...")
addprocs(nchains, exeflags = "--threads=$(Threads.nthreads())")
println("Done")

@everywhere begin
    using BASSGWAS
    using Random
    using UnPack
end

# Seed the main process for xi adaptation, chain seeds, and experimental design.
# Limit BLAS threading to avoid competing with parallel chain workers.
Random.seed!(seed)
BLAS.set_num_threads(1)
tol = 1.0e-8

# Stage 1: read the user-supplied CSV files and prepare consistent model inputs.
# The predictor table contains strain identifiers followed by variant columns.
# U and S describe population structure and must follow the predictor row order;
# the observation table contains strain identifiers and binary phenotypes.
T = Float64
data = DataFrame(CSV.File(pfile));
U = Array{T}(DataFrame(CSV.File(ufile))[:, 1:end]);
S = reshape(Array{T}(DataFrame(CSV.File(sfile))[:, 1:end]), :);
obsU = DataFrame(CSV.File(obsfile))

# Reorder phenotypes to match predictor rows and locate the observed strains.
obs = obsU[sortperm([findfirst(j .== data[:, 1]) for j in obsU[:, 1]]), :]
idx_obs = findall(in.(data[:, 1], Ref(obs[:, 1])))

# Map any strains required in the next batch to their full-dataset row indices.
idx_fixed = similar(idx_obs, 0)
if incl_next
    idx_fixed = read_fixed_indices(parsed["inclnext"], data[:, 1])
end

# Restrict the population covariance to observed strains and recompute its basis.
# Discard near-zero singular values so sampling uses the supported covariance modes.
Ur = U[idx_obs, :]
Cs = Ur * Diagonal(S) * Ur'
sv = svd(Cs)

nnz = findall(sv.S .> tol)
So = sv.S[nnz]
Uo = sv.U[:, nnz]
# Keep predictors for all strains for design, and build numeric sampling arrays
# containing only observed strains with phenotypes in the same row order.
X = Array{T}(data[:, 2:end]);
Xo = Array{T}(data[idx_obs, 2:end]);
Yo = reshape(Array{Int64}(obs[:, 2]), :)

# Calibrate the beta-binomial model-size prior to the requested null probability
# and expected number of included variants conditional on a nonempty model.
p = size(Xo, 2)

# Optimize on the log scale to keep both beta shape parameters positive.
function bbinom_obj(x, n, pr0, mu)
    v = pdf(BetaBinomial(n, exp(x[1]), exp(x[2])))
    v0 = v[1]
    lp0 = log(pr0 / (1.0 - pr0))
    lv0 = log(v0 / (1.0 - v0))

    cmu = sum(v[2:end] ./ (1.0 - v0) .* collect(1:n))
    return (lp0 - lv0)^2.0 + (mu - cmu)^2.0
end

sol = optimize(
    x -> bbinom_obj(x, p, pr0, pmean),
    [-3.0, -3.0],
    NelderMead(),
    Optim.Options(g_tol = 1.0e-12)
)

alpha = exp(sol.minimizer[1])
beta = exp(sol.minimizer[2])

# Stop if the fitted prior does not match the requested targets closely enough.
sol.minimum > tol &&
    error("Failed to calibrate model size prior. Objective: $(sol.minimum)")

# Stage 2: run one short chain to estimate the auxiliary intensity xi, which
# controls the frequency of hyperparameter updates relative to variant toggles.
# Retain only its adapted xi value for use by every production chain.
println("Adapting xi ...")
xi = let samp = ProbitBVSSampler(Xo, Yo, Uo, So, alpha, beta; nadapt = 50 * nthin)
    while !is_adapted(samp)
        next_step!(samp)
    end
    samp.wTGSState.xi
end
println("Found xi=$(xi)")
# Stage 3: define the production sampler on every worker process.
@everywhere begin
    using LinearAlgebra.BLAS
    BLAS.set_num_threads(1)
    function run_sampling(x, y, u, s, xi, seed, K, thin, alpha, beta)
        p = size(x, 2)
        n = size(x, 1)
        rng = Xoshiro(seed)

        # Give each chain its own seeded RNG and discard its initial warm-up steps.
        _burnin = 100 * thin
        _it = K * thin

        # Allocate storage for K retained draws of variant indicators, inclusion
        # probabilities, scalar parameters, latent liabilities, and coefficients.
        gamma_d = zeros(Int16, K, p)
        pips_d = zeros(Float64, K, p)
        cpars_d = zeros(Float64, K, 6)
        Ys_d = zeros(Float64, K, n)
        betas_d = zeros(Float64, K, p)
        us_d = zeros(Float64, K, size(s, 1))

        # Hold xi fixed at the estimate from the short preliminary chain.
        samp = ProbitBVSSampler(x, y, u, s, alpha, beta; nadapt = _burnin, xi_init = xi, adapt_xi = false, rng = rng)
        for _ in 1:_burnin
            next_step!(samp)
        end

        # After warm-up, retain one state every thin transitions until K are saved.
        for i in 1:_it
            next_step!(samp)
            if i % thin == 0
                j = trunc(Int64, i / thin)

                @unpack γ, pips, icept, beta, u, Y, sigma, r, c, Z = get_state(samp)
                gamma_d[j, :] .= γ
                pips_d[j, :] .= pips
                betas_d[j, :] .= beta
                us_d[j, :] .= u
                Ys_d[j, :] .= Y

                # Store model size, sigma, r, c, intercept, and importance weight Z
                # in the same column order used for diagnostics and CSV export.
                cpars_d[j, 1] = sum(γ)
                cpars_d[j, 2] = sigma
                cpars_d[j, 3] = r
                cpars_d[j, 4] = c
                cpars_d[j, 5] = icept
                cpars_d[j, 6] = Z
            end
        end
        return (gammas = gamma_d, pips = pips_d, cpars = cpars_d, Ys = Ys_d, betas = betas_d, us = us_d)
    end
end
# Generate reproducible chain seeds and distribute the chains across workers.
seeds = rand(1:100000000000, nchains)
println("Running parallel chains ...")
draws = pmap(1:nchains) do i
    run_sampling(Xo, Yo, Uo, So, xi, seeds[i], ndraws, nthin, alpha, beta)
end
println("Done")

# Stage 4: combine chain results into draw-by-parameter-by-chain arrays for
# diagnostics, CSV export, and the subsequent experimental-design calculation.
println("Processing draws ...")
gammas_draws = zeros(Int64, (size(draws[1].gammas)..., nchains))
betas_draws = zeros(Float64, (size(draws[1].betas)..., nchains))
pips_draws = zeros(Float64, (size(draws[1].pips)..., nchains))
Ys_draws = zeros(Float64, (size(draws[1].Ys)..., nchains))
us_draws = zeros(Float64, (size(draws[1].us)..., nchains))
cpars_draws = zeros(Float64, (size(draws[1].cpars)..., nchains))

for i in 1:nchains
    @unpack gammas, pips, cpars, Ys, betas, us = draws[i]
    betas_draws[:, :, i] .= betas
    gammas_draws[:, :, i] .= gammas
    pips_draws[:, :, i] .= pips
    us_draws[:, :, i] .= us
    Ys_draws[:, :, i] .= Ys
    cpars_draws[:, :, i] .= cpars
end
draws = nothing

# Pool aligned random-effect and variance draws, then conditionally expand the
# random-effect coefficients from the observed-strain basis to the full basis.
# These expanded draws support prediction for strains without phenotypes.
usf = flatten_chains(us_draws)
rsf = flatten_chains(cpars_draws[:, 3, :])

usf_full = expand_randeff(Uo, So, U, S, idx_obs, usf, rsf, Random.default_rng())
us_full_draws = permutedims(reshape(usf_full, ndraws, nchains, :), (1, 3, 2))
!isapprox(usf_full[1:ndraws, :], us_full_draws[1:ndraws, :, 1]) && throw(ErrorException("us dont match"))

# Report diagnostics for the tempered sampling distribution before saving draws.
# The design stage uses Z to reweight these draws for posterior expectations.
cchn = Chains(cpars_draws, [:n, :sigma, :r, :c, :icept, :Z])
println("Tempered distribution diagnostics:")
show(stdout, "text/plain", summarystats(cchn))
println("Saving draws ...")

# Save scalar parameter draws, including model size and importance weights.
CSV.write(outdir * "/par_draws.csv", DataFrame(cchn))

# Save latent liability draws for observed strains.
let chn = Chains(Ys_draws)
    CSV.write(outdir * "/Y_draws.csv", DataFrame(chn))
end

# Save random-effect coefficient draws in the full population basis.
let chn = Chains(us_full_draws)
    CSV.write(outdir * "/u_draws.csv", DataFrame(chn))
end

# Save regression coefficient draws for all variants.
let chn = Chains(betas_draws)
    CSV.write(outdir * "/b_draws.csv", DataFrame(chn))
end

# Save per-draw conditional variant inclusion probabilities.
let chn = Chains(pips_draws)
    CSV.write(outdir * "/pip_draws.csv", DataFrame(chn))
end

# Save sampled binary variant inclusion indicators.
let chn = Chains(gammas_draws)
    CSV.write(outdir * "/gamma_draws.csv", DataFrame(chn))
end
println("Done")


# Stage 5: select strains for the next round of experimental phenotyping.
println("Designing next experiment ...")
# A zero batch size requests sampling and CSV export without experimental design.
if batch_sz > 0

    # Pool coefficient and intercept draws in the same order as expanded random
    # effects, checking that the first chain retains its original draw ordering.
    betasf = flatten_chains(betas_draws)
    !isapprox(betasf[1:ndraws, :], betas_draws[1:ndraws, :, 1]) && throw(ErrorException("betas dont match"))

    iceptf = flatten_chains(cpars_draws[:, 5, :])
    !isapprox(iceptf[1:ndraws, :], cpars_draws[1:ndraws, 5, 1]) && throw(ErrorException("icept doesnt match"))

    # Normalize the aligned importance weights for posterior expectations in EIG.
    zz = flatten_chains(cpars_draws[:, 6, :])
    !isapprox(zz[1:ndraws, :], cpars_draws[1:ndraws, 6, 1]) && throw(ErrorException("Zs dont match"))
    Zs = (zz ./ sum(zz))

    # Only strains without observed phenotypes are eligible for the next batch.
    n_samp = size(data[:, 1], 1)
    idx_rem = (1:n_samp)[1:n_samp .∉ Ref(idx_obs)]

    # Build the batch greedily by EIG, honoring any required strains first.
    # With --randdes, choose the remaining strains randomly instead.
    design = optimize_design(idx_rem, U, S, X, iceptf, betasf, usf_full, Zs, batch_sz; randdes = randdes, idx_fixed = idx_fixed)
    # Write selected strain identifiers in dataset order, one per line in design.txt.
    selected = data[sort(design[1]), 1]

    println("Design objective value: $(design[2])")
    println("Writing design ...")
    outfile = outdir * "/design.txt"
    open(outfile, "w") do f
        for i in selected
            println(f, i)
        end
    end
end
