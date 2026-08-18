using LinearAlgebra
using MCMCChains
using DataFrames
using CSV
using Distributed
using UnPack
using ArgParse
using Optim
using Distributions

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
            help = "Design batch size"
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
            default = 4
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

function flatten_chains(x)
    if length(size(x)) == 2
        x_flat = reshape(x, :)
    elseif length(size(x)) == 3
        x_flat = reshape(permutedims(x,(1,3,2)), :, size(x,2))
    else
        throw(ErrorException("x must be a 2 or 3 dimensional array"))
    end
    return x_flat
end

function expand_chains(x, nchains, ndraws)
    return permutedims(reshape(x, ndraws, nchains, :),(1,3,2))
end

#Parse args
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

#Print a trace
receiptfile = outdir*"/receipt.txt"
open(receiptfile, "w") do f

println(f,"")
println(f,"--- Adaptive sampling for genome wide prediction ---")
println(f,"")
println(f," -- Input Files --")
println(f,"  - Predictor matrix: $(pfile)")
println(f,"  - U matrix:         $(ufile)")
println(f,"  - S matrix:         $(sfile)")
println(f,"  - Observation file: $(obsfile)")
println(f,"")
println(f," Output Directory: $(outdir)")
println(f,"")
println(f," -- Model Parameters --")
println(f,"  - Prior model size conditional mean $(pmean)")
println(f,"  - Prior model size null probability $(pr0)")
println(f,"")
println(f," -- Run Specification --")
println(f,"  - Using seed:       $(seed)")
println(f,"  - Draws per chain:  $(ndraws)")
println(f,"  - Number of chains: $(nchains)")
println(f,"  - Thinning factor:  $(nthin)")
println(f,"")
println(f," -- Design Parameters --")
println(f,"  - Batch size:               $(batch_sz)")
println(f,"  - Fixed samples in design?: $(incl_next)")
println(f,"    Fixed sample file:        $(incl_next ? parsed["inclnext"] : "")")
println(f,"  - Random design:            $(randdes)")
println(f,"")

end

#Actual set up begins here
println("Adding worker processes ...")
addprocs(nchains)
println("Done")

@everywhere begin
    using BASSGWAS
    using Random
    using UnPack
end

Random.seed!(seed)
BLAS.set_num_threads(1)
tol = 1e-8

T = Float64
data = DataFrame(CSV.File(pfile));
U = Array{T}(DataFrame(CSV.File(ufile))[:,1:end]);
S = reshape(Array{T}(DataFrame(CSV.File(sfile))[:,1:end]),:);
obsU = DataFrame(CSV.File(obsfile))

#Ensure proper data ordering -- observation ordering must be consistent with original dataset order
obs = obsU[sortperm([findfirst(j .== data[:,1]) for j in obsU[:,1]]),:]
idx_obs = findall(in.(data[:,1], Ref(obs[:,1])))

idx_fixed = similar(idx_obs,0)
if incl_next
    obsfixed = DataFrame(CSV.File(parsed["inclnext"]));
    idx_fixed = [findfirst(j .== data[:,1]) for j in obsfixed[:,1]]
end

#Prepare restricted SVD basis
Ur = U[idx_obs, :]
Cs = Ur * Diagonal(S) * Ur'
sv = svd(Cs)

nnz = findall(sv.S .> tol)
So = sv.S[nnz]
Uo = sv.U[:, nnz]
X = Array{T}(data[:, 2:end]);
Xo = Array{T}(data[idx_obs,2:end]);
Yo = reshape(Array{Int64}(obs[:, 2]),:) 

#Optimize inclusion probability prior
p = size(Xo, 2)

function bbinom_obj(x, n, pr0, mu)
    v = pdf(BetaBinomial(n,exp(x[1]),exp(x[2])))
    v0 = v[1]
    lp0 = log(pr0/(1.0-pr0))
    lv0 = log(v0/(1.0-v0))

    cmu = sum(v[2:end]./(1.0-v0) .* collect(1:n))
    return (lp0-lv0)^2.0 + (mu-cmu)^2.0
end

sol = optimize(x -> bbinom_obj(x, p, pr0, pmean),
    [-3.0,-3.0], 
    NelderMead(),
    Optim.Options(g_tol = 1e-12))

alpha = exp(sol.minimizer[1])
beta = exp(sol.minimizer[2])

sol.minimum > tol && 
    error("Failed to calibrate model size prior. Objective: $(sol.minimum)")

#Xi adaptation with a single chain
println("Adapting xi ...")
xi = let samp = ProbitBVSSampler(Xo, Yo, Uo, So, alpha, beta; nadapt=50*nthin) 
    while !is_adapted(samp)
        next_step!(samp)
    end
    samp.wTGSState.xi
end
println("Found xi=$(xi)")
#Sampler call
@everywhere begin    
    using LinearAlgebra.BLAS
    BLAS.set_num_threads(1)
    function run_sampling(x, y, u, s, xi, seed, K, thin, alpha, beta)
        p = size(x, 2)
        n = size(x, 1)
        rng = Xoshiro(seed)

        _burnin = 100*thin
        _it = K*thin

        gamma_d = zeros(Int16, K, p)
        pips_d = zeros(Float64, K, p)
        cpars_d = zeros(Float64, K, 6)
        Ys_d = zeros(Float64, K, n)
        betas_d = zeros(Float64, K, p)
        us_d = zeros(Float64, K, size(s,1))

        samp = ProbitBVSSampler(x, y, u, s, alpha, beta; nadapt=_burnin, xi_init = xi, adapt_xi=false, rng=rng) 
        for _ in 1:_burnin
            next_step!(samp)
        end

        for i in 1:_it 
            next_step!(samp)
            if i % thin == 0
                j = trunc(Int64,i/thin)

                @unpack γ, pips, icept, beta, u, Y, sigma, r, c, Z = get_state(samp)
                gamma_d[j, :] .= γ
                pips_d[j, :] .= pips
                betas_d[j, :] .= beta
                us_d[j, :] .= u 
                Ys_d[j, :] .= Y

                cpars_d[j, 1] = sum(γ)
                cpars_d[j, 2] = sigma
                cpars_d[j, 3] = r
                cpars_d[j, 4] = c
                cpars_d[j, 5] = icept
                cpars_d[j, 6] = Z
            end
        end
        return (gammas=gamma_d, pips=pips_d, cpars=cpars_d, Ys=Ys_d, betas=betas_d, us=us_d)
    end
end
#Run parallel sampling
seeds = rand(1:100000000000,nchains)
println("Running parallel chains ...")
draws = pmap(1:nchains) do i
    run_sampling(Xo, Yo, Uo, So, xi, seeds[i], ndraws, nthin, alpha, beta)
end
println("Done")

#Process and save draws
println("Processing draws ...")
gammas_draws = zeros(Int64, (size(draws[1].gammas)..., nchains)) 
betas_draws = zeros(Float64, (size(draws[1].betas)..., nchains))
pips_draws = zeros(Float64, (size(draws[1].pips)..., nchains)) 
Ys_draws = zeros(Float64, (size(draws[1].Ys)..., nchains)) 
us_draws = zeros(Float64, (size(draws[1].us)..., nchains)) 
cpars_draws = zeros(Float64, (size(draws[1].cpars)..., nchains)) 

for i in 1:nchains
    @unpack gammas, pips, cpars, Ys, betas, us = draws[i]
    betas_draws[:,:, i] .= betas
    gammas_draws[:,:, i] .= gammas
    pips_draws[:,:, i] .= pips
    us_draws[:, :, i] .= us
    Ys_draws[:,:, i] .= Ys
    cpars_draws[:,:, i] .= cpars
end
draws = nothing

usf = flatten_chains(us_draws)
rsf = flatten_chains(cpars_draws[:,3,:])

usf_full = expand_randeff(Uo, So, U, S, idx_obs, usf, rsf, Random.default_rng())
us_full_draws = permutedims(reshape(usf_full, ndraws, nchains, :),(1,3,2))
!isapprox(usf_full[1:ndraws,:], us_full_draws[1:ndraws, :, 1]) && throw(ErrorException("us dont match"))

cchn = Chains(cpars_draws, [:n, :sigma, :r, :c, :icept, :Z])
show(stdout, "text/plain", describe(cchn)[1])
println("Saving draws ...")

    CSV.write(outdir*"/par_draws.csv",DataFrame(cchn))

    let chn = Chains(Ys_draws)
        CSV.write(outdir*"/Y_draws.csv",DataFrame(chn))
    end

    let chn = Chains(us_full_draws)
        CSV.write(outdir*"/u_draws.csv",DataFrame(chn))
    end

    let chn = Chains(betas_draws)
        CSV.write(outdir*"/b_draws.csv",DataFrame(chn))
    end

    let chn = Chains(pips_draws)
        CSV.write(outdir*"/pip_draws.csv",DataFrame(chn))
    end

    let chn = Chains(gammas_draws)
        CSV.write(outdir*"/gamma_draws.csv",DataFrame(chn))
    end
println("Done")


#Experimental design
println("Designing next experiment ...")
#Skip if batch size = 0
if batch_sz > 0

    betasf = flatten_chains(betas_draws) #reshape(permutedims(betas_draws,(1,3,2)), :, size(betas_draws,2))
    !isapprox(betasf[1:ndraws,:], betas_draws[1:ndraws, :, 1]) && throw(ErrorException("betas dont match"))

    iceptf = flatten_chains(cpars_draws[:, 5, :]) #reshape(cpars_draws[:,5,:], :, 1)
    !isapprox(iceptf[1:ndraws,:], cpars_draws[1:ndraws, 5, 1]) && throw(ErrorException("icept doesnt match"))

    zz = flatten_chains(cpars_draws[:, 6, :]) #reshape(cpars_draws[:, 6, :], :)
    !isapprox(zz[1:ndraws,:], cpars_draws[1:ndraws, 6, 1]) && throw(ErrorException("Zs dont match"))
    Zs = (zz ./ sum(zz))

    n_samp = size(data[:,1], 1)
    idx_rem = (1:n_samp)[1:n_samp .∉ Ref(idx_obs)]

    design = optimize_design(idx_rem, U, S, X, iceptf, betasf, usf_full, Zs, batch_sz; randdes=randdes, idx_fixed=idx_fixed)
    selected = data[sort([idx_obs; design[1]]), 1]

    println("Design objective value: $(design[2])")
    println("Writing design ...")
    outfile = outdir*"/design.txt"
    open(outfile, "w") do f
        for i in selected
            println(f, i)
        end
    end
end