using LinearAlgebra
using DataFrames
using CSV
using UnPack
using ArgParse
using BASSGWAS

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
        "--b_draws"
            required = true
            arg_type = String
            help = "beta draws"
        "--u_draws"
            required = true
            arg_type = String
            help = "u draws"
        "--par_draws"
            required = true
            arg_type = String
            help = "Additional parameter draws"
        "--ofile"
            required = true
            arg_type = String
            help = "Output file"
        "--batchsz"
            required = true
            arg_type = Int
            help = "Design batch size"
        "--inclnext"
            arg_type = String
            help = "File of sample names to include in the next batch"
    end

    s.description = "Optimise an experimental design using existing BASSGWAS samples."

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

function read_chains(path)
    draws = DataFrame(CSV.File(path))
    ndraws = maximum(draws[:, :iteration])
    nchains = maximum(draws[:, :chain])
    return expand_chains(Array{Float64}(draws)[:,3:end],nchains,ndraws)
end

#Parse args
parsed = parse_commandline()

pfile = parsed["pfile"]
ufile = parsed["ufile"]
sfile = parsed["sfile"]
obsfile = parsed["obsfile"]

bfile = parsed["b_draws"]
refile = parsed["u_draws"]
parfile = parsed["par_draws"]

batch_sz = parsed["batchsz"]
incl_next = parsed["inclnext"] !== nothing

#Print a trace
receiptfile = "./receipt.txt"
open(receiptfile, "w") do f
println(f,"")
println(f,"--- Adaptive design for genome wide prediction ---")
println(f,"")
println(f," -- Input Files --")
println(f,"  - Predictor matrix: $(pfile)")
println(f,"  - U matrix:         $(ufile)")
println(f,"  - S matrix:         $(sfile)")
println(f,"  - Observation file: $(obsfile)")
println(f,"  - beta draws: $(bfile)")
println(f,"  - u draws: $(refile)")
println(f,"  - Additional parameter draws file: $(parfile)")
println(f,"")
println(f," -- Design Parameters --")
println(f,"  - Batch size:               $(batch_sz)")
println(f,"  - Fixed samples in design?: $(incl_next)")
println(f,"    Fixed sample file:        $(incl_next ? parsed["inclnext"] : "")")
println(f,"")

end

BLAS.set_num_threads(1)
tol = 1e-8

T = Float64
data = DataFrame(CSV.File(pfile));
U = Array{T}(DataFrame(CSV.File(ufile))[:,1:end]);
S = reshape(Array{T}(DataFrame(CSV.File(sfile))[:,1:end]),:);
X = Array{T}(data[:, 2:end]);

obsU = DataFrame(CSV.File(obsfile))
#Ensure proper data ordering -- observation ordering must be consistent with original dataset order
obs = obsU[sortperm([findfirst(j .== data[:,1]) for j in obsU[:,1]]),:]
idx_obs = findall(in.(data[:,1], Ref(obs[:,1])))

idx_fixed = similar(idx_obs,0)
if incl_next
    obsfixed = DataFrame(CSV.File(parsed["inclnext"]));
    idx_fixed = [findfirst(j .== data[:,1]) for j in obsfixed[:,1]]
end

cpars_draws = read_chains(parfile)
betas_draws = read_chains(bfile)
u_draws = read_chains(refile)

#Experimental design
println("Designing next experiment ...")
#Skip if batch size = 0
betasf = flatten_chains(betas_draws) #reshape(permutedims(betas_draws,(1,3,2)), :, size(betas_draws,2))
iceptf = flatten_chains(cpars_draws[:, 5, :]) #reshape(cpars_draws[:,5,:], :, 1)
usf = flatten_chains(u_draws)
zz = flatten_chains(cpars_draws[:, 6, :]) #reshape(cpars_draws[:, 6, :], :)
Zs = (zz ./ sum(zz))

n_samp = size(data[:,1], 1)
idx_rem = (1:n_samp)[1:n_samp .∉ Ref(idx_obs)]

design = optimize_design(idx_rem, U, S, X, iceptf, betasf, usf, Zs, batch_sz; randdes=false, idx_fixed=idx_fixed)
selected = data[sort([idx_obs; design[1]]), 1]

println("Design objective value: $(design[2])")

println("Writing design ...")
outfile = parsed["ofile"]
open(outfile, "w") do f
    for i in selected
        println(f, i)
    end
end
