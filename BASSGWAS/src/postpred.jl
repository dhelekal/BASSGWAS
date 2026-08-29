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

#U :: random effect singular vectors all Ys
#S :: random effect singular values for all Ys
#X :: predictor matrix for all Ys
#icept :: intercept draws
#betas :: beta draws
#us :: random effect draws in full basis
function postpred(U::AbstractMatrix{Tx}, S::AbstractVector{Tx}, 
        X::AbstractMatrix{Tx}, icept::AbstractVector{Tx}, 
        betas::AbstractMatrix{Tx}, us::AbstractMatrix{Tx}) where {Tx<: Real}

    n_Y = size(X, 1)

    V = U * Diagonal(sqrt.(S))
    K = Matrix([X ones(n_Y) V]') #model matrix [predictors, intercept, random effect] NxP
    bs = Matrix([betas icept us]') #PxM
    linpred = bs'*K #bs' * K' #(K*bs)' #MxN
    lp_success = similar(linpred)
    lp_fail = similar(linpred)

    for i in eachindex(linpred, lp_success, lp_fail)
        lp_success[i] = logccdf(Normal(linpred[i],1.0), 0.0)
        lp_fail[i] = logcdf(Normal(linpred[i],1.0), 0.0)
    end

    return lp_success, lp_fail
end

#Um :: random effect singular vectors used during sampling
#Sm :: random effect singular values used during sampling
#U :: random effect singular vectors all Ys
#S :: random effect singular values for all Ys
#idx_obs :: index set of observed Ys
#us :: random effect draws
#rs :: random effect variance draws
function expand_randeff(Um::AbstractMatrix{Tx}, Sm::AbstractVector{Tx},
    U::AbstractMatrix{Tx}, S::AbstractVector{Tx},
    idx_obs::Array{Ty, 1}, us::AbstractMatrix{Tx}, 
    rs::AbstractVector{Tx}, rng::AbstractRNG) where {Tx<: Real, Ty<:Int}
    
    #P = n_pred
    #N = n_resp
    #M = n_it
    n_it = size(us, 1)
    k = size(S, 1)

    #first extrapolate random effect
    #pseudoinverse for conditional gaussian
    pinv = Um * Diagonal(inv.(Sm))*Um'
    Vm = Um * Diagonal(sqrt.(Sm))
    #observed random effects
    re_obs = Vm * us'
    
    #restricted full svd square root
    Cu = U[idx_obs, :] * Diagonal(sqrt.(S))
    #sanity check that restricted covariances match
    !(isapprox(Cu*Cu', Vm*Vm'; rtol=1e-8, atol=1e-8)) && 
        error("Covariance doesn't match restricted covariance.")
    
    rD = Diagonal(sqrt.(rs))    
    #scaled iid normal draws
    Zs = rand(rng, Normal(), k, n_it) * rD

    #conditional means for all random effects in the svd basis
    cond_means = Cu' * pinv * re_obs
    #conditional covariance for all random effects in the svd basis
    cond_cov = Diagonal(ones(k)) - Cu'*pinv*Cu 
    
    #square root of cond cov via svd
    #Use QR iteration as it seems to be more stable -- LAPACK's gesdd is unreliable compared to gesvd
    sv=svd(cond_cov, alg=LinearAlgebra.QRIteration())
    
    B = sv.U * Diagonal(sqrt.(sv.S))
    re_coeffs = cond_means .+ (B*Zs)

    #Sanity check
    V = U * Diagonal(sqrt.(S))
    #sanity check that predicted random effects for observations match those that were observed, up to precision
    #the condition number appears to be poor
    !(isapprox(V[idx_obs, :]*re_coeffs, re_obs; rtol=1e-3, atol=1e-3)) && 
        error("Random effects of observations don't match. $(maximum(abs.(V[idx_obs, :]*re_coeffs .- re_obs)))")

    !(isapprox(V[idx_obs, :]*re_coeffs, re_obs; rtol=1e-4, atol=1e-4)) && 
        @warn "Poor accuracy of expanded random effects detected. Max abserr: $(maximum(abs.(V[idx_obs, :]*re_coeffs .- re_obs))), Max relerr: $(maximum(abs.((V[idx_obs, :]*re_coeffs .- re_obs))./re_obs))"

    return re_coeffs'
end
