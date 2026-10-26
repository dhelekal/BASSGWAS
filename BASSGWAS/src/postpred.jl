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
    postpred(U, S, X, icept, betas, us)

Compute posterior-predictive log probabilities for binary probit outcomes.

`X` is an `n × p` predictor matrix for all prediction targets, `U` is an
`n × k` random-effect eigenvector matrix, and `S` contains its `k` eigenvalues.
For `m` posterior draws, `icept` has length `m`, `betas` is `m × p`, and `us`
is `m × k` in the eigenbasis represented by `U` and `S`.

Return `(lp_success, lp_fail)`, two `m × n` matrices containing `log P(Y=1)`
and `log P(Y=0)` for every draw and target. The inputs are not mutated.
"""
function postpred(
        U::AbstractMatrix{Tx}, S::AbstractVector{Tx},
        X::AbstractMatrix{Tx}, icept::AbstractVector{Tx},
        betas::AbstractMatrix{Tx}, us::AbstractMatrix{Tx}
    ) where {Tx <: Real}

    n_Y = size(X, 1)

    V = U * Diagonal(sqrt.(S))
    # Arrange model columns and posterior draws for one matrix multiplication.
    K = Matrix([X ones(n_Y) V]')
    bs = Matrix([betas icept us]')
    linpred = bs' * K
    lp_success = similar(linpred)
    lp_fail = similar(linpred)

    for i in eachindex(linpred, lp_success, lp_fail)
        lp_success[i] = logccdf(Normal(linpred[i], 1.0), 0.0)
        lp_fail[i] = logcdf(Normal(linpred[i], 1.0), 0.0)
    end

    return lp_success, lp_fail
end

"""
    expand_randeff(Um, Sm, U, S, idx_obs, us, rs, rng)

Conditionally expand sampled random-effect coefficients from observed targets
to the eigenbasis of the full target covariance.

`Um` and `Sm` describe the covariance used for the `n_obs` sampled targets.
`U` and `S` describe the covariance for all targets, and `idx_obs` gives the
one-based rows of `U` corresponding to the sampled targets. For `m` posterior
draws, `us` is `m × k_obs` and `rs` is a length-`m` vector of random-effect
variance draws.

Return an `m × k_full` matrix of full-basis random-effect coefficients. The
method samples from the conditional Gaussian distribution, advances `rng`,
checks that the restricted covariances agree, and warns if reconstruction of
the observed random effects loses numerical accuracy.
"""
function expand_randeff(
        Um::AbstractMatrix{Tx}, Sm::AbstractVector{Tx},
        U::AbstractMatrix{Tx}, S::AbstractVector{Tx},
        idx_obs::Array{Ty, 1}, us::AbstractMatrix{Tx},
        rs::AbstractVector{Tx}, rng::AbstractRNG
    ) where {Tx <: Real, Ty <: Int}

    n_it = size(us, 1)
    k = size(S, 1)

    # Form the covariance pseudoinverse for the conditional Gaussian draw.
    pinv = Um * Diagonal(inv.(Sm)) * Um'
    Vm = Um * Diagonal(sqrt.(Sm))
    # Convert sampled basis coefficients into observed random effects.
    re_obs = Vm * us'

    # Restrict the full covariance square root to observed targets.
    Cu = U[idx_obs, :] * Diagonal(sqrt.(S))
    # Conditioning is valid only if both decompositions encode the same submatrix.
    !(isapprox(Cu * Cu', Vm * Vm'; rtol = 1.0e-8, atol = 1.0e-8)) &&
        error("Covariance doesn't match restricted covariance.")

    rD = Diagonal(sqrt.(rs))
    # Scale independent standard normals by each posterior variance draw.
    Zs = rand(rng, Normal(), k, n_it) * rD

    # Compute conditional means and covariance in the full eigenbasis.
    cond_means = Cu' * pinv * re_obs
    cond_cov = Diagonal(ones(k)) - Cu' * pinv * Cu

    # QR iteration is more stable here than LAPACK's divide-and-conquer SVD.
    sv = svd(cond_cov, alg = LinearAlgebra.QRIteration())

    B = sv.U * Diagonal(sqrt.(sv.S))
    re_coeffs = cond_means .+ (B * Zs)

    V = U * Diagonal(sqrt.(S))
    # Verify that expanded coefficients reproduce sampled effects at observed rows.
    !(isapprox(V[idx_obs, :] * re_coeffs, re_obs; rtol = 1.0e-3, atol = 1.0e-3)) &&
        error("Random effects of observations don't match. $(maximum(abs.(V[idx_obs, :] * re_coeffs .- re_obs)))")

    !(isapprox(V[idx_obs, :] * re_coeffs, re_obs; rtol = 1.0e-4, atol = 1.0e-4)) &&
        @warn "Poor accuracy of expanded random effects detected. Max abserr: $(maximum(abs.(V[idx_obs, :] * re_coeffs .- re_obs))), Max relerr: $(maximum(abs.((V[idx_obs, :] * re_coeffs .- re_obs)) ./ re_obs))"

    return re_coeffs'
end
