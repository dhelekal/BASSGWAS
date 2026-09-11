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
    ParState2

Mutable model parameters and data shared by the sampler's numerical caches.

# Fields
- `Xaug`: Predictor matrix with an intercept column appended last.
- `Y`: Current latent-response vector.
- `U`, `S`: Eigenvectors and eigenvalues of the random-effect covariance.
- `p`, `n`: Numbers of predictors and observations, respectively.
- `alpha`, `beta`: Beta-Binomial inclusion-prior hyperparameters.
- `c`: Predictor-effect variance.
- `r`: Random-effect variance multiplier.
- `s`: Residual variance multiplier.
- `mc`: Intercept variance.
"""
mutable struct ParState2{Tx, Ty, Tv, Tn, Tr}
    Xaug::Tx
    Y::Ty
    U::Tx
    S::Tv
    p::Tn
    n::Tn
    alpha::Tr
    beta::Tr
    c::Tr
    r::Tr
    s::Tr
    mc::Tr
end

"""
    ParState2(X, Y, U, S, alpha, beta, c, r, s, mc)

Construct sampler parameters and append a column of ones to `X` for the
intercept.

`X` has one row per observation, `Y` has one element per observation, and the
rows of `U` correspond to the same observations. The scalar hyperparameters
must share, or promote compatibly to, a common storage type.
"""
function ParState2(X, Y, U, S, alpha, beta, c, r, s, mc)
    alpha, beta, c, r, s = promote(alpha, beta, c, r, s, mc)
    p = size(X, 2)    
    n = size(Y, 1)

    Xaug = [X ones(n)]
    return ParState2(Xaug, Y, U, S, p, n, alpha, beta, c, r, s, mc)
end
