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
    ProbitBVSSampler

Mutable sampler for Bayesian variable selection in a probit mixed model.

The sampler owns the discrete inclusion state, latent liabilities, low-rank
random-effect caches, continuous variance parameters, adaptation state, and
random-number generator.

# Fields
- `parState`: Current model data and continuous parameters.
- `scache`, `qcache`: Low-rank covariance and active-design caches.
- `wTGSState`: Weighted tempered Gibbs variable-selection state.
- `loglik`: Collapsed likelihood evaluator.
- `T`: Number of completed transitions.
- `naux`: Reserved auxiliary-transition counter.
- `nadapt`, `isadapted`: Adaptation horizon and completion flag.
- `adapt_xi`, `xi_targ`: Auxiliary-intensity adaptation controls.
- `sample_c`: Whether to sample the predictor-effect variance.
- `adaptor`: Random-walk Metropolis adaptor.
- `shape`, `scale`: Inverse-gamma latent-scale hyperparameters.
- `rng`: Random-number generator advanced by sampling operations.
"""
mutable struct ProbitBVSSampler{Tp, Tc, Ts, Tk, Tl, Ti, Tb, Tn, Ta, Tr}
    parState::Tp
    scache::Tc
    wTGSState::Ts
    qcache::Tk
    loglik::Tl
    T::Ti
    naux::Ti
    nadapt::Ti
    isadapted::Tb
    adapt_xi::Tb
    xi_targ::Tn
    sample_c::Tb
    adaptor::Ta
    shape::Tn
    scale::Tn
    rng::Tr
end

"""
    ProbitBVSSampler(X, Y, U, S, alpha, beta; kwargs...)

Construct a weighted-TGS sampler for a binary probit variable-selection model
with low-rank correlated random effects.

`X` is an `n × p` predictor matrix, `Y` is a length-`n` integer response, `U`
is an `n × k` covariance-eigenvector matrix, and `S` contains its `k`
eigenvalues. Positive response values are treated as successes and nonpositive
values as failures. `alpha` and `beta` are the positive Beta-Binomial
inclusion-prior hyperparameters. Retained eigenvalues must be nonnegative;
variance and scale parameters must be positive. The rows of `X`, `Y`, and `U`
must align. A common floating-point type is inferred from the design,
eigensystem, hyperparameters and numerical keyword arguments. State, cache,
likelihood and adaptor storage use that type.

# Keywords
- `c=6.0`: Predictor-effect variance when `sample_c=false`; otherwise the
  variance is initialized and sampled internally.
- `sample_c=true`: Sample the predictor-effect variance.
- `mcinit=100.0`: Intercept variance.
- `eps=5.0`: Weighted-TGS regularization weight.
- `nadapt=10000`: Number of transitions used for proposal adaptation; zero
  disables all adaptation.
- `xi_init=5.0`: Initial auxiliary-update intensity.
- `adapt_xi=true`: Also adapt the auxiliary-update intensity during warm-up.
  When false, `xi` stays fixed while variance proposals still adapt.
- `xi_targ=0.25`: Target fraction of total move intensity assigned to the
  auxiliary update.
- `nu=5.0`, `lambda=0.6`: Residual-scale prior degrees of freedom and scale.
- `rng=Random.default_rng()`: Random-number generator owned and advanced by
  the sampler.

The initial predictor model is empty, with an intercept always included.
Construction draws initial latent liabilities and positive variance values, so
it advances `rng`.
"""
function ProbitBVSSampler(
        X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Integer}, U::AbstractMatrix{<:Real}, S::AbstractVector{<:Real},
        alpha, beta; c = 6.0, sample_c = true, mcinit = 100.0, eps = 5.0, nadapt = 10000, xi_init = 5.0, adapt_xi = true, xi_targ = 0.25, nu = 5.0, lambda = 0.6, rng = Random.default_rng()
    )

    nadapt isa Integer && nadapt >= 0 || throw(ArgumentError("nadapt must be a nonnegative integer"))

    T = _state_float_type(
        eltype(X), eltype(U), eltype(S), typeof(alpha), typeof(beta),
        typeof(c), typeof(mcinit), typeof(eps), typeof(xi_init),
        typeof(xi_targ), typeof(nu), typeof(lambda)
    )
    one_t = one(T)
    initial_floor = T(1.0e-8)
    alpha, beta, nu, lambda, eps, xi_init, xi_targ = T.((alpha, beta, nu, lambda, eps, xi_init, xi_targ))
    p = size(X, 2)
    n = size(Y, 1)

    γ0 = zeros(Int, p)

    shape = nu / T(2)
    scale = shape * lambda

    loglik = TDistLogLik{T}(nu, lambda, n)

    # Initialize latent liabilities on the side of zero fixed by each binary outcome.
    signs = ifelse.(Y .> 0, one_t, -one_t)
    Yinit = abs.(randn(rng, T, n)) .* signs
    rinit = abs(randn(rng, T)) + initial_floor
    cinit = sample_c ? abs(randn(rng, T)) + initial_floor : T(c)

    sinit = one_t

    parState = ParState2{T}(X, Yinit, U, S, alpha, beta, cinit, rinit, sinit, mcinit)
    cache = makeSCache(parState)
    tgsstate, qcache = init_state2(loglik, parState, cache, γ0; eps = eps, xi_init = xi_init)

    adaptor = RWMAdaptor(T, sample_c ? 2 : 1, 25 * div(nadapt, 12))

    return ProbitBVSSampler(parState, cache, tgsstate, qcache, loglik, 0, 0, nadapt, iszero(nadapt), adapt_xi, xi_targ, sample_c, adaptor, shape, scale, rng)
end

"""
    next_step!(samp::ProbitBVSSampler)

Advance `samp` by one weighted-TGS transition.

The first eleven transitions update only latent and continuous quantities.
Subsequent transitions select either a predictor toggle or an auxiliary Gibbs
and Metropolis update according to the current move intensities. When enabled,
adaptation is disabled once the transition count reaches `nadapt`. The sampler
and its RNG are mutated; the return value is not part of the public interface.
"""
function next_step!(samp::ProbitBVSSampler)
    if samp.T > 10
        next_flip = sample_next2(samp.wTGSState, samp.rng)
        if next_flip <= samp.parState.p
            flip_gamma!(samp.wTGSState, samp.parState, samp.qcache, next_flip)
            update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
        else
            gibbs_CG!(samp)
            update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
        end
        # Target the auxiliary update's share of the total move intensity.
        if !samp.isadapted && samp.adapt_xi
            xi_targ = samp.xi_targ
            xi = samp.wTGSState.xi
            logtot = samp.wTGSState.logtot
            samp.wTGSState.xi = xi + (xi_targ - exp(log(xi) - logtot)) / (sqrt(samp.T))
        end
    else
        gibbs_CG!(samp)
        update_state2!(samp.wTGSState, samp.parState, samp.scache, samp.qcache, samp.loglik)
    end
    samp.T += 1
    if samp.T >= samp.nadapt
        samp.isadapted = true
    end

    n_active = sum(samp.wTGSState.γ)
    return n_active >= 100 &&
        @warn "Warning: The number of active variants has exceed 100. Number of active variants: $(n_active). If this warning appears repeatedly, the model is likely misspecified, or there is unmeasured confounding present!"
end

"""
    gibbs_CG!(samp::ProbitBVSSampler)

Update the sampler's continuous variances, Gaussian coefficients, and probit
liabilities conditional on the selected predictor model.

The transition combines adaptive random-walk Metropolis proposals with Gibbs
draws and parameter-expanded shift and scale moves. It mutates `samp`, its
caches, and its RNG.
"""
function gibbs_CG!(samp::ProbitBVSSampler)
    shiftmv = true
    scalemv = true
    scale1 = true

    rng = samp.rng
    ps = samp.parState
    sc = samp.scache
    qc = samp.qcache
    shape = samp.shape
    scale = samp.scale

    T = eltype(ps.Y)
    one_t, zero_t = one(T), zero(T)
    proposal_floor = T(1.0e-4)
    Y = ps.Y
    r = ps.r
    c = ps.c
    mc = ps.mc

    X = get_X1(qc)
    V = sc.V

    # Proposals are made on positive standard-deviation scales and squared after acceptance.
    rsqrt = sqrt(r)
    csqrt = sqrt(c)

    ll_curr = samp.wTGSState.ll_curr

    rprior = Exponential(-T(3) / log(T(0.05)))
    cprior = truncated(InverseGamma(T(6), T(12)); lower = zero_t)

    adaptor = samp.adaptor

    # Metropolis updates use a likelihood collapsed over Gaussian coefficients.
    for i in 1:25

        dt = get_dt(adaptor)
        rwm_scales = get_scales(adaptor)

        # Reflective jitter keeps the proposed standard deviation positive.
        rprop = abs(rsqrt + i * randn(rng, T) * dt * rwm_scales[1] - proposal_floor) + proposal_floor
        cprop = csqrt

        if samp.sample_c
            # Apply the same positive-scale safeguard to the predictor effect.
            cprop = abs(csqrt + i * randn(rng, T) * dt * rwm_scales[2] - proposal_floor) + proposal_floor
        end

        prll_curr = logpdf(rprior, rsqrt) + logpdf(cprior, csqrt)
        prll_prop = logpdf(rprior, rprop) + logpdf(cprior, cprop)

        ssq, det = proposal_terms!(qc, sc, rprop * rprop, cprop * cprop, mc)
        ll_prop = samp.loglik(det, ssq)

        qr = ll_prop + prll_prop - ll_curr - prll_curr
        u = log(rand(rng, T))

        accept = 0

        if u <= qr
            accept = 1
            rsqrt = rprop
            csqrt = cprop
            ll_curr = ll_prop
        end

        if !samp.isadapted
            adapt!(adaptor, samp.sample_c ? [rsqrt, csqrt] : [rsqrt], accept)
        end
    end

    # Convert accepted standard deviations back to variance parameters.
    r_new = rsqrt * rsqrt
    c_new = csqrt * csqrt

    # Liability signs retain the observed binary outcomes throughout augmentation.
    signs = sign.(Y)

    n = size(Y, 1)
    l = size(X, 2)
    m = sc.ns

    # Active predictors, intercept, and random-effect basis form the Gaussian model matrix.
    K = [X V]
    Y_old = copy(Y)

    # Prepare the accepted factor once, independently of the last proposal.
    accepted_ssq, _ = proposal_terms!(qc, sc, r_new, c_new, mc)

    # Draw an auxiliary residual scale on a copy used only for coefficients.
    if scalemv && scale1
        _ = draw_Ysigma!(Y_old, shape, scale, accepted_ssq, rng)
    end

    beta = draw_betas(Y_old, r_new, c_new, mc, l, m, K, rng)

    # Reuse the fitted means as storage for the new latent liabilities.
    Y_new = K * beta

    # Resample liabilities from the half-line compatible with each observed outcome.
    for i in eachindex(Y_new)
        y_mu = Y_new[i]
        if signs[i] >= 0.0
            Y_new[i] = rand(rng, truncated(Normal(y_mu, one_t); lower = zero_t))
        else
            Y_new[i] = rand(rng, truncated(Normal(y_mu, one_t); upper = zero_t))
        end
    end

    # The feasible interval preserves all liability signs under the parameter-expanded shift.
    if shiftmv
        sig = T(10)
        mu = rand(rng, Normal(zero_t, sig))
        Y_new .+= mu
        sqI1, YI1 = shift_terms!(qc, sc, Y_new)
        # Normal's second argument is a standard deviation, so its precision is 1/sig^2.
        tauup = one_t / sig^2 + sqI1
        sigup = one_t / tauup
        muup = YI1 * sigup
        # An absent outcome class imposes no bound on that side of the shift.
        lo = maximum(Y_new[findall(signs .< 0.0)]; init = T(-Inf))
        up = minimum(Y_new[findall(signs .> 0.0)]; init = T(Inf))
        mu_new = rand(rng, truncated(Normal(muup, sqrt(sigup)), lower = lo, upper = up))
        Y_new .-= mu_new
    end

    # Positive rescaling changes latent scale without changing the binary observations.
    if scalemv
        sigma = rand(rng, InverseGamma(shape, scale))
        Y_new .*= sqrt(sigma)
    end

    if scalemv && !scale1
        _ = draw_Ysigma!(Y_new, shape, scale, response_ssq!(qc, sc, Y_new), rng)
    end

    ps.Y = Y_new
    ps.c = c_new
    ps.r = r_new

    return updateSCache!(sc, ps)
end

"""
    draw_Ysigma!(Y, r, c, mc, shape, scale, X, XtX, UtX, U, S, rng)

Draw the residual variance conditional on `Y` under the normal-inverse-gamma
model and standardize `Y` by the sampled standard deviation.

`Y` is mutated in place, `rng` is advanced, and the sampled variance is
returned.
"""
function draw_Ysigma!(Y, r, c, mc, shape, scale, X, XtX, UtX, U, S, rng)
    ssq = _comp_ssq(Y, r, c, mc, X, XtX, UtX, U, S)
    return draw_Ysigma!(Y, shape, scale, ssq, rng)
end

# Shared draw preserves RNG ordering for both cached and stand-alone callers.
function draw_Ysigma!(Y, shape, scale, ssq, rng)
    T = eltype(Y)
    two_t = T(2)
    aup = T(shape) + T(length(Y)) / two_t
    bup = T(scale) + T(ssq) / two_t

    sigma = rand(rng, InverseGamma(aup, bup))
    Y .*= one(T) / sqrt(sigma)
    return sigma
end

"""
    draw_betas(Y, r, c, mc, l, m, K, rng)

Draw active predictor, intercept, and random-effect coefficients from their
Gaussian conditional posterior.

The first `l - 1` columns of `K` are active predictors, column `l` is the
intercept, and the final `m` columns are random-effect basis vectors. The
returned vector follows that same order and advances `rng`.
"""
function draw_betas(Y, r, c, mc, l, m, K, rng)

    T = eltype(K)
    one_t = one(T)
    Qinv = K'K

    # Add predictor prior precisions to the active fixed-effect block.
    for i in 1:(l - 1)
        Qinv[i, i] += one_t / c
    end

    # Use the separate intercept prior precision.
    Qinv[l, l] += one_t / mc

    # Add the shared random-effect prior precision.
    for i in (l + 1):(m + l)
        Qinv[i, i] += one_t / r
    end

    Qf = cholesky(Qinv)
    beta_mu = Qf \ (K' * Y)
    eta = rand(rng, Normal(zero(T), one_t), l + m)
    beta = Qf.U \ eta + beta_mu

    return beta
end

"""
    get_state(samp::ProbitBVSSampler)

Draw continuous posterior quantities conditional on the sampler's current
selected model and return a named tuple.

The result contains:
- `γ`: Current binary inclusion vector.
- `pips`: Current conditional predictor-inclusion probabilities.
- `icept`: Intercept draw.
- `beta`: Length-`p` predictor-effect draw with zeros at inactive positions.
- `u`: Random-effect coefficient draw in the retained eigenbasis.
- `Y`: Standardized latent-liability draw.
- `sigma`: Residual-variance draw used to standardize `Y`.
- `r`: Current random-effect variance multiplier.
- `c`: Current predictor-effect variance.
- `Z`: Weighted-TGS importance weight, the reciprocal total move intensity.

`γ` and `pips` alias the sampler's internal arrays; callers that retain them
across transitions should copy them. Other returned arrays are newly
allocated. The sampler's RNG is advanced, but its model state is otherwise
unchanged.
"""
function get_state(samp::ProbitBVSSampler)

    ps = samp.parState
    sc = samp.scache
    qc = samp.qcache

    γ = samp.wTGSState.γ
    pips = samp.wTGSState.pips
    # Reciprocal total move rate is the weighted-TGS importance weight.
    Z = exp(-samp.wTGSState.logtot)

    Y = copy(ps.Y)
    r = ps.r
    c = ps.c
    mc = ps.mc

    shape = samp.shape
    scale = samp.scale
    rng = samp.rng

    X = get_X1(qc)

    V = sc.V

    l = size(X, 2)
    m = size(V, 2)
    K = [X V]

    work = prepare_active!(qc, sc, c, mc)
    sigma = draw_Ysigma!(Y, shape, scale, work.ssq, rng)
    bs = draw_betas(Y, r, c, mc, l, m, K, rng)
    beta = zeros(eltype(pips), size(pips, 1))

    # Expand active coefficients back into the full predictor coordinate system.
    beta[qc.i1_set[1:(l - 1)]] .= bs[1:(l - 1)]
    icept = bs[l]
    u = bs[(l + 1):end]

    return (γ = γ, pips = pips, icept = icept, beta = beta, u = u, Y = Y, sigma = sigma, r = r, c = c, Z = Z)
end

"""
    is_adapted(samp::ProbitBVSSampler)

Return the sampler's adaptation-completion flag.

The flag is initialized to `true` only when `nadapt=0`. Otherwise it becomes
`true` after exactly `nadapt` transitions, independently of `adapt_xi`.
Setting `adapt_xi=false` fixes the auxiliary intensity but does not disable
variance-proposal adaptation during warm-up.
"""
function is_adapted(samp::ProbitBVSSampler)
    return samp.isadapted
end

"""
    RWMAdaptor

Three-stage adaptor for random-walk Metropolis proposals.

Stages one and three tune the global proposal step toward the target acceptance
rate. Stage two estimates relative marginal scales for the sampled variance
parameters.

# Fields
- `stage_len`: Number of updates in each of the first two adaptation stages.
- `buffer_len`, `buffer_it`: Rolling acceptance-window length and position.
- `stage`, `steps`: Current stage and updates completed within it.
- `n_acc`: Accepted proposals retained in the rolling window.
- `n_var`: Number of jointly adapted variables.
- `dt`: Global proposal step size.
- `scales`: Relative proposal scale for each variable.
- `sums`, `sums_sq`: Stage-two moment accumulators.
"""
mutable struct RWMAdaptor{T <: AbstractFloat}
    stage_len::Int
    buffer_len::Int
    buffer_it::Int
    stage::Int
    steps::Int
    n_acc::Int
    n_var::Int
    dt::T
    scales::Vector{T}
    sums::Vector{T}
    sums_sq::Vector{T}
end

"""
    RWMAdaptor(n_var::Int, stage_len::Int)
    RWMAdaptor(T, n_var::Int, stage_len::Int)

Construct an adaptor for `n_var` variables with equal initial scales and global
step size `0.1`. Storage uses `T`, defaulting to `Float64` in the two-argument
constructor. Samplers pass their selected floating-point type explicitly.

The first two adaptation stages contain `stage_len` updates each. The third
stage continues tuning the global step until the owning sampler ends
adaptation.
"""
RWMAdaptor(n_var::Int, stage_len::Int) = RWMAdaptor(Float64, n_var, stage_len)

function RWMAdaptor(::Type{T}, n_var::Int, stage_len::Int) where {T <: AbstractFloat}
    isconcretetype(T) || throw(ArgumentError("adaptor element type must be concrete"))
    buffer_len = div(stage_len, 2)
    return RWMAdaptor(
        stage_len,
        buffer_len,
        0,
        1,
        0,
        0,
        n_var,
        T(0.1),
        ones(T, n_var),
        zeros(T, n_var),
        zeros(T, n_var)
    )
end

"""
    adapt!(adaptor::RWMAdaptor, state, acc)

Update proposal adaptation from the current parameter `state` and acceptance
indicator `acc`.

The adaptor is mutated in place. During the middle stage, empirical variances
set normalized relative proposal scales with a small positive floor.
"""
function adapt!(adaptor::RWMAdaptor{T}, state::AbstractVector{T}, acc::Int) where {T}
    adaptation_floor = T(0.001)
    adaptor.buffer_it += 1
    adaptor.n_acc += acc
    adaptor.steps += 1

    # Stages 1 and 3 target acceptance; stage 2 estimates relative marginal scales.
    if adaptor.stage == 1 || adaptor.stage == 3
        accr = T(adaptor.n_acc) / T(adaptor.buffer_it)
        adaptor.dt = max(adaptor.dt + one(T) * (accr - T(0.22)) / sqrt(T(adaptor.steps)), adaptation_floor)
    else
        adaptor.sums_sq .+= state .* state
        adaptor.sums .+= state
    end

    # Retain a rolling half-window for the acceptance-rate estimate.
    if adaptor.buffer_it >= adaptor.buffer_len
        adaptor.n_acc = div(adaptor.n_acc, 2)
        adaptor.buffer_it = div(adaptor.buffer_it, 2)
    end

    return if adaptor.steps >= adaptor.stage_len && adaptor.stage < 3
        if adaptor.stage == 2
            n = adaptor.steps
            n2 = n * n
            var = adaptor.sums_sq ./ n .- (adaptor.sums .* adaptor.sums) ./ n2
            adaptor.scales .= sqrt.(var) .+ T(1.0e-6)
            adaptor.scales ./= maximum(adaptor.scales)
            adaptor.scales .= max.(adaptor.scales, adaptation_floor)
        end

        adaptor.sums .= 0.0
        adaptor.sums_sq .= 0.0
        adaptor.steps = 0
        adaptor.n_acc = 0
        adaptor.buffer_it = 0
        adaptor.stage += 1
    end
end

"""
    get_scales(adaptor::RWMAdaptor)

Return the current vector of relative proposal scales.

The returned vector aliases storage owned by `adaptor`.
"""
function get_scales(adaptor::RWMAdaptor)
    return adaptor.scales
end

"""
    get_dt(adaptor::RWMAdaptor)

Return the adaptor's current global proposal step size.
"""
function get_dt(adaptor::RWMAdaptor)
    return adaptor.dt
end
