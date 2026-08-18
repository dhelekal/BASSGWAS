abstract type Aloglik end

struct TDistLogLik{T<:Real} <: Aloglik
    nu::T
    lambda::T
    n::T
end

function TDistLogLik(nu,lambda,n)
    nu, lambda, n = promote(nu, lambda, n)
    TDistLogLik(nu,lambda,n)
end

(x::TDistLogLik{T})(det::T, ssq::T) where{T} = -0.5 * det - 0.5 * (x.nu+x.n) * log(x.nu*x.lambda + ssq) 

struct StdNormLogLik{T<:Real} <: Aloglik end
(x::StdNormLogLik{T})(det::T, ssq::T) where{T} = -0.5 * det - 0.5 * ssq 

