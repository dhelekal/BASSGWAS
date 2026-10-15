abstract type EIGest end

function eig(eig::EIGest, idx::Vector{T}) where T<:Integer
    error("Unimplemented EIG estimator")
end

function update_eig(eig::EIGest, idx::T) where T<:Integer
    error("Unimplemented EIG estimator")
end