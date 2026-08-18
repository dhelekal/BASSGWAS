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

function ParState2(X, Y, U, S, alpha, beta, c, r, s, mc)
    alpha, beta, c, r, s = promote(alpha, beta, c, r, s, mc)
    p = size(X, 2)    
    n = size(Y, 1)

    Xaug = [X ones(n)]
    #Xaug = [X ones(n)]
    return ParState2(Xaug, Y, U, S, p, n, alpha, beta, c, r, s, mc)
end