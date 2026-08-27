using BASSGWAS: ParState2, 
    makeSCache, 
    TDistLogLik, 
    init_state2, 
    update_state2!, 
    _update_state_test!, 
    updateSCache!,
    flip_gamma!,
    comp_ll2,
    _direct_ll,
    get_X1,
    _ssq_direct,
    _det_direct,
    _comp_ssq,
    _comp_llterms
    
    
using LinearAlgebra
using Random

Random.seed!(1)
BLAS.set_num_threads(1)

# Linear algebra tests
# Use explicit calculations as a ground truth
let n = 300, d=300, p=10000, alpha = 0.5, beta = 0.3, nu=3.0, lambda=2.0, r=5.0, c=60.0, s=1.0, mc=100.0

    K = randn(n,n);
    K = Hermitian(K'K + Diagonal(ones(n))*0.001);

    X = rand(0:1,n,p);
    Y = randn(n);
    I = Diagonal(ones(n));
    
    γ0 = zeros(Int64, p)
    aset = rand(1:p, 10)
    γ0[aset] .= 1

    ns = max(300, n)
    sv = svd(K)
    U = sv.U[:, 1:ns]
    S = sv.S[1:ns]

    parState2 = ParState2(X, Y, U, S, alpha, beta, c, r, s, mc)
    sc = makeSCache(parState2)

    loglik = TDistLogLik(nu, lambda, n)

    s2, qc2 = init_state2(loglik, parState2, sc, γ0)
    s3, qc3 = init_state2(loglik, parState2, sc, γ0)

    #update_state!(s1, parState1, cc)
    update_state2!(s2, parState2, sc, qc2, loglik)
    _update_state_test!(s3, parState2, sc, qc2, loglik)

    @test s2.dets ≈ s3.dets
    @test s2.ssq ≈ s3.ssq

    updateSCache!(sc, parState2)
    update_state2!(s2, parState2, sc, qc2, loglik)

    @test s2.dets ≈ s3.dets
    @test s2.ssq ≈ s3.ssq

    flip_gamma!(s2, parState2, qc2, 301)
    flip_gamma!(s3, parState2, qc3, 301)

    update_state2!(s2, parState2, sc, qc2, loglik)
    ll_curr2 = s2.ll_curr
    _update_state_test!(s3, parState2, sc, qc3,loglik)
    ll_curr3 = s3.ll_curr
    ll_d = comp_ll2(sc, qc2, parState2, s2.γ, loglik)
    ll_dir = _direct_ll(parState2.Y, parState2.r, parState2.c, parState2.mc, get_X1(qc2), sc.U, sc.S, nu, lambda)

    @test s2.dets ≈ s3.dets
    @test s2.ssq ≈ s3.ssq

    @test ll_curr2 ≈ ll_curr3
    @test ll_curr2 ≈ ll_d
    @test ll_curr2 ≈ ll_dir

    Xsq = get_X1(qc2)'*get_X1(qc2)
    ssq_gt = _ssq_direct(parState2.Y, parState2.Y, parState2.r, parState2.c, parState2.mc, get_X1(qc2), sc.U, sc.S)
    ssq_f = _comp_ssq(parState2.Y, parState2.Y, parState2.r, parState2.c, parState2.mc, get_X1(qc2), Xsq, sc.UtX[:, qc2.i1_set], sc.U, sc.S)
    ssq_f2 = _comp_ssq(parState2.Y, parState2.r, parState2.c, parState2.mc, get_X1(qc2), Xsq, sc.UtX[:, qc2.i1_set], sc.U, sc.S)
    @test ssq_gt ≈ ssq_f
    @test ssq_gt ≈ ssq_f2

    ssq_c, det_c = _comp_llterms(parState2.Y, parState2.r, parState2.c, parState2.mc, get_X1(qc2), Xsq, sc.UtX[:, qc2.i1_set], sc.YtU, sc.S)
    det_gt = _det_direct(parState2.r, parState2.c, parState2.mc, get_X1(qc2), sc.U, sc.S)
    @test ssq_gt ≈ ssq_c
    @test det_gt ≈ det_c

    ssq_gt = _ssq_direct(parState2.Y, ones(size(parState2.Y)), parState2.r, parState2.c, parState2.mc, get_X1(qc2), sc.U, sc.S)
    ssq_f = _comp_ssq(parState2.Y, ones(size(parState2.Y)), parState2.r, parState2.c, parState2.mc, get_X1(qc2), Xsq, sc.UtX[:, qc2.i1_set], sc.U, sc.S)
    
    @test ssq_gt ≈ ssq_f

    parState2.Y = randn(n)
    updateSCache!(sc, parState2)
    update_state2!(s2, parState2, sc, qc2, loglik)
    _update_state_test!(s3, parState2, sc, qc2, loglik)

    @test s2.dets ≈ s3.dets
    @test s2.ssq ≈ s3.ssq
end