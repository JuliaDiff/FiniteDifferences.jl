using FDM: grad, jacobian, jvp, j′vp

@testset "grad" begin
    x = randn(MersenneTwister(123456), 2)
    xc = copy(x)
    @test grad(central_fdm(5, 1), x->sin(x[1]) + cos(x[2]), x) ≈ [cos(x[1]), -sin(x[2])]
    @test xc == x
end

function check_jac_and_jvp_and_j′vp(fdm, f, ȳ, x, ẋ, J_exact)
    xc = copy(x)
    @test jacobian(fdm, f, x, length(ȳ)) ≈ J_exact
    @test jacobian(fdm, f, x) == jacobian(fdm, f, x, length(ȳ))
    @test jvp(fdm, f, x, ẋ) ≈ J_exact * ẋ
    @test j′vp(fdm, f, ȳ, x) ≈ J_exact' * ȳ
    @test xc == x
end

@testset "jacobian / jvp / j′vp" begin
    rng, P, Q, fdm = MersenneTwister(123456), 3, 2, central_fdm(5, 1)
    ȳ, A, x, ẋ = randn(rng, P), randn(rng, P, Q), randn(rng, Q), randn(rng, Q)
    Ac = copy(A)

    check_jac_and_jvp_and_j′vp(fdm, x->A * x, ȳ, x, ẋ, A)
    @test Ac == A
    check_jac_and_jvp_and_j′vp(fdm, x->sin.(A * x), ȳ, x, ẋ, cos.(A * x) .* A)
    @test Ac == A
end
