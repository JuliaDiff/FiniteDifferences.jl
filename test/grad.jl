using FDM: grad, jacobian, adjoint

@testset "grad" begin
    x = randn(MersenneTwister(123456), 2)
    xc = copy(x)
    @test grad(central_fdm(5, 1), x->sin(x[1]) + cos(x[2]), x) ≈ [cos(x[1]), -sin(x[2])]
    @test xc == x
end

function check_jac_and_adjoint(fdm, f, ȳ, x, J_exact)
    xc = copy(x)
    @test jacobian(fdm, f, x, length(ȳ)) ≈ J_exact
    @test jacobian(fdm, f, x) == jacobian(fdm, f, x, length(ȳ))
    @test adjoint(fdm, f, ȳ, x) ≈ J_exact' * ȳ
    @test xc == x
end

@testset "jacobian / adjoint" begin
    rng, P, Q, fdm = MersenneTwister(123456), 3, 2, central_fdm(5, 1)
    ȳ, A, x = randn(rng, P), randn(rng, P, Q), randn(rng, Q)
    Ac = copy(A)

    check_jac_and_adjoint(fdm, x->A * x, ȳ, x, A)
    @test Ac == A
    check_jac_and_adjoint(fdm, x->sin.(A * x), ȳ, x, cos.(A * x) .* A)
    @test Ac == A
end
