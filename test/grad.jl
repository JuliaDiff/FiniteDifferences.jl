using FDM: grad, jacobian, _jvp, _j′vp, jvp, j′vp, to_vec

# Dummy type where length(x::DummyType) ≠ length(first(to_vec(x)))
struct DummyType{TX<:Matrix}
    X::TX
end

function FDM.to_vec(x::DummyType)
    x_vec, back = to_vec(x.X)
    return x_vec, x_vec -> DummyType(back(x_vec))
end

Base.:(==)(x::DummyType, y::DummyType) = x.X == y.X
Base.length(x::DummyType) = size(x.X, 1)


@testset "grad" begin

    @testset "grad" begin
        rng, fdm = MersenneTwister(123456), central_fdm(5, 1)
        x = randn(rng, 2)
        xc = copy(x)
        @test grad(fdm, x->sin(x[1]) + cos(x[2]), x) ≈ [cos(x[1]), -sin(x[2])]
        @test xc == x
    end

    function check_jac_and_jvp_and_j′vp(fdm, f, ȳ, x, ẋ, J_exact)
        xc = copy(x)
        @test jacobian(fdm, f, x, length(ȳ)) ≈ J_exact
        @test jacobian(fdm, f, x) == jacobian(fdm, f, x, length(ȳ))
        @test _jvp(fdm, f, x, ẋ) ≈ J_exact * ẋ
        @test _j′vp(fdm, f, ȳ, x) ≈ J_exact' * ȳ
        @test xc == x
    end

    @testset "jacobian / _jvp / _j′vp" begin
        rng, P, Q, fdm = MersenneTwister(123456), 3, 2, central_fdm(5, 1)
        ȳ, A, x, ẋ = randn(rng, P), randn(rng, P, Q), randn(rng, Q), randn(rng, Q)
        Ac = copy(A)

        check_jac_and_jvp_and_j′vp(fdm, x->A * x, ȳ, x, ẋ, A)
        @test Ac == A
        check_jac_and_jvp_and_j′vp(fdm, x->sin.(A * x), ȳ, x, ẋ, cos.(A * x) .* A)
        @test Ac == A
    end

    function test_to_vec(x)
        x_vec, back = to_vec(x)
        @test x_vec isa Vector
        @test x == back(x_vec)
        return nothing
    end

    @testset "to_vec" begin
        test_to_vec(1.0)
        test_to_vec(1)
        test_to_vec(randn(3))
        test_to_vec(randn(5, 11))
        test_to_vec(randn(13, 17, 19))
        test_to_vec(randn(13, 0, 19))
        test_to_vec(UpperTriangular(randn(13, 13)))
        test_to_vec(Symmetric(randn(11, 11)))
        test_to_vec(Diagonal(randn(7)))
        test_to_vec(DummyType(randn(2, 9)))

        @testset "$T" for T in (Adjoint, Transpose)
            test_to_vec(T(randn(4, 4)))
            test_to_vec(T(randn(6)))
            test_to_vec(T(randn(2, 5)))
        end

        @testset "Tuples" begin
            test_to_vec((5, 4))
            test_to_vec((5, randn(5)))
            test_to_vec((randn(4), randn(4, 3, 2), 1))
            test_to_vec((5, randn(4, 3, 2), UpperTriangular(randn(4, 4)), 2.5))
            test_to_vec(((6, 5), 3, randn(3, 2, 0, 1)))
            test_to_vec((DummyType(randn(2, 7)), DummyType(randn(3, 9))))
            test_to_vec((DummyType(randn(3, 2)), randn(11, 8)))
        end
    end

    @testset "jvp" begin
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        x, y = randn(rng, N), randn(rng, M)
        ẋ, ẏ = randn(rng, N), randn(rng, M)
        xy, ẋẏ = vcat(x, y), vcat(ẋ, ẏ)
        ż_manual = _jvp(fdm, (xy)->sum(sin, xy), xy, ẋẏ)[1]
        ż_auto = jvp(fdm, x->sum(sin, x[1]) + sum(sin, x[2]), ((x, y), (ẋ, ẏ)))
        ż_multi = jvp(fdm, (x, y)->sum(sin, x) + sum(sin, y), (x, ẋ), (y, ẏ))
        @test ż_manual ≈ ż_auto
        @test ż_manual ≈ ż_multi
    end

    @testset "j′vp" begin
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        x, y = randn(rng, N), randn(rng, M)
        z̄ = randn(rng, N + M)
        xy = vcat(x, y)
        x̄ȳ_manual = j′vp(fdm, xy->sin.(xy), z̄, xy)
        x̄ȳ_auto = j′vp(fdm, x->sin.(vcat(x[1], x[2])), z̄, (x, y))
        x̄ȳ_multi = j′vp(fdm, (x, y)->sin.(vcat(x, y)), z̄, x, y)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_auto...)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_multi...)
    end
end
