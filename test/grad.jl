using FiniteDifferences: grad, jacobian, _jvp, _j′vp, jvp, j′vp, to_vec

# Dummy type where length(x::DummyType) ≠ length(first(to_vec(x)))
struct DummyType{TX<:Matrix}
    X::TX
end

function FiniteDifferences.to_vec(x::DummyType)
    x_vec, back = to_vec(x.X)
    return x_vec, x_vec -> DummyType(back(x_vec))
end

Base.:(==)(x::DummyType, y::DummyType) = x.X == y.X
Base.length(x::DummyType) = size(x.X, 1)


@testset "grad" begin

    @testset "grad(::$T)" for T in (Float64, ComplexF64)
        rng, fdm = MersenneTwister(123456), central_fdm(5, 1)
        x = randn(rng, T, 2)
        xc = copy(x)
        @test grad(fdm, x->sin(x[1]) + cos(x[2]), x)[1] ≈ [cos(x[1]), -sin(x[2])]
        @test xc == x
    end

    function check_jac_and_jvp_and_j′vp(fdm, f, ȳ, x, ẋ, J_exact)
        xc = copy(x)
        @test jacobian(fdm, f, x; len=length(ȳ))[1] ≈ J_exact
        @test jacobian(fdm, f, x)[1] == jacobian(fdm, f, x; len=length(ȳ))[1]
        @test _jvp(fdm, f, x, ẋ) ≈ J_exact * ẋ
        @test _j′vp(fdm, f, ȳ, x) ≈ transpose(J_exact) * ȳ
        @test xc == x
    end

    @testset "jacobian / _jvp / _j′vp (::$T)" for T in (Float64, ComplexF64)
        rng, P, Q, fdm = MersenneTwister(123456), 3, 2, central_fdm(5, 1)
        ȳ, A, x, ẋ = randn(rng, T, P), randn(rng, T, P, Q), randn(rng, T, Q), randn(rng, T, Q)
        Ac = copy(A)

        check_jac_and_jvp_and_j′vp(fdm, x->A * x, ȳ, x, ẋ, A)
        @test Ac == A
        check_jac_and_jvp_and_j′vp(fdm, x->sin.(A * x), ȳ, x, ẋ, cos.(A * x) .* A)
        @test Ac == A
    end

    @testset "multi vars jacobian/grad" begin
        rng, fdm = MersenneTwister(123456), central_fdm(5, 1)
        
        f1(x, y) = x * y + x
        f2(x, y) = sum(x * y + x)
        f3(x::Tuple) = sum(x[1]) + x[2]
        f4(d::Dict) = sum(d[:x]) + d[:y]

        @testset "jacobian" begin
            @testset "check multiple matrices" begin
                x, y = rand(rng, 3, 3), rand(rng, 3, 3)
                jac_xs = jacobian(fdm, f1, x, y)
                @test jac_xs[1] ≈ jacobian(fdm, x->f1(x, y), x)[1]
                @test jac_xs[2] ≈ jacobian(fdm, y->f1(x, y), y)[1]
            end

            @testset "check mixed scalar and matrices" begin
                x, y = rand(3, 3), 2
                jac_xs = jacobian(fdm, f1, x, y)
                @test jac_xs[1] ≈ jacobian(fdm, x->f1(x, y), x)[1]
                @test jac_xs[2] ≈ jacobian(fdm, y->f1(x, y), y)[1]
            end
        end

        @testset "grad" begin
            @testset "check multiple matrices" begin
                x, y = rand(rng, 3, 3), rand(rng, 3, 3)
                dxs = grad(fdm, f2, x, y)
                @test dxs[1] ≈ grad(fdm, x->f2(x, y), x)[1]
                @test dxs[2] ≈ grad(fdm, y->f2(x, y), y)[1]
            end

            @testset "check mixed scalar & matrices" begin
                x, y = rand(rng, 3, 3), 2
                dxs = grad(fdm, f2, x, y)
                @test dxs[1] ≈ grad(fdm, x->f2(x, y), x)[1]
                @test dxs[2] ≈ grad(fdm, y->f2(x, y), y)[1]
            end

            @testset "check tuple" begin
                x, y = rand(rng, 3, 3), 2
                dxs = grad(fdm, f3, (x, y))[1]
                @test dxs[1] ≈ grad(fdm, x->f3((x, y)), x)[1]
                @test dxs[2] ≈ grad(fdm, y->f3((x, y)), y)[1]   
            end

            @testset "check dict" begin
                x, y = rand(rng, 3, 3), 2
                d = Dict(:x=>x, :y=>y)
                dxs = grad(fdm, f4, d)[1]
                @test dxs[:x] ≈ grad(fdm, x->f3((x, y)), x)[1]
                @test dxs[:y] ≈ grad(fdm, y->f3((x, y)), y)[1]
            end
        end
    end

    function test_to_vec(x)
        x_vec, back = to_vec(x)
        @test x_vec isa Vector
        @test x == back(x_vec)
        return nothing
    end

    @testset "to_vec(::$T)" for T in (Float64, ComplexF64)
        if T == Float64
            test_to_vec(1.0)
            test_to_vec(1)
        else
            test_to_vec(.7 + .8im)
            test_to_vec(1 + 2im)
        end
        test_to_vec(randn(T, 3))
        test_to_vec(randn(T, 5, 11))
        test_to_vec(randn(T, 13, 17, 19))
        test_to_vec(randn(T, 13, 0, 19))
        test_to_vec([1.0, randn(T, 2), randn(T, 1), 2.0])
        test_to_vec([randn(T, 5, 4, 3), (5, 4, 3), 2.0])
        test_to_vec(reshape([1.0, randn(T, 5, 4, 3), randn(T, 4, 3), 2.0], 2, 2))
        test_to_vec(UpperTriangular(randn(T, 13, 13)))
        test_to_vec(Symmetric(randn(T, 11, 11)))
        test_to_vec(Diagonal(randn(T, 7)))
        test_to_vec(DummyType(randn(T, 2, 9)))
    
        @testset "$Op" for Op in (Adjoint, Transpose)
            test_to_vec(Op(randn(T, 4, 4)))
            test_to_vec(Op(randn(T, 6)))
            test_to_vec(Op(randn(T, 2, 5)))
        end
    
        @testset "Tuples" begin
            test_to_vec((5, 4))
            test_to_vec((5, randn(T, 5)))
            test_to_vec((randn(T, 4), randn(T, 4, 3, 2), 1))
            test_to_vec((5, randn(T, 4, 3, 2), UpperTriangular(randn(T, 4, 4)), 2.5))
            test_to_vec(((6, 5), 3, randn(T, 3, 2, 0, 1)))
            test_to_vec((DummyType(randn(T, 2, 7)), DummyType(randn(T, 3, 9))))
            test_to_vec((DummyType(randn(T, 3, 2)), randn(T, 11, 8)))
        end
        @testset "Dictionary" begin
            if T == Float64
                test_to_vec(Dict(:a=>5, :b=>randn(10, 11), :c=>(5, 4, 3)))
            else
                test_to_vec(Dict(:a=>3 + 2im, :b=>randn(T, 10, 11), :c=>(5+im, 2-im, 1+im)))
            end
        end
    end

    @testset "jvp(::$T)" for T in (Float64, ComplexF64)
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        x, y = randn(rng, T, N), randn(rng, T, M)
        ẋ, ẏ = randn(rng, T, N), randn(rng, T, M)
        xy, ẋẏ = vcat(x, y), vcat(ẋ, ẏ)
        ż_manual = _jvp(fdm, (xy)->sum(sin, xy), xy, ẋẏ)[1]
        ż_auto = jvp(fdm, x->sum(sin, x[1]) + sum(sin, x[2]), ((x, y), (ẋ, ẏ)))
        ż_multi = jvp(fdm, (x, y)->sum(sin, x) + sum(sin, y), (x, ẋ), (y, ẏ))
        @test ż_manual ≈ ż_auto
        @test ż_manual ≈ ż_multi
    end

    @testset "j′vp(::$T)" for T in (Float64, ComplexF64)
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        x, y = randn(rng, T, N), randn(rng, T, M)
        z̄ = randn(rng, T, N + M)
        xy = vcat(x, y)
        x̄ȳ_manual = j′vp(fdm, xy->sin.(xy), z̄, xy)
        x̄ȳ_auto = j′vp(fdm, x->sin.(vcat(x[1], x[2])), z̄, (x, y))
        x̄ȳ_multi = j′vp(fdm, (x, y)->sin.(vcat(x, y)), z̄, x, y)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_auto...)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_multi...)
    end
end
