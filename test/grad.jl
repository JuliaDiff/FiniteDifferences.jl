using FiniteDifferences: grad, jacobian, _jvp, jvp, j′vp, _j′vp, to_vec

@testset "grad" begin

    @testset "jvp(::$T)" for T in (Float64,)
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        @testset "scalar output" begin
            x, y = randn(rng, T, N), randn(rng, T, M)
            ẋ, ẏ = randn(rng, T, N), randn(rng, T, M)
            xy, ẋẏ = vcat(x, y), vcat(ẋ, ẏ)
            ż_manual = _jvp(fdm, (xy)->sum(sin, xy), xy, ẋẏ)[1]
            ż_auto = jvp(fdm, x->sum(sin, x[1]) + sum(sin, x[2]), ((x, y), (ẋ, ẏ)))
            ż_multi = jvp(fdm, (x, y)->sum(sin, x) + sum(sin, y), (x, ẋ), (y, ẏ))
            @test ż_manual ≈ ż_auto
            @test ż_manual ≈ ż_multi
        end
        @testset "vector output" begin
            x, y = randn(rng, T, N), randn(rng, T, N)
            ẋ, ẏ = randn(rng, T, N), randn(rng, T, N)
            ż_manual = @. cos(x) * ẋ + cos(y) * ẏ
            ż_auto = jvp(fdm, x->sin.(x[1]) .+ sin.(x[2]), ((x, y), (ẋ, ẏ)))
            ż_multi = jvp(fdm, (x, y)->sin.(x) .+ sin.(y), (x, ẋ), (y, ẏ))
            @test ż_manual ≈ ż_auto
            @test ż_manual ≈ ż_multi
        end
        @testset "tuple output" begin
            x, y = randn(rng, T, N), randn(rng, T, N)
            ẋ, ẏ = randn(rng, T, N), randn(rng, T, N)
            ż_manual = (cos.(x) .* ẋ, cos.(y) .* ẏ)
            ż_auto = jvp(fdm, x->(sin.(x[1]), sin.(x[2])), ((x, y), (ẋ, ẏ)))
            ż_multi = jvp(fdm, (x, y)->(sin.(x), sin.(y)), (x, ẋ), (y, ẏ))
            @test ż_auto isa Tuple
            @test ż_multi isa Tuple
            @test collect(ż_manual) ≈ collect(ż_auto)
            @test collect(ż_manual) ≈ collect(ż_multi)
        end
    end

    @testset "grad(::$T)" for T in (Float64,)
        rng, fdm = MersenneTwister(123456), central_fdm(5, 1)
        x = randn(rng, T, 2)
        xc = copy(x)
        @test grad(fdm, x->sin(x[1]) + cos(x[2]), x)[1] ≈ [cos(x[1]), -sin(x[2])]
        @test xc == x
    end

    function check_jac_and_jvp_and_j′vp(fdm, f, ȳ::Array, x::Array, ẋ::Array, J_exact)
        xc = copy(x)

        # Validate inputs.
        @assert length(x) == length(ẋ)
        @assert length(ȳ) == length(f(x))

        # Check that the jacobian is as expected.
        J_fdm = jacobian(fdm, f, x)[1]
        @test size(J_fdm) == (length(ȳ), length(x))
        @test J_fdm ≈ J_exact
        @test J_fdm == jacobian(fdm, f, x)[1]

        # Check that the estimated jvp and j′vp are consistent with their definitions. 
        @test _jvp(fdm, f, x, ẋ) ≈ J_exact * ẋ
        @test _j′vp(fdm, f, ȳ, x) ≈ transpose(J_exact) * ȳ

        # Check that no mutation occured that wasn't reverted.
        @test xc == x
    end

    @testset "jacobian / _jvp / _j′vp (::$T)" for T in (Float64,)
        rng, P, Q, fdm = MersenneTwister(123456), 3, 2, central_fdm(5, 1)
        ȳ = randn(rng, T, P)
        A = randn(rng, T, P, Q)
        x = randn(rng, T, Q)
        ẋ = randn(rng, T, Q)
        Ac = copy(A)

        check_jac_and_jvp_and_j′vp(fdm, x->A * x, ȳ, x, ẋ, A)
        @test Ac == A
        check_jac_and_jvp_and_j′vp(fdm, x->sin.(A * x), ȳ, x, ẋ, cos.(A * x) .* A)
        @test Ac == A

        # Prevent regression against https://github.com/JuliaDiff/FiniteDifferences.jl/issues/67
        J = first(jacobian(fdm, identity, x))
        @test J ≈ one(Matrix{T}(undef, size(J)))
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
                x, y = rand(3, 3), 2.0
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
                x, y = rand(rng, 3, 3), 2.0
                dxs = grad(fdm, f2, x, y)
                @test dxs[1] ≈ grad(fdm, x->f2(x, y), x)[1]
                @test dxs[2] ≈ grad(fdm, y->f2(x, y), y)[1]
            end

            @testset "check tuple" begin
                x, y = rand(rng, 3, 3), 2.0
                dxs = grad(fdm, f3, (x, y))[1]
                @test dxs[1] ≈ grad(fdm, x->f3((x, y)), x)[1]
                @test dxs[2] ≈ grad(fdm, y->f3((x, y)), y)[1]   
            end

            @testset "check dict" begin
                x, y = rand(rng, 3, 3), 2.0
                d = Dict(:x=>x, :y=>y)
                dxs = grad(fdm, f4, d)[1]
                @test dxs[:x] ≈ grad(fdm, x->f3((x, y)), x)[1]
                @test dxs[:y] ≈ grad(fdm, y->f3((x, y)), y)[1]
            end
        end
    end

    @testset "j′vp(::$T)" for T in (Float64,)
        rng, N, M, fdm = MersenneTwister(123456), 2, 3, central_fdm(5, 1)
        x, y = randn(rng, T, N), randn(rng, T, M)
        z̄ = randn(rng, T, N + M)
        xy = vcat(x, y)
        x̄ȳ_manual = j′vp(fdm, xy->sin.(xy), z̄, xy)[1]
        x̄ȳ_auto = j′vp(fdm, x->sin.(vcat(x[1], x[2])), z̄, (x, y))[1]
        x̄ȳ_multi = j′vp(fdm, (x, y)->sin.(vcat(x, y)), z̄, x, y)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_auto...)
        @test x̄ȳ_manual ≈ vcat(x̄ȳ_multi...)
    end

    # Tests for complex numbers, to prevent regressions against
    # https://github.com/JuliaDiff/FiniteDifferences.jl/pull/76 and
    # https://github.com/JuliaDiff/FiniteDifferences.jl/issues/75
    @testset "Complex correctness - $T" for T in (ComplexF64,)
        rng = MersenneTwister(123456)
        x = randn(rng, T)
        y = randn(rng, T)
        fdm = FiniteDifferences.Central(5, 1)

        # Addition.
        dx, dy = FiniteDifferences.jacobian(fdm, +, x, y)
        @test dx ≈ [1 0; 0 1]
        @test dy ≈ [1 0; 0 1]

        # Negation.
        dx, = FiniteDifferences.jacobian(fdm, -, x)
        @test dx ≈ [-1 0; 0 -1]

        # Multiplication.
        dx, dy = FiniteDifferences.jacobian(fdm, *, x, y)
        @test dx ≈ [real(y) -imag(y); imag(y) real(y)]
        @test dy ≈ [real(x) -imag(x); imag(x) real(x)]

        # Magnitude
        dx, = FiniteDifferences.grad(fdm, abs2, x)
        @test real(dx) ≈ 2 * real(x)
        @test imag(dx) ≈ 2 * imag(x)

        # Concatenate tangents / cotangents.
        ẋ = randn(rng, T)
        ẏ = randn(rng, T)
        ẋẏ = vcat(first(to_vec(ẋ)), first(to_vec(ẏ)))
        z̄ = randn(rng, T)
        z̄_vec = first(to_vec(z̄))

        # Check jvp against jacobian result.
        f(x, y) = 5 * sin(x) * cos(y)
        Jx, Jy = FiniteDifferences.jacobian(fdm, f, x, y)
        ż = jvp(fdm, f, (x, ẋ), (y, ẏ))
        ż_vec = [real(ż), imag(ż)]
        ż_manual = hcat(Jx, Jy) * ẋẏ
        @test ż_vec ≈ ż_manual

        # Check j′vp against jacobian result.
        x̄, ȳ = j′vp(fdm, f, z̄, x, y)
        @test [real(x̄), imag(x̄)] ≈ Jx'z̄_vec
        @test [real(ȳ), imag(ȳ)] ≈ Jy'z̄_vec
    end
end
