@testset "Methods" begin
    @testset "Correctness" begin
        # Finite difference methods to test.
        methods = [forward_fdm, backward_fdm, central_fdm]

        # The different floating point types to try.
        types = [Float32, Float64]

        # The different functions to evaluate (`.f`), their first derivative at 1 (`.d1`),
        # and their second derivative at 1 (`.d2`).
        fs = [
            (f=sin, d1=cos(1), d2=-sin(1), only_forward=false),
            (f=exp, d1=exp(1), d2=exp(1), only_forward=false),
            (f=abs2, d1=2, d2=2, only_forward=false),
            (f=x -> sqrt(x + 1), d1=0.5 / sqrt(2), d2=-0.25 / 2^(3/2), only_forward=true),
        ]

        # Test all combinations of the above settings, i.e. differentiate all functions
        # using all methods and data types.
        @testset "f=$(f.f), method=$m, type=$(T)" for f in fs, m in methods, T in types
            # Check if only `forward_fdm` is allowed.
            f.only_forward && m != forward_fdm && continue

            @testset "method-order=$order" for order in [1, 2, 3, 4, 5]
                @test m(order, 0; adapt=2)(f.f, T(1)) isa T
                @test m(order, 0; adapt=2)(f.f, T(1)) == T(f.f(1))
            end

            @test m(10, 1)(f.f, T(1)) isa T
            @test m(10, 1)(f.f, T(1)) ≈ T(f.d1)

            @test m(10, 2)(f.f, T(1)) isa T
            T == Float64 && @test m(10, 2)(f.f, T(1)) ≈ T(f.d2)
        end
    end

    @testset "Accuracy" begin
        # Finite difference methods to test.
        methods = [forward_fdm, backward_fdm, central_fdm]

        # `f`s, `x`s, the derivatives of `f` at `x`, and a factor that loosens tolerances.
        fs = [
            (f=x -> 0, x=0, d=0, factor=1),
            (f=x -> x, x=0, d=1, factor=1),
            (f=exp, x=0, d=1, factor=1),
            (f=sin, x=0, d=1, factor=1),
            (f=cos, x=0, d=0, factor=1),
            (f=sinc, x=0, d=0, factor=1),
            (f=cosc, x=0, d=-(pi ^ 2) / 3, factor=10)
        ]
        @testset "f=$(f.f), method=$m" for f in fs, m in methods
            @test m(4, 1)(f.f, f.x) ≈ f.d rtol=0 atol=5e-8 * f.factor
            @test m(5, 1)(f.f, f.x) ≈ f.d rtol=0 atol=1e-8 * f.factor
            @test m(6, 1)(f.f, f.x) ≈ f.d rtol=0 atol=1e-9 * f.factor
            @test m(7, 1)(f.f, f.x) ≈ f.d rtol=0 atol=5e-10 * f.factor
            @test m(8, 1)(f.f, f.x) ≈ f.d rtol=0 atol=1e-10 * f.factor
            @test m(9, 1)(f.f, f.x) ≈ f.d rtol=0 atol=5e-11 * f.factor
            @test m(10, 1)(f.f, f.x) ≈ f.d rtol=0 atol=1e-11 * f.factor
            @test m(11, 1)(f.f, f.x) ≈ f.d rtol=0 atol=5e-12 * f.factor
        end
    end

    @testset "Derivative of cosc at 0 (issue #124)" begin
        @test central_fdm(10, 1, adapt=3)(cosc, 0) ≈ -(pi ^ 2) / 3 atol=5e-13
    end

    @testset "Derivative of a constant (issue #125)" begin
        @test central_fdm(2, 1)(x -> 0, 0) ≈ 0 atol=1e-10
    end

    @testset "Test custom grid" begin
        @test FiniteDifferenceMethod([-2, 0, 5], 1)(sin, 1) ≈ cos(1)
    end

    @testset "Test allocations" begin
        m = central_fdm(5, 2, adapt=2)
        @test (@benchmark $m(sin, 1; samples=1, evals=1)).allocs == 0
    end

    # Integration test to ensure that Integer-output functions can be tested.
    @testset "Integer output" begin
        @test isapprox(central_fdm(5, 1)(x -> 5, 0), 0; rtol=1e-12, atol=1e-12)
    end

    @testset "Adaptation improves estimate" begin
        @test abs(forward_fdm(5, 1, adapt=0)(log, 0.001) - 1000) > 1
        @test forward_fdm(5, 1, adapt=1)(log, 0.001) ≈ 1000
    end

    @testset "Limiting step size" begin
        @test !isfinite(central_fdm(5, 1, max_range=0)(abs, 0.001))
        @test central_fdm(10, 1, max_range=9e-4)(log, 1e-3) ≈ 1000
        @test central_fdm(5, 1)(abs, 0.001) ≈ 1.0
    end

    @testset "Accuracy at high orders and high adaptation (issue #64)" begin
        # Regression test against issues with precision during computation of coefficients.
        @test central_fdm(9, 5, adapt=4)(exp, 1.0) ≈ exp(1) atol=2e-7
        @test central_fdm(15, 5, adapt=2)(exp, 1.0) ≈ exp(1) atol=1e-10
        poly(x) = 4x^3 + 3x^2 + 2x + 1
        @test central_fdm(9, 3, adapt=4)(poly, 1.0) ≈ 24 atol=1e-11
    end

    @testset "Printing FiniteDifferenceMethods" begin
        @test sprint(show, "text/plain", central_fdm(2, 1)) == """
            FiniteDifferenceMethod:
              order of method:       2
              order of derivative:   1
              grid:                  [-1, 1]
              coefficients:          [-0.5, 0.5]
            """
    end

    @testset "_is_symmetric" begin
        # Test odd grids:
        @test FiniteDifferences._is_symmetric(SVector{5}(2, 1, 0, 1, 2))
        @test !FiniteDifferences._is_symmetric(SVector{5}(2, 1, 0, 3, 2))
        @test !FiniteDifferences._is_symmetric(SVector{5}(4, 1, 0, 1, 2))

        # Test even grids:
        @test FiniteDifferences._is_symmetric(SVector{4}(2, 1, 1, 2))
        @test !FiniteDifferences._is_symmetric(SVector{4}(2, 1, 3, 2))
        @test !FiniteDifferences._is_symmetric(SVector{4}(4, 1, 1, 2))

        # Test zero at centre:
        @test FiniteDifferences._is_symmetric(SVector{5}(2, 1, 4, 1, 2))
        @test !FiniteDifferences._is_symmetric(SVector{5}(2, 1, 4, 1, 2), centre_zero=true)
        @test FiniteDifferences._is_symmetric(SVector{4}(2, 1, 1, 2), centre_zero=true)

        # Test negation of a half:
        @test !FiniteDifferences._is_symmetric(SVector{4}(2, 1, -1, -2))
        @test FiniteDifferences._is_symmetric(SVector{4}(2, 1, -1, -2), negate_half=true)
        @test FiniteDifferences._is_symmetric(SVector{5}(2, 1, 0, -1, -2), negate_half=true)
        @test FiniteDifferences._is_symmetric(SVector{5}(2, 1, 4, -1, -2), negate_half=true)
        @test !FiniteDifferences._is_symmetric(
            SVector{5}(2, 1, 4, -1, -2);
            negate_half=true,
            centre_zero=true
        )

        # Test symmetry of `central_fdm`.
        for p in 2:10
            m = central_fdm(p, 1)
            @test FiniteDifferences._is_symmetric(m)
        end

        # Test asymmetry of `forward_fdm` and `backward_fdm`.
        for p in 2:10
            for f in [forward_fdm, backward_fdm]
                m = f(p, 1)
                @test !FiniteDifferences._is_symmetric(m)
            end
        end
    end

    @testset "extrapolate_fdm" begin
        # Also test an `Integer` argument as input.
        for x in [1, 1.0]
            for f in [forward_fdm, central_fdm, backward_fdm]
                estimate, _ = extrapolate_fdm(f(4, 3), exp, x, contract=0.8)
                @test estimate ≈ exp(1.0) atol=1e-7
            end
        end
    end

end
