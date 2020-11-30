import FiniteDifferences: estimate_magitude

@testset "Methods" begin
    @testset "estimate_magitude" begin
        f64(x::Float64) = x
        f64_int(x::Float64) = Int(x)
        @assert estimate_magitude(f64, 1.0) == 1.0
        @assert estimate_magitude(f64_int, 1.0) == 1.0

        f32(x::Float32) = x
        f32_int(x::Float32) = Int(x)
        @assert estimate_magitude(f32, 1f0) == 1f0
        @assert estimate_magitude(f32_int, 1f0) == 1f0
    end


    # The different approaches to approximating the gradient to try.
    methods = [forward_fdm, backward_fdm, central_fdm]

    # The different floating point types to try and the associated required relative
    # tolerance.
    types = [Float32, Float64]

    # The different functions to evaluate (.f), their first derivative at 1 (.d1),
    # and second derivative at 1 (.d2).
    foos = [
        (f=sin, d1=cos(1), d2=-sin(1)),
        (f=exp, d1=exp(1), d2=exp(1)),
        (f=abs2, d1=2, d2=2),
        (f=x -> sqrt(x + 1), d1=0.5 / sqrt(2), d2=-0.25 / 2^(3/2)),
    ]

    # Test all combinations of the above settings, i.e. differentiate all functions using
    # all methods and data types.
    @testset "foo=$(foo.f), method=$m, type=$(T)" for foo in foos, m in methods, T in types
        @testset "method-order=$order" for order in [1, 2, 3, 4, 5]
            @test m(order, 0, adapt=2)(foo.f, T(1)) isa T
            @test m(order, 0, adapt=2)(foo.f, T(1)) == T(foo.f(1))
        end

        @test m(10, 1)(foo.f, T(1)) isa T
        @test m(10, 1)(foo.f, T(1)) ≈ T(foo.d1)

        @test m(10, 2)(foo.f, T(1)) isa T
        if T == Float64
            @test m(10, 2)(foo.f, T(1)) ≈ T(foo.d2)
        end
    end

    # Integration test to ensure that Integer-output functions can be tested.
    @testset "Integer output" begin
        @test isapprox(central_fdm(5, 1)(x -> 5, 0), 0; rtol=1e-12, atol=1e-12)
    end

    @testset "Adaptation improves estimate" begin
        @test forward_fdm(5, 1, adapt=0)(log, 0.001) ≈ 997.077814
        @test forward_fdm(5, 1, adapt=1)(log, 0.001) ≈ 1000
    end

    @testset "Limiting step size" begin
        @test !isfinite(central_fdm(5, 1)(abs, 0.001, max_step=0))
        @test central_fdm(5, 1)(abs, 0.001) ≈ 1.0
    end

    @testset "Accuracy at high orders, with high adapt" begin
        # Regression test against issues with precision during computation of _coeffs
        # see https://github.com/JuliaDiff/FiniteDifferences.jl/issues/64

        @test central_fdm(9, 5, adapt=4, condition=1)(exp, 1.0) ≈ exp(1) atol=1e-7

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
        @test FiniteDifferences._is_symmetric([2, 1, 0, 1, 2])
        @test !FiniteDifferences._is_symmetric([2, 1, 0, 3, 2])
        @test !FiniteDifferences._is_symmetric([4, 1, 0, 1, 2])

        # Test even grids:
        @test FiniteDifferences._is_symmetric([2, 1, 1, 2])
        @test !FiniteDifferences._is_symmetric([2, 1, 3, 2])
        @test !FiniteDifferences._is_symmetric([4, 1, 1, 2])

        # Test zero at centre:
        @test FiniteDifferences._is_symmetric([2, 1, 4, 1, 2])
        @test !FiniteDifferences._is_symmetric([2, 1, 4, 1, 2], centre_zero=true)
        @test FiniteDifferences._is_symmetric([2, 1, 1, 2], centre_zero=true)

        # Test negation of a half:
        @test !FiniteDifferences._is_symmetric([2, 1, -1, -2])
        @test FiniteDifferences._is_symmetric([2, 1, -1, -2], negate_half=true)
        @test FiniteDifferences._is_symmetric([2, 1, 0, -1, -2], negate_half=true)
        @test FiniteDifferences._is_symmetric([2, 1, 4, -1, -2], negate_half=true)
        @test !FiniteDifferences._is_symmetric(
            [2, 1, 4, -1, -2],
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

    @testset "Derivative of cosc at 0 (#124)" begin
        @test central_fdm(5, 1)(cosc, 0) ≈ -(pi ^ 2) / 3 atol=1e-9
        @test central_fdm(10, 1, adapt=3)(cosc, 0) ≈ -(pi ^ 2) / 3 atol=5e-14
    end
end
