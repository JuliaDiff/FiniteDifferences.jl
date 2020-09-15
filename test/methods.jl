using FiniteDifferences: add_tiny

@testset "Methods" begin

    @testset "add_tiny" begin
        @test add_tiny(convert(Float64, 5)) isa Float64
        @test add_tiny(convert(Float32, 5)) isa Float32
        @test add_tiny(convert(Float16, 5)) isa Float16

        @test add_tiny(convert(Int, 5)) isa Float64
        @test add_tiny(convert(UInt, 5)) isa Float64
        @test add_tiny(convert(Bool, 1)) isa Float64
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
        @test sprint(show, central_fdm(2, 1)) == """
            FiniteDifferenceMethod:
              order of method:       2
              order of derivative:   1
              grid:                  [-1, 1]
              coefficients:          [-0.5, 0.5]
            """
    end

    @testset "extrapolate_fdm" begin
        # Also test an `Integer` argument as input.
        for x in [1, 1.0]
            estimate, _ = extrapolate_fdm(forward_fdm(4, 3), exp, x, contract=0.8)
            @test estimate ≈ exp(1.0) atol=1e-7
        end
    end
end
