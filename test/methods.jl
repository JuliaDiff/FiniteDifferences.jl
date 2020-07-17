using FiniteDifferences: Forward, Backward, Central, Nonstandard

@testset "Methods" begin

    # The different approaches to approximating the gradient to try.
    methods = [forward_fdm, backward_fdm, central_fdm]

    # The different floating-point types to try.
    types = [Float32, Float64]

    # The different functions to evaluate (.f), their first derivative at 1 (.d1),
    # and second derivative at 1 (.d2).
    foos = [
        (f=sin, d1=cos(1), d2=-sin(1)),
        (f=exp, d1=exp(1), d2=exp(1)),
        (f=abs2, d1=2, d2=2),
        (f=x -> sqrt(x + 1), d1=0.5 / sqrt(2), d2=-0.25 / 2^(3/2)),
    ]

    # Test all combinations of the above settings. i.e. differentiate all functions using
    # all methods and data types.
    @testset "foo=$(foo.f), method=$m, type=$T" for foo in foos, m in methods, T in types

        @testset "method-order=$order" for order in [1, 2, 3]
            @test m(order, 0; bound=1)(foo.f, T(1)) isa T
            @test m(order, 0; bound=1)(foo.f, T(1)) == T(foo.f(1))
        end

        @test m(10, 1; bound=1)(foo.f, T(1)) isa T
        @test m(10, 1; bound=1)(foo.f, T(1)) ≈ T(foo.d1)

        @test m(10, 2; bound=1)(foo.f, T(1)) isa T
        if T == Float64
            @test m(10, 2; bound=1)(foo.f, T(1)) ≈ T(foo.d2)
        end
    end

    @testset "Adaptation improves estimate" begin
        @test forward_fdm(5, 1)(log, 0.001; adapt=0) ≈ 969.2571703
        @test forward_fdm(5, 1)(log, 0.001; adapt=1) ≈ 1000
    end

    @testset "Limiting step size" begin
        @test !isfinite(central_fdm(5, 1)(abs, 0.001; max_step=0))
        @test central_fdm(5, 1)(abs, 0.001) ≈ 1.0
    end

    @testset "Accuracy at high orders, with high adapt" begin
        # Regression test against issues with precision during computation of _coeffs
        # see https://github.com/JuliaDiff/FiniteDifferences.jl/issues/64

        @test fdm(central_fdm(9, 5), exp, 1.0, adapt=4) ≈ exp(1) atol=1e-7

        poly(x) = 4x^3 + 3x^2 + 2x + 1
        @test fdm(central_fdm(9, 3), poly, 1.0, adapt=4) ≈ 24 atol=1e-11
    end


    @testset "Printing FiniteDifferenceMethods" begin
        @test sprint(show, central_fdm(2, 1)) == """
            FiniteDifferenceMethod:
              order of method:       2
              order of derivative:   1
              grid:                  [-1, 1]
              coefficients:          [-0.5, 0.5]
            """
        m, _ = fdm(central_fdm(2, 1), sin, 1, Val(true))
        report = sprint(show, m)
        regex_float = r"[\d\.\+-e]+"
        regex_array = r"\[([\d.+-e]+(, )?)+\]"
        @test occursin(Regex(join(map(x -> x.pattern,
            [
                r"FiniteDifferenceMethod:",
                r"order of method:", r"\d+",
                r"order of derivative:", r"\d+",
                r"grid:", regex_array,
                r"coefficients:", regex_array,
                r"roundoff error:", regex_float,
                r"bounds on derivatives:", regex_float,
                r"step size:", regex_float,
                r"accuracy:", regex_float,
                r""
            ]
        ), r"\s*".pattern)), report)
    end

    @testset "Breaking deprecations" begin
        @test_throws ErrorException fdm([1,2,3], 4)  # Custom grids need Nonstandard
        for f in (forward_fdm, backward_fdm, central_fdm)
            @test_throws ErrorException f(2, 1; M=1)  # Old kwarg, now misplaced
            @test_throws ErrorException f(2, 1, Val(true))  # Ask fdm for reports instead
        end
    end

    @testset "Types" begin
        @testset "$T" for T in (Forward, Backward, Central)
            @test T(5, 1)(sin, 1; adapt=4) ≈ cos(1)
            @test_throws ArgumentError T(3, 3)
            @test_throws ArgumentError T(3, 4)
            @test_throws ArgumentError T(40, 5)
            @test_throws ArgumentError T(5, 1)(sin, 1; adapt=200)
            @test_throws ArgumentError T(5, 1)(sin, 1; eps=0.0)
            @test_throws ArgumentError T(5, 1)(sin, 1; bound=0.0)
        end
        @testset "Nonstandard" begin
            @test Nonstandard([-2, -1, 1], 1)(sin, 1) ≈ cos(1)
            @test_throws ArgumentError Nonstandard([-2, -1, 1], 1)(sin, 1; adapt=2)
        end
    end
end
