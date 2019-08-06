using FiniteDifferences: Forward, Backward, Central, Nonstandard

@testset "Methods" begin
    for f in [:forward_fdm, :backward_fdm, :central_fdm]
        @eval @test $f(1, 0; bound=1)(sin, 1) == sin(1)
        @eval @test $f(2, 0; bound=1)(sin, 1) == sin(1)
        @eval @test $f(3, 0; bound=1)(sin, 1) == sin(1)
        @eval @test $f(10, 1; bound=1)(sin, 1) ≈ cos(1)
        @eval @test $f(10, 2; bound=1)(sin, 1) ≈ -sin(1)

        @eval @test $f(1, 0; bound=1)(exp, 1) == exp(1)
        @eval @test $f(2, 0; bound=1)(exp, 1) == exp(1)
        @eval @test $f(3, 0; bound=1)(exp, 1) == exp(1)
        @eval @test $f(10, 1; bound=1)(exp, 1) ≈ exp(1)
        @eval @test $f(10, 2; bound=1)(exp, 1) ≈ exp(1)

        @eval @test $f(1, 0; bound=1)(abs2, 1) == 1
        @eval @test $f(2, 0; bound=1)(abs2, 1) == 1
        @eval @test $f(3, 0; bound=1)(abs2, 1) == 1
        @eval @test $f(10, 1; bound=1)(abs2, 1) ≈ 2
        @eval @test $f(10, 2; bound=1)(abs2, 1) ≈ 2

        @eval @test $f(1, 0; bound=1)(sqrt, 1) == 1
        @eval @test $f(2, 0; bound=1)(sqrt, 1) == 1
        @eval @test $f(3, 0; bound=1)(sqrt, 1) == 1
        @eval @test $f(10, 1; bound=1)(sqrt, 1) ≈ .5
        @eval @test $f(10, 2; bound=1)(sqrt, 1) ≈ -.25
    end

    @testset "Adaptation improves estimate" begin
        @test forward_fdm(5, 1)(log, 0.001; adapt=0) ≈ 969.2571703
        @test forward_fdm(5, 1)(log, 0.001; adapt=1) ≈ 1000
    end

    @testset "Limiting step size" begin
        @test !isfinite(central_fdm(5, 1)(abs, 0.001; max_step=0))
        @test central_fdm(5, 1)(abs, 0.001) ≈ 1.0
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
