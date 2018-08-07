@testset "Methods" begin
    for f in [:forward_fdm, :backward_fdm, :central_fdm]
        @eval @test $f(10, 1; M=1)(sin, 1) ≈ cos(1)
        @eval @test $f(10, 2; M=1)(sin, 1) ≈ -sin(1)

        @eval @test $f(10, 1; M=1)(exp, 1) ≈ exp(1)
        @eval @test $f(10, 2; M=1)(exp, 1) ≈ exp(1)

        @eval @test $f(10, 1; M=1)(abs2, 1) ≈ 2
        @eval @test $f(10, 2; M=1)(abs2, 1) ≈ 2

        @eval @test $f(10, 1; M=1)(sqrt, 1) ≈ .5
        @eval @test $f(10, 2; M=1)(sqrt, 1) ≈ -.25
    end

    @test_throws ArgumentError central_fdm(100, 1)

    # Test that printing an instance of `FDMReport` contains the information that it should
    # contain.
    buffer = IOBuffer()
    show(buffer, central_fdm(2, 1; report=true)[2])
    report = @compat String(take!(copy(buffer)))
    regex_float = r"[\d\.\+-e]+"
    regex_array = r"\[([\d.+-e]+(, )?)+\]"
    @test @compat occursin(Regex(join(map(x -> x.pattern,
        [
            r"FDMReport:",
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
