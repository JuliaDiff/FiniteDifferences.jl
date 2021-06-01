using FiniteDifferences: rand_tangent

@testset "generate_tangent" begin
    rng = MersenneTwister(123456)

    @testset "Primal: $(typeof(x)), Tangent: $T_tangent" for (x, T_tangent) in [

        # Things without sensible tangents.
        ("hi", NoTangent),
        ('a', NoTangent),
        (:a, NoTangent),
        (true, NoTangent),
        (4, NoTangent),

        # Numbers.
        (5.0, Float64),
        (5.0 + 0.4im, Complex{Float64}),
        (big(5.0), BigFloat),

        # StridedArrays.
        (randn(Float32, 3), Vector{Float32}),
        (randn(Complex{Float64}, 2), Vector{Complex{Float64}}),
        (randn(5, 4), Matrix{Float64}),
        (randn(Complex{Float32}, 5, 4), Matrix{Complex{Float32}}),
        ([randn(5, 4), 4.0], Vector{Any}),

        # Wrapper Arrays
        (randn(5, 4)', Adjoint{Float64, Matrix{Float64}}),
        (transpose(randn(5, 4)), Transpose{Float64, Matrix{Float64}}),


        # Tuples.
        ((4.0, ), Tangent{Tuple{Float64}}),
        ((5.0, randn(3)), Tangent{Tuple{Float64, Vector{Float64}}}),

        # NamedTuples.
        ((a=4.0, ), Tangent{NamedTuple{(:a,), Tuple{Float64}}}),
        ((a=5.0, b=1), Tangent{NamedTuple{(:a, :b), Tuple{Float64, Int}}}),

        # structs.
        (Foo(5.0, 4, rand(rng, 3)), Tangent{Foo}),
        (Foo(4.0, 3, Foo(5.0, 2, 4)), Tangent{Foo}),
        (sin, NoTangent),
        # all fields NoTangent implies NoTangent
        (Pair(:a, "b"), NoTangent),
        (1:10, NoTangent),
        (1:2:10, NoTangent),

        # LinearAlgebra types (also just structs).
        (
            UpperTriangular(randn(3, 3)),
            Tangent{UpperTriangular{Float64, Matrix{Float64}}},
        ),
        (
            Diagonal(randn(2)),
            Tangent{Diagonal{Float64, Vector{Float64}}},
        ),
        (
            SVector{2, Float64}(1.0, 2.0),
            Tangent{typeof(SVector{2, Float64}(1.0, 2.0))},
        ),
        (
            SMatrix{2, 2, ComplexF64}(1.0, 2.0, 3.0, 4.0),
            Tangent{typeof(SMatrix{2, 2, ComplexF64}(1.0, 2.0, 3.0, 4.0))},
        ),
        (
            Symmetric(randn(2, 2)),
            Tangent{Symmetric{Float64, Matrix{Float64}}},
        ),
        (
            Hermitian(randn(ComplexF64, 1, 1)),
            Tangent{Hermitian{ComplexF64, Matrix{ComplexF64}}},
        ),
    ]
        @test rand_tangent(rng, x) isa T_tangent
        @test rand_tangent(x) isa T_tangent
    end

    @testset "erroring cases" begin
        # Ensure struct fallback errors for non-struct types.
        @test_throws ArgumentError invoke(rand_tangent, Tuple{AbstractRNG, Any}, rng, 5.0)
    end

    @testset "compsition of addition" begin
        x = Foo(1.5, 2, Foo(1.1, 3, [1.7, 1.4, 0.9]))
        @test x + rand_tangent(x) isa typeof(x)
        @test x + (rand_tangent(x) + rand_tangent(x)) isa typeof(x)
    end

    # Julia 1.6 changed to using Ryu printing algorithm and seems better at printing short
    VERSION > v"1.6" && @testset "niceness of printing" begin
        for i in 1:50
            @test length(string(rand_tangent(1.0))) <= 6
            @test length(string(rand_tangent(1.0 + 1.0im))) <= 12
            @test length(string(rand_tangent(1f0))) <= 12
            @test length(string(rand_tangent(big"1.0"))) <= 9
        end
    end
end
