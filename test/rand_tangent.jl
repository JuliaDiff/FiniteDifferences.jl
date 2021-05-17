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

        # Tuples.
        ((4.0, ), Composite{Tuple{Float64}}),
        ((5.0, randn(3)), Composite{Tuple{Float64, Vector{Float64}}}),

        # NamedTuples.
        ((a=4.0, ), Composite{NamedTuple{(:a,), Tuple{Float64}}}),
        ((a=5.0, b=1), Composite{NamedTuple{(:a, :b), Tuple{Float64, Int}}}),

        # structs.
        (Foo(5.0, 4, rand(rng, 3)), Composite{Foo}),
        (Foo(4.0, 3, Foo(5.0, 2, 4)), Composite{Foo}),
        (sin, typeof(NO_FIELDS)),
        # all fields NoTangent implies NoTangent
        (Pair(:a, "b"), NoTangent),
        (1:10, NoTangent),
        (1:2:10, NoTangent),

        # LinearAlgebra types (also just structs).
        (
            UpperTriangular(randn(3, 3)),
            Composite{UpperTriangular{Float64, Matrix{Float64}}},
        ),
        (
            Diagonal(randn(2)),
            Composite{Diagonal{Float64, Vector{Float64}}},
        ),
        (
            SVector{2, Float64}(1.0, 2.0),
            Composite{typeof(SVector{2, Float64}(1.0, 2.0))},
        ),
        (
            SMatrix{2, 2, ComplexF64}(1.0, 2.0, 3.0, 4.0),
            Composite{typeof(SMatrix{2, 2, ComplexF64}(1.0, 2.0, 3.0, 4.0))},
        ),
        (
            Symmetric(randn(2, 2)),
            Composite{Symmetric{Float64, Matrix{Float64}}},
        ),
        (
            Hermitian(randn(ComplexF64, 1, 1)),
            Composite{Hermitian{ComplexF64, Matrix{ComplexF64}}},
        ),
        (
            Adjoint(randn(ComplexF64, 3, 3)),
            Composite{Adjoint{ComplexF64, Matrix{ComplexF64}}},
        ),
        (
            Transpose(randn(3)),
            Composite{Transpose{Float64, Vector{Float64}}},
        ),
    ]
        @test rand_tangent(rng, x) isa T_tangent
        @test rand_tangent(x) isa T_tangent
        @test x + rand_tangent(rng, x) isa typeof(x)
    end

    # Ensure struct fallback errors for non-struct types.
    @test_throws ArgumentError invoke(rand_tangent, Tuple{AbstractRNG, Any}, rng, 5.0)
end
