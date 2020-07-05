# Test struct for `rand_tangent`.
struct Foo
   a::Float64
   b::Int
   c::Any
end

@testset "generate_tangent" begin
    rng = MersenneTwister(123456)

    foreach([
        ("hi", DoesNotExist),
        ('a', DoesNotExist),
        (:a, DoesNotExist),
        (true, DoesNotExist),
        (4, DoesNotExist),
        (5.0, Float64),
        (5.0 + 0.4im, Complex{Float64}),
        (randn(Float32, 3), Vector{Float32}),
        (randn(Complex{Float64}, 2), Vector{Complex{Float64}}),
        (randn(5, 4), Matrix{Float64}),
        (randn(Complex{Float32}, 5, 4), Matrix{Complex{Float32}}),
        ([randn(5, 4), 4.0], Vector{Any}),
        ((4.0, ), Composite{Tuple{Float64}}),
        ((5.0, randn(3)), Composite{Tuple{Float64, Vector{Float64}}}),
        ((a=4.0, ), Composite{NamedTuple{(:a,), Tuple{Float64}}}),
        ((a=5.0, b=1), Composite{NamedTuple{(:a, :b), Tuple{Float64, Int}}}),
        (sin, typeof(NO_FIELDS)),
        (Foo(5.0, 4, rand(rng, 3)), Composite{Foo}),
        (Foo(4.0, 3, Foo(5.0, 2, 4)), Composite{Foo}),
    ]) do (x, T_tangent)
        @test rand_tangent(rng, x) isa T_tangent
        @test rand_tangent(x) isa T_tangent
        @test x + rand_tangent(rng, x) isa typeof(x)
    end

    # Ensure struct fallback errors for non-struct types.
    @test_throws ArgumentError invoke(rand_tangent, Tuple{AbstractRNG, Any}, rng, 5.0)
end
