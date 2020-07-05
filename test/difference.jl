function test_difference(ε::Real, x, dx)
    y = x + ε * dx
    dx_diff = difference(ε, y, x)
    @test typeof(dx) == typeof(dx_diff)
end

@testset "difference" begin

    @testset "$(typeof(x))" for (ε, x) in [

        # Test things that don't have tangents.
        (randn(), :a),
        (randn(), 'a'),
        (randn(), "a"),
        (randn(), 0),

        # Test Numbers.
        (randn(Float32), randn(Float32)),
        (randn(Float64), randn(Float64)),
        (randn(Float32), randn(ComplexF32)),
        (randn(Float64), randn(ComplexF64)),

        # Test StridedArrays.
        (randn(), randn(5)),
        (randn(Float32), randn(ComplexF32, 5, 2)),
        (randn(), [randn(1) for _ in 1:3]),
        (randn(), [randn(5, 4), "a"]),

        # Tuples.
        (randn(), (randn(5, 2), )),
        (randn(), (randn(), 4)),
        (randn(), (4, 3, 2)),

        # NamedTuples.
        (randn(), (a=randn(5, 2),)),
        (randn(), (a=randn(), b=4)),
        (randn(), (a=4, b=3, c=2)),

        # Arbitrary structs.
        (randn(), sin),
        (randn(), cos),
        (randn(), Foo(5.0, 4, randn(5, 2))),
        (randn(), Foo(randn(), 1, Foo(randn(), 1, 1))),
    ]
        test_difference(ε, x, rand_tangent(x))
    end
end
