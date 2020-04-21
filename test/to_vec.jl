# Dummy type where length(x::DummyType) ≠ length(first(to_vec(x)))
struct DummyType{TX<:Matrix}
    X::TX
end

function FiniteDifferences.to_vec(x::DummyType)
    x_vec, back = to_vec(x.X)
    return x_vec, x_vec -> DummyType(back(x_vec))
end

Base.:(==)(x::DummyType, y::DummyType) = x.X == y.X
Base.length(x::DummyType) = size(x.X, 1)

# A dummy FillVector. This is a type for which the fallback implementation of
# `to_vec` should fail loudly.
struct FillVector <: AbstractVector{Float64}
    x::Float64
    len::Int
end

Base.size(x::FillVector) = (x.len,)
Base.getindex(x::FillVector, n::Int) = x.x

function test_to_vec(x::T) where {T}
    x_vec, back = to_vec(x)
    @test x_vec isa Vector
    @test x == back(x_vec)
    return nothing
end

@testset "to_vec" begin
    @testset "$T" for T in (Float32, ComplexF32, Float64, ComplexF64)
        if T == Float64
            test_to_vec(1.0)
            test_to_vec(1)
        else
            test_to_vec(.7 + .8im)
            test_to_vec(1 + 2im)
        end
        test_to_vec(randn(T, 3))
        test_to_vec(randn(T, 5, 11))
        test_to_vec(randn(T, 13, 17, 19))
        test_to_vec(randn(T, 13, 0, 19))
        test_to_vec([1.0, randn(T, 2), randn(T, 1), 2.0])
        test_to_vec([randn(T, 5, 4, 3), (5, 4, 3), 2.0])
        test_to_vec(reshape([1.0, randn(T, 5, 4, 3), randn(T, 4, 3), 2.0], 2, 2))
        test_to_vec(UpperTriangular(randn(T, 13, 13)))
        test_to_vec(Symmetric(randn(T, 11, 11)))
        test_to_vec(Diagonal(randn(T, 7)))
        test_to_vec(DummyType(randn(T, 2, 9)))
        test_to_vec(SVector{2, T}(1.0, 2.0))
        test_to_vec(SMatrix{2, 2, T}(1.0, 2.0, 3.0, 4.0))

        @testset "$Op" for Op in (Adjoint, Transpose)
            test_to_vec(Op(randn(T, 4, 4)))
            test_to_vec(Op(randn(T, 6)))
            test_to_vec(Op(randn(T, 2, 5)))

            A = randn(T, 3, 3)
            @test reshape(first(to_vec(Op(A))), 3, 3) == Op(A)
        end

        @testset "Tuples" begin
            test_to_vec((5, 4))
            test_to_vec((5, randn(T, 5)))
            test_to_vec((randn(T, 4), randn(T, 4, 3, 2), 1))
            test_to_vec((5, randn(T, 4, 3, 2), UpperTriangular(randn(T, 4, 4)), 2.5))
            test_to_vec(((6, 5), 3, randn(T, 3, 2, 0, 1)))
            test_to_vec((DummyType(randn(T, 2, 7)), DummyType(randn(T, 3, 9))))
            test_to_vec((DummyType(randn(T, 3, 2)), randn(T, 11, 8)))
        end
        @testset "Dictionary" begin
            if T == Float64
                test_to_vec(Dict(:a=>5, :b=>randn(10, 11), :c=>(5, 4, 3)))
            else
                test_to_vec(Dict(:a=>3 + 2im, :b=>randn(T, 10, 11), :c=>(5+im, 2-im, 1+im)))
            end
        end
    end

    @testset "FillVector" begin
        x = FillVector(5.0, 10)
        x_vec, from_vec = to_vec(x)
        @test_throws MethodError from_vec(randn(10))
    end

    # Actually test that the correct thing happens via to_vec.
    @testset "Complex correctness - $T" for T in [ComplexF32, ComplexF64]
        rng = MersenneTwister(123456)
        x = randn(rng, T)
        y = randn(rng, T)
        fdm = FiniteDifferences.Central(5, 1)

        # Addition.
        dx, dy = FiniteDifferences.jacobian(fdm, +, x, y)
        @test dz ≈ [1 0; 0 1]
        @test dy ≈ [1 0; 0 1]

        # Negation.
        dx, = FiniteDifferences.jacobian(fdm, -, x)
        @test dx ≈ [-1 0; 0 -1]

        # Multiplication.
        dx, dy = FiniteDifferences.jacobian(fdm, *, x, y)
        @test dx ≈ [real(y) -imag(y); imag(y) real(y)]
        @test dy ≈ [real(x) -imag(x); imag(x) real(x)]
    end
end
