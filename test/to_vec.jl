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
    @test all(s -> s isa Real, x_vec)
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
        test_to_vec(Diagonal(randn(T, 7)))
        test_to_vec(DummyType(randn(T, 2, 9)))
        test_to_vec(SVector{2, T}(1.0, 2.0))
        test_to_vec(SMatrix{2, 2, T}(1.0, 2.0, 3.0, 4.0))

        @testset "$Op" for Op in (Symmetric, Hermitian)
            test_to_vec(Op(randn(T, 11, 11)))
            @testset "$uplo" for uplo in (:L, :U)
                A = Op(randn(T, 11, 11), uplo)
                test_to_vec(A)
                x_vec, back = to_vec(A)
                @test back(x_vec).uplo == A.uplo
            end
        end

        @testset "$Op" for Op in (Adjoint, Transpose)
            test_to_vec(Op(randn(T, 4, 4)))
            test_to_vec(Op(randn(T, 6)))
            test_to_vec(Op(randn(T, 2, 5)))
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

        @testset "Generator(identity, x)" begin
            for dims in ((3,), (3, 4), (3, 4, 5))
                xarray = randn(T, dims...)
                x = (xi for xi in xarray)
                x_vec, back = to_vec(x)
                @test x_vec isa Vector
                @test all(s -> s isa Real, x_vec)
                @test back(x_vec) isa Base.Generator{typeof(xarray),typeof(identity)}
                @test collect(back(x_vec)) == xarray
            end
        end
    end

    @testset "FillVector" begin
        x = FillVector(5.0, 10)
        x_vec, from_vec = to_vec(x)
        @test_throws MethodError from_vec(randn(10))
    end
end
