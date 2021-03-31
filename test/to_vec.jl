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

# For testing Composite{ThreeFields}
struct ThreeFields
    a
    b
    c
end

# For testing nested fallback for structs
struct Singleton end
struct Nested
    x::ThreeFields
    y::Singleton
end

Base.size(x::FillVector) = (x.len,)
Base.getindex(x::FillVector, n::Int) = x.x

function test_to_vec(x::T; check_inferred = true) where {T}
    check_inferred && @inferred to_vec(x)
    x_vec, back = to_vec(x)
    @test x_vec isa Vector
    @test all(s -> s isa Real, x_vec)
    check_inferred && @inferred back(x_vec)
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
        test_to_vec(T[])
        test_to_vec(Vector{T}[])
        test_to_vec(Matrix{T}[])
        test_to_vec(randn(T, 3))
        test_to_vec(randn(T, 5, 11))
        test_to_vec(randn(T, 13, 17, 19))
        test_to_vec(randn(T, 13, 0, 19))
        test_to_vec([1.0, randn(T, 2), randn(T, 1), 2.0]; check_inferred = false)
        test_to_vec([randn(T, 5, 4, 3), (5, 4, 3), 2.0]; check_inferred = false)
        test_to_vec(reshape([1.0, randn(T, 5, 4, 3), randn(T, 4, 3), 2.0], 2, 2); check_inferred = false)
        test_to_vec(UpperTriangular(randn(T, 13, 13)))
        test_to_vec(Diagonal(randn(T, 7)))
        test_to_vec(DummyType(randn(T, 2, 9)))
        test_to_vec(SVector{2, T}(1.0, 2.0))
        test_to_vec(SMatrix{2, 2, T}(1.0, 2.0, 3.0, 4.0))
        test_to_vec(@view randn(T, 10)[1:4])  # SubArray -- Vector
        test_to_vec(@view randn(T, 10, 2)[1:4, :])  # SubArray -- Matrix
        test_to_vec(Base.ReshapedArray(rand(T, 3, 3), (9,), ()))

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

            # Ensure that if an `AbstractVector` is `Adjoint`ed, then the reconstructed
            # version also contains an `AbstractVector`, rather than an `AbstractMatrix`
            # whose 2nd dimension is of size 1.
            @testset "Vector" begin
                x_vec, back = to_vec(Op(randn(T, 5)))
                @test parent(back(x_vec)) isa AbstractVector
            end
        end

        @testset "PermutedDimsArray" begin
            test_to_vec(PermutedDimsArray(randn(T, 3, 1), (2, 1)))
            test_to_vec(PermutedDimsArray(randn(T, 4, 2, 3), (3, 1, 2)))
            test_to_vec(
                PermutedDimsArray(
                    [randn(T, 3) for _ in 1:3, _ in 1:2, _ in 1:4], (2, 1, 3),
                ),
            )
        end

        @testset "Tuples" begin
            test_to_vec((5, 4))
            test_to_vec((5, randn(T, 5)); check_inferred = VERSION ≥ v"1.2" && VERSION < v"1.6") # remove 1.6 once https://github.com/JuliaLang/julia/issues/40277
            test_to_vec((randn(T, 4), randn(T, 4, 3, 2), 1); check_inferred = false)
            test_to_vec((5, randn(T, 4, 3, 2), UpperTriangular(randn(T, 4, 4)), 2.5); check_inferred = VERSION ≥ v"1.2")
            test_to_vec(((6, 5), 3, randn(T, 3, 2, 0, 1)); check_inferred = false)
            test_to_vec((DummyType(randn(T, 2, 7)), DummyType(randn(T, 3, 9))))
            test_to_vec((DummyType(randn(T, 3, 2)), randn(T, 11, 8)))
        end
        @testset "NamedTuple" begin
            if T == Float64
                test_to_vec((a=5, b=randn(10, 11), c=(5, 4, 3)); check_inferred = VERSION ≥ v"1.2")
            else
                test_to_vec((a=3 + 2im, b=randn(T, 10, 11), c=(5+im, 2-im, 1+im)); check_inferred = VERSION ≥ v"1.2")
            end
        end
        @testset "Dictionary" begin
            if T == Float64
                test_to_vec(Dict(:a=>5, :b=>randn(10, 11), :c=>(5, 4, 3)); check_inferred = false)
            else
                test_to_vec(Dict(:a=>3 + 2im, :b=>randn(T, 10, 11), :c=>(5+im, 2-im, 1+im)); check_inferred = false)
            end
        end
    end

    @testset "ChainRulesCore Differentials" begin
        @testset "Composite{Tuple}" begin
            @testset "basic" begin
                x_tup = (1.0, 2.0, 3.0)
                x_comp = Composite{typeof(x_tup)}(x_tup...)
                test_to_vec(x_comp)
            end

            @testset "nested" begin
                x_inner = (2, 3)
                x_outer = (1, x_inner)
                x_comp = Composite{typeof(x_outer)}(1, Composite{typeof(x_inner)}(2, 3))
                test_to_vec(x_comp; check_inferred = false)
            end
        end

        @testset "Composite Struct" begin
            @testset "NamedTuple basic" begin
                nt = (; a=1.0, b=20.0)
                comp = Composite{typeof(nt)}(; nt...)
                test_to_vec(comp)
            end

            @testset "Struct" begin
                test_to_vec(Composite{ThreeFields}(; a=10.0, b=20.0, c=30.0))
                test_to_vec(Composite{ThreeFields}(; a=10.0, b=20.0,))
                test_to_vec(Composite{ThreeFields}(; a=10.0, c=30.0))
                test_to_vec(Composite{ThreeFields}(; c=30.0, a=10.0, b=20.0))
            end
        end

        @testset "AbstractZero" begin
            test_to_vec(Zero())
            test_to_vec(DoesNotExist())
        end
    end

    @testset "FillVector" begin
        x = FillVector(5.0, 10)
        x_vec, from_vec = to_vec(x)
        @test_throws MethodError from_vec(randn(10))
    end

    @testset "fallback" begin
        nested = Nested(ThreeFields(1.0, 2.0, "Three"), Singleton())
        test_to_vec(nested; check_inferred=false) # map
    end
end
