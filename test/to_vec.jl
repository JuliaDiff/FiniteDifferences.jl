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

# A dummy FillVector
struct FillVector <: AbstractVector{Float64}
    x::Float64
    len::Int
end

Base.size(x::FillVector) = (x.len,)
Base.getindex(x::FillVector, n::Int) = x.x

# For testing Tangent{ThreeFields}
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

# For testing generic subtypes of AbstractArray
struct WrapperArray{T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end
function WrapperArray(a::AbstractArray{T, N}) where {T, N}
    return WrapperArray{T, N, AbstractArray{T, N}}(a)
end
Base.size(a::WrapperArray) = size(a.data)
Base.getindex(a::WrapperArray, inds...) = getindex(a.data, inds...)

# can not construct it from: cca = CustomConstructorArray(rand(2, 2))
# T = typeof(cca) # CustomConstructorArray{Float64, 2, Matrix{Float64}}
# T(rand(2, 3)) # errors
struct CustomConstructorArray{T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
    function CustomConstructorArray(data::A) where {T, N, A<:AbstractArray{T, N}}
        return new{T, N, A}(data)
    end
end
Base.size(a::CustomConstructorArray) = size(a.data)
Base.getindex(a::CustomConstructorArray, inds...) = getindex(a.data, inds...)

function test_to_vec(x::T; check_inferred=true) where {T}
    check_inferred && @inferred to_vec(x)
    x_vec, back = to_vec(x)
    @test x_vec isa Vector
    @test all(s -> s isa Real, x_vec)
    @test all(!isnan, x_vec)
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
        test_to_vec([1.0, randn(T, 2), randn(T, 1), 2.0]; check_inferred=false)
        test_to_vec([randn(T, 5, 4, 3), (5, 4, 3), 2.0]; check_inferred=false)
        test_to_vec(reshape([1.0, randn(T, 5, 4, 3), randn(T, 4, 3), 2.0], 2, 2); check_inferred=false)
        test_to_vec(UpperTriangular(randn(T, 13, 13)))
        test_to_vec(Diagonal(randn(T, 7)))
        test_to_vec(DummyType(randn(T, 2, 9)))
        test_to_vec(SVector{2, T}(1.0, 2.0); check_inferred=false)
        test_to_vec(SMatrix{2, 2, T}(1.0, 2.0, 3.0, 4.0); check_inferred=false)
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

        @testset "Factorizations" begin
            # (100, 100) is needed to test for the NaNs that can appear in the
            # qr(M).T matrix
            for dims in [(7, 3), (100, 100)]
                M = randn(T, dims...)
                P = M * M' + I  # Positive definite matrix
                test_to_vec(svd(M))
                test_to_vec(cholesky(P))

                # Special treatment for QR since it is represented by a matrix
                # with some arbirtrary values.
                F = qr(M)
                @inferred to_vec(F)
                F_vec, back = to_vec(F)
                @test F_vec isa Vector
                @test all(s -> s isa Real, F_vec)
                @test all(!isnan, F_vec)
                @inferred back(F_vec)
                F_back = back(F_vec)
                @test F_back.Q == F.Q
                @test F_back.R == F.R

                # Make sure the result is consistent despite the arbitrary
                # values in F.T.
                @test first(to_vec(F)) == first(to_vec(F))

                # Test F.Q as well since it has a special type. Since it is
                # represented by the same T and factors matrices than F
                # it needs the same special treatment.
                Q = F.Q
                @inferred to_vec(Q)
                Q_vec, back = to_vec(Q)
                @test Q_vec isa Vector
                @test all(s -> s isa Real, Q_vec)
                @test all(!isnan, Q_vec)
                @inferred back(Q_vec)
                Q_back = back(Q_vec)
                @test Q_back == Q
            end
        end

        @testset "Tuples" begin
            test_to_vec((5, 4))
            test_to_vec((5, randn(T, 5)); check_inferred = VERSION ≥ v"1.2") # broken on Julia 1.6.0, fixed on 1.6.1 
            test_to_vec((randn(T, 4), randn(T, 4, 3, 2), 1); check_inferred=false)
            test_to_vec((5, randn(T, 4, 3, 2), UpperTriangular(randn(T, 4, 4)), 2.5); check_inferred = VERSION ≥ v"1.2") # broken on Julia 1.6.0, fixed on 1.6.1 
            test_to_vec(((6, 5), 3, randn(T, 3, 2, 0, 1)); check_inferred=false)
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
                test_to_vec(Dict(:a=>5, :b=>randn(10, 11), :c=>(5, 4, 3)); check_inferred=false)
            else
                test_to_vec(Dict(:a=>3 + 2im, :b=>randn(T, 10, 11), :c=>(5+im, 2-im, 1+im)); check_inferred=false)
            end
        end
    end

    @testset "ChainRulesCore Differentials" begin
        @testset "Tangent{Tuple}" begin
            @testset "basic" begin
                x_tup = (1.0, 2.0, 3.0)
                x_comp = Tangent{typeof(x_tup)}(x_tup...)
                test_to_vec(x_comp)
            end

            @testset "nested" begin
                x_inner = (2, 3)
                x_outer = (1, x_inner)
                x_comp = Tangent{typeof(x_outer)}(1, Tangent{typeof(x_inner)}(2, 3))
                test_to_vec(x_comp; check_inferred=false)
            end
        end

        @testset "Tangent Struct" begin
            @testset "NamedTuple basic" begin
                nt = (; a=1.0, b=20.0)
                comp = Tangent{typeof(nt)}(; nt...)
                test_to_vec(comp)
            end

            @testset "Struct" begin
                test_to_vec(Tangent{ThreeFields}(; a=10.0, b=20.0, c=30.0))
                test_to_vec(Tangent{ThreeFields}(; a=10.0, b=20.0,)) # broken on Julia 1.6.0, fixed on 1.6.1 
                test_to_vec(Tangent{ThreeFields}(; a=10.0, c=30.0))
                test_to_vec(Tangent{ThreeFields}(; c=30.0, a=10.0, b=20.0))
            end
        end

        @testset "AbstractZero" begin
            test_to_vec(ZeroTangent())
            test_to_vec(NoTangent())
        end

        @testset "Thunks" begin
            test_to_vec(@thunk(3.2+4.3))
        end
    end

    @testset "FillVector" begin
        test_to_vec(FillVector(5.0, 10); check_inferred=false)
    end

    @testset "fallback" begin
        nested = Nested(ThreeFields(1.0, 2.0, "Three"), Singleton())
        test_to_vec(nested; check_inferred=false) # map
    end

    @testset "WrapperArray" begin
        wa = WrapperArray(rand(4, 5))
        test_to_vec(wa; check_inferred=false)
    end

    @testset "CustomConstructorArray" begin
        cca = CustomConstructorArray(rand(2, 3))
        test_to_vec(cca; check_inferred=false)
    end
end
