function test_to_vec(x)
    x_vec, back = to_vec(x)
    @test x_vec isa Vector
    @test x == back(x_vec)
    return nothing
end

@testset "to_vec(::$T)" for T in (Float64, ComplexF64)
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
end
