using FiniteDifferences, Test, Random, Printf, LinearAlgebra, StaticArrays

@testset "FiniteDifferences" begin
    include("methods.jl")
    include("numerics.jl")
    include("to_vec.jl")
    include("grad.jl")
end
