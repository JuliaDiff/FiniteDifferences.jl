using FiniteDifferences, Test, Random, Printf, LinearAlgebra

@testset "FiniteDifferences" begin
    include("methods.jl")
    include("numerics.jl")
    include("grad.jl")
end
