using FDM, Test, Random, Printf, LinearAlgebra

@testset "FDM" begin
    include("methods.jl")
    include("numerics.jl")
    include("grad.jl")
end
