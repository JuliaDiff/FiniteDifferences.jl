using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Test

@testset "FiniteDifferences" begin
    include("rand_tangent.jl")
    include("difference.jl")
    include("methods.jl")
    include("numerics.jl")
    include("to_vec.jl")
    include("grad.jl")
end
