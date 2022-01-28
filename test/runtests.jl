using BenchmarkTools
using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Test


Random.seed!(1)

@testset "FiniteDifferences" begin
    include("deprecated.jl")
    include("methods.jl")
    include("numerics.jl")
    include("to_vec.jl")
    include("grad.jl")
end
