using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Test

# Test struct for `rand_tangent` and `difference`.
struct Foo
   a::Float64
   b::Int
   c::Any
end

@testset "FiniteDifferences" begin
    # include("rand_tangent.jl")
    # include("difference.jl")
    include("methods.jl")
    # include("numerics.jl")
    # include("to_vec.jl")
    # include("grad.jl")
end
