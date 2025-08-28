module FiniteDifferences

using ChainRulesCore
using LinearAlgebra
using Printf
using Random
using Richardson
using SparseArrays
using StaticArrays
using Unitful

export to_vec, grad, jacobian, jvp, j′vp

include("deprecated.jl")
include("methods.jl")
include("numerics.jl")
include("to_vec.jl")
include("grad.jl")

end
