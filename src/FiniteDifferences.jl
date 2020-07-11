module FiniteDifferences

    using ChainRulesCore
    using LinearAlgebra
    using Printf
    using Random

    export to_vec, grad, jacobian, jvp, j′vp

    include("rand_tangent.jl")
    include("difference.jl")
    include("methods.jl")
    include("numerics.jl")
    include("to_vec.jl")
    include("grad.jl")
end
