module FiniteDifferences

    using Printf, LinearAlgebra

    include("methods.jl")
    include("numerics.jl")
    include("to_vec.jl")
    include("grad.jl")
end
