module FiniteDifferences

    using Printf, LinearAlgebra

    const AV = AbstractVector

    include("methods.jl")
    include("numerics.jl")
    include("grad.jl")
end
