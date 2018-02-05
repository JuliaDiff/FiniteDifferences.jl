__precompile__()

module FDM

    using Compat
    using Compat.Printf

    include("methods.jl")
    include("numerics.jl")
end
