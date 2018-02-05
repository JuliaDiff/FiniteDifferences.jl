__precompile__()

module FDM

    using Compat
    import Compat.String
    @compat using Printf

    include("methods.jl")
    include("numerics.jl")
end
