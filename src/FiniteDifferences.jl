module FiniteDifferences

    using Printf, LinearAlgebra

    const AV = AbstractVector

    include("methods.jl")
    include("numerics.jl")
    include("grad.jl")


    @deprecate jacobian(fdm, f, x::Vector, D::Int) jacobian(fdm, f, x; len=D)
end
