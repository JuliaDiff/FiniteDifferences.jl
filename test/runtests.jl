using FDM
using Base.Test

@testset "FDM" begin
    include("methods.jl")
    include("numerics.jl")
end
