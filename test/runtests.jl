using FDM, Test, Random, Printf

@testset "FDM" begin
    include("methods.jl")
    include("numerics.jl")
    include("grad.jl")
end
