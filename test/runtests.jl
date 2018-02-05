using FDM, Compat.Test, Compat

@testset "FDM" begin
    include("methods.jl")
    include("numerics.jl")
end
