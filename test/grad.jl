using FDM: grad

@testset "grad" begin
    x = randn(MersenneTwister(123456), 2)
    xc = copy(x)
    @test grad(central_fdm(5, 1), x->sin(x[1]) + cos(x[2]), x) ≈ [cos(x[1]), -sin(x[2])]
    @test xc == x
end
