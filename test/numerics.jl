@testset "Numerics" begin
    @test_throws ErrorException assert_approx_equal(1, 1 + 1e-5, 1e-10, 1e-6, "test")
    @test assert_approx_equal(1, 1 + 1e-7, 1e-10, 1e-6, "test")
    @test_throws ErrorException assert_approx_equal(0, 1e-9, 1e-10, 1e-6, "test")
    @test assert_approx_equal(0, 1e-11, 1e-10, 1e-6, "test")
end
