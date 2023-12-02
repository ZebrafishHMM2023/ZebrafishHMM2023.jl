using ZebrafishHMM2023: ATol
using Test: @test, @testset

@testset "ATol" begin
    @test -0.1 < ATol(1)
    @test +0.1 < ATol(1)

    @test !(+5 < ATol(1))
    @test !(-5 < ATol(1))
end
