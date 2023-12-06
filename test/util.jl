using Test: @test, @testset, @inferred
using ZebrafishHMM2023: chunks

@testset "chunks" begin
    @test chunks(1:54, 5) == [1:11, 12:22, 23:33, 34:44, 45:54]
    @test chunks(2:55, 5) == [2:12, 13:23, 24:34, 35:45, 46:55]
end
