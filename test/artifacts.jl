using Test: @test
using Test: @testset
using ZebrafishHMM2023: load_behaviour_free_swimming_data

@testset "load_behaviour_free_swimming_data" begin
    data = load_behaviour_free_swimming_data(18)
    @test data.units["displacements"] == "mm"
    @test size(data.bouttime) == (642, 532)
    @test data.temperature == "18°C"
end
