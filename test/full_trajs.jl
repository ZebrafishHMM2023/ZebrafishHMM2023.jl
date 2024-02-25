using Test: @test, @testset
using ZebrafishHMM2023: legoc2021_single_fish_T26_full_obs

@testset "legoc2021_single_fish_T26_full_obs" begin
    long_trajs = legoc2021_single_fish_T26_full_obs()
    @test length(long_trajs) == 18 # number of fish
end
