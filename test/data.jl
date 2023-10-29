using ZebrafishHMM2023: load_behaviour_free_swimming_data, build_trajectories
using Test: @testset, @test

@testset "build_trajectories" begin
    data26 = load_behaviour_free_swimming_data(26)
    all_trajs = build_trajectories(data26.dtheta)
    for t in all_trajs, x in t
        @test !isnan(x)
        @test isfinite(x)
    end
end
