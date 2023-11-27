using Test: @testset, @test
using ZebrafishHMM2023: load_behaviour_free_swimming_data,
    behaviour_free_swimming_temperatures,
    legoc2021_single_fish_T26_trajs

@testset "NaN are used for padding ($(T)°C)" for T in behaviour_free_swimming_temperatures()
    data = load_behaviour_free_swimming_data(T)
    for col in eachcol(data.dtheta)
        i = findfirst(isnan, col)
        if !isnothing(i)
            @test all(isfinite, col[1:i-1])
            @test all(isnan, col[i:end])
        end
    end
end

@testset "Single fish trajectories" begin
    sf = legoc2021_single_fish_T26_trajs()
    @test length(sf) == 18
    @test sf[1][1] isa AbstractVector{<:Real}
end
