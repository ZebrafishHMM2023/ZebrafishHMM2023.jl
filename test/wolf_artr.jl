using Test: @test, @testset
using MAT: matread
using Statistics: mean
using ZebrafishHMM2023: artr_wolf_2023, artr_wolf_2023_mat, artr_wolf_2023_temperatures, artr_wolf_2023_fishes

@testset "artr_wolf_2023" begin
    for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(; temperature)
        @test only(keys(matread(artr_wolf_2023_mat(; temperature=18, fish=12)))) == "Dinference_corr"
        if temperature == 26
            @test artr_wolf_2023(; temperature, fish)["T"] ∈ (26, 27)
        else
            @test artr_wolf_2023(; temperature, fish)["T"] == temperature
        end

        ts = vec(artr_wolf_2023(; temperature, fish)["time"])
        Δs = diff(ts)

        # measurements are equidistant in time
        for t = Δs
            @test t ≈ mean(Δs) rtol=1e-4
        end

        for side = ("rightspikesbin_data", "leftspikesbin_data")
            S = artr_wolf_2023(; temperature, fish)[side]
            @test size(S, 1) == length(ts)
        end
    end
end
