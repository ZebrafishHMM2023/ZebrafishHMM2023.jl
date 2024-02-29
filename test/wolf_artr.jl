using Test: @test, @testset
using MAT: matread
using Statistics: mean
using ZebrafishHMM2023: artr_wolf_2023, artr_wolf_2023_mat, artr_wolf_2023_temperatures,
    artr_wolf_2023_fishes, load_artr_wolf_2023

@testset "artr_wolf_2023 temperatures and fishes" begin
    for fish = artr_wolf_2023_fishes(), T = artr_wolf_2023_temperatures(fish)
        @test fish ∈ artr_wolf_2023_fishes(T)
    end
    for T = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(T)
        @test T ∈ artr_wolf_2023_temperatures(fish)
    end
end

@testset "artr_wolf_2023" begin
    for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(temperature)
        @test only(keys(matread(artr_wolf_2023_mat(; temperature=18, fish=12)))) == "Dinference_corr"
        data = artr_wolf_2023(; temperature, fish)
        if temperature == 26
            @test data["T"] ∈ (26, 27)
        else
            @test data["T"] == temperature
        end

        ts = vec(data["time"])
        Δs = diff(ts)

        # measurements are equidistant in time
        for t = Δs
            @test t ≈ mean(Δs) rtol=1e-4
        end

        for side = ("rightspikesbin_data", "leftspikesbin_data")
            S = data[side]
            @test size(S, 1) == length(ts)
        end

        @test vec(data["L_reg"]) == 1:length(data["L_reg"])

        data2 = load_artr_wolf_2023(; temperature, fish)
        @test data2.left == data["leftspikesbin_data"]'
        @test data2.right == data["rightspikesbin_data"]'
        @test data2.time == ts
        @test data2.temperature == data["T"]
    end
end
