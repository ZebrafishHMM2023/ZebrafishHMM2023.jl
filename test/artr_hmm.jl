using Statistics: mean
using Test: @test, @testset
using MAT: matread
using HiddenMarkovModels: logdensityof
using ZebrafishHMM2023: artr_wolf_2023, load_artr_wolf_2023, artr_wolf_2023_mat, HMM_ARTR,
    artr_wolf_2023_temperatures, artr_wolf_2023_fishes, easy_train_artr_hmm

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

@testset "easy_train_artr_hmm" begin
    data = load_artr_wolf_2023(; temperature=18, fish=12)
    trajs = collect(eachcol(vcat(data.left, data.right)))

    hmm, lL = easy_train_artr_hmm(; temperature=18, fish=12, verbose=false, atol=1e-5)
    @test logdensityof(hmm, trajs) ≈ lL[end]
end
