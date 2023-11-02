using Test: @test, @testset
using ZebrafishHMM2023: half_normal_fit, half_normal_loglikelihood
using SpecialFunctions: logerfcx
using Distributions: truncated, Normal, logpdf
using Statistics: mean

@testset "half_normal_loglikelihood" begin
    for _ = 1:10
        x = randn(100).^2 + randn(100)
        S1 = mean(abs, x)
        S2 = mean(abs2, x)
        μ = randn()
        σ = 5rand()
        ll = mean(logpdf.(truncated(Normal(μ, σ), 0, nothing), abs.(x)))
        @test half_normal_loglikelihood(μ/σ, 1/σ, S1, S2) ≈ ll
    end
end

@testset "half_normal_fit" begin
    μ, σ = half_normal_fit(sqrt(2/π), 1)
    @test abs(μ) < 1e-10
    @test σ ≈ 1

    μ, σ = half_normal_fit(3.28205277499448947798906269824, 15.5641055499889789559781253965)
    @test μ ≈ 2
    @test σ ≈ 3

    μ, σ = half_normal_fit(1.79553402202613262311852316249, 5.40893195594773475376295367502)
    @test μ ≈ -2
    @test σ ≈ 3
end
