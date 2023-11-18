using Test: @test, @testset
using ZebrafishHMM2023: ZebrafishHMM_Elife2020, stubborness_factor

@testset "Elife2020 stubborness" begin
    hmm = ZebrafishHMM_Elife2020(; pinit_turn=0.4, pturn=0.27, pflip=0.35, σturn=2, σforw=1)
    for q = 1:10
        @test stubborness_factor(hmm, q) ≈ (1 + (1 - 2hmm.pflip)^(q+2)) / (1 - (1 - 2hmm.pflip)^(q+2))
    end
end
