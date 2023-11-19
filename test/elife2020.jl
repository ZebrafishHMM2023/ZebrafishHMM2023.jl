using Test: @test, @testset
using HiddenMarkovModels: logdensityof, baum_welch, transition_matrix, initial_distribution
using Distributions: Normal, Gamma
using ZebrafishHMM2023: ZebrafishHMM_Elife2020, stubborness_factor,
    ZebrafishHMM_Elife2020_Gamma, ZebrafishHMM_G4_Sym,
    load_behaviour_free_swimming_trajs, normalize_all!

@testset "Elife2020 stubborness" begin
    hmm = ZebrafishHMM_Elife2020(; pinit_turn=0.4, pturn=0.27, pflip=0.35, σturn=2, σforw=1)
    for q = 1:10
        @test stubborness_factor(hmm, q) ≈ (1 + (1 - 2hmm.pflip)^(q+2)) / (1 - (1 - 2hmm.pflip)^(q+2))
    end
end

@testset "ZebrafishHMM_G4_Sym" begin
    trajs = load_behaviour_free_swimming_trajs(18)
    hmm_elife = ZebrafishHMM_Elife2020_Gamma(; pinit_turn=rand(), pturn=rand(), pflip=rand(), σforw=0.1, turn=Gamma(1, 15))
    (hmm_elife, lL) = baum_welch(hmm_elife, trajs, length(trajs); max_iterations = 50)

    hmm = ZebrafishHMM_G4_Sym(
        hmm_elife.pinit_turn,
        Matrix(transition_matrix(hmm_elife)),
        Normal(0.0, hmm_elife.σforw),
        hmm_elife.turn,
    )
    normalize_all!(hmm)
    @test logdensityof(hmm, trajs, length(trajs)) ≈ logdensityof(hmm_elife, trajs, length(trajs))
end
