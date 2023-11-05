using ZebrafishHMM2023: normalize_all!, load_behaviour_free_swimming_trajs, ZebrafishHMM_TN03, markov_equilibrium
using HiddenMarkovModels: baum_welch
using Distributions: Normal
using Test: @test, @testset

@testset "markov_equilibrium" begin
    trajs = load_behaviour_free_swimming_trajs(22)

    hmm = ZebrafishHMM_TN03(
        rand(3),
        rand(3,3),
        Normal(0, 3),
        Normal(0, 50)
    )
    normalize_all!(hmm)

    (hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)

    p_eq = markov_equilibrium(hmm.transition_matrix)
    @test sum(p_eq) ≈ 1
    @test p_eq ≈ hmm.transition_matrix' * p_eq
end
