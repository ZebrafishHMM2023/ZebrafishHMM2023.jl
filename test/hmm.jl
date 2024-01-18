using ZebrafishHMM2023: normalize_all!, load_behaviour_free_swimming_trajs, ZebrafishHMM_TN03,
    markov_equilibrium, stubborness_factor, ZebrafishHMM_TN04, FL_FR_canon!,
    ZebrafishHMM_G4_Sym
using HiddenMarkovModels: baum_welch, logdensityof
using Distributions: Normal, Gamma
using Statistics: middle
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

trajs = load_behaviour_free_swimming_trajs(22)

@testset "stubborness_factor, 3-state non-symmetric" begin
    hmm = ZebrafishHMM_TN03(
        rand(3),
        rand(3,3),
        Normal(0, 3),
        Normal(0, 50)
    )
    normalize_all!(hmm)

    # no dependence on q for 3-states
    @test stubborness_factor(hmm, 1) == stubborness_factor(hmm, 2)
end

@testset "stubborness_factor, 3-state symmetric" begin
    # symmetrize left / right
    p_forw = 0.6 # forw -> forw
    p_turn = 0.2 # forw -> turn
    p_retu = 0.3 # turn -> forw
    p_circ = 0.35 # turn -> turn (same direction)
    p_reve = 0.35 # turn -> turn (opposite direction)
    T = [
        p_forw p_turn p_turn;
        p_retu p_circ p_reve;
        p_retu p_reve p_circ;
    ]
    @test sum(T; dims=2) ≈ ones(3)
    hmm = ZebrafishHMM_TN03(rand(3), T, Normal(0, 3), Normal(0, 50))
    normalize_all!(hmm)
    peq = markov_equilibrium(hmm.transition_matrix)
    @test peq[2] ≈ peq[3]
    # stubborness_factor is 1 for symmetric 3-state HMM
    @test stubborness_factor(hmm, 1) == stubborness_factor(hmm, 2) ≈ 1
end

@testset "stubborness_factor, 4-state symmetric" begin
    # symmetrize left / right
    p_forw_1 = 0.4 # FL -> FL, FR -> FR
    p_forw_2 = 0.2 # FL -> FR, FR -> FL
    p_turn_1 = 0.15 # FL -> L, FR -> R
    p_turn_2 = 0.25 # FL -> R, FR -> L
    p_straig = 0.3 # L -> FL, R -> FR
    p_tanh = 0.32 # L -> L, R -> R
    p_rev = 0.38 # L -> R, R -> L
    T = [
        p_forw_1 p_forw_2 p_turn_1 p_turn_2;
        p_forw_2 p_forw_1 p_turn_2 p_turn_1;
        p_straig 0 p_tanh p_rev;
        0 p_straig p_rev p_tanh
    ]
    @test sum(T; dims=2) ≈ ones(4)
    hmm = ZebrafishHMM_TN04(rand(4), T, Normal(0, 3), Normal(0, 50))
    normalize_all!(hmm)
    peq = markov_equilibrium(hmm.transition_matrix)
    @test peq[1] ≈ peq[2]
    @test peq[3] ≈ peq[4]

    # test exact formula for the 4-state symmetric case
    U = (p_forw_1 - p_forw_2) / (p_forw_1 + p_forw_2)
    W = (p_turn_1 - p_turn_2) / (p_turn_1 + p_turn_2)
    for q = 1:10
        @test stubborness_factor(hmm, q) ≈ (1 + U^(q - 1) * W) / (1 - U^(q - 1) * W)
    end
end

@testset "g4_sym" begin
    hmm = ZebrafishHMM_G4_Sym(
        rand(),
        rand(4,4),
        1.0,
        Gamma(0.5, 15.0)
    )
    normalize_all!(hmm)
    trajs = load_behaviour_free_swimming_trajs(22)
    ll0 = logdensityof(hmm, trajs, length(trajs))
    FL_FR_canon!(hmm)
    ll1 = logdensityof(hmm, trajs, length(trajs))
    @test ll1 ≈ ll0
end
