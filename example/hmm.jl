using ZebrafishHMM2023: load_behaviour_free_swimming_data, normalize_all!,
    normalize_transition_matrix!,
    load_behaviour_free_swimming_trajs,
    ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4, markov_equilibrium, stubborness_factor,
    ZebrafishHMM_Elife2020, ZebrafishHMM_Elife2020_Gamma, ZebrafishHMM_G4_Sym
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, forward, viterbi,
    transition_matrix, initial_distribution
using Statistics: mean, std
using Distributions: Normal, Gamma, fit_mle
using LinearAlgebra


#= Load trajectories. =#
trajs = load_behaviour_free_swimming_trajs(18)

hmm_elife = ZebrafishHMM_Elife2020_Gamma(; pinit_turn=rand(), pturn=rand(), pflip=rand(), σforw=0.1, turn=Gamma(1, 15))
(hmm_elife, lL) = baum_welch(hmm_elife, trajs, length(trajs); max_iterations = 500)
hmm_elife
stubborness_factor(hmm_elife, 2)

#= Zero values give trouble with Gamma.
There are only 3 trajectories with a zero, so I'll just filter those out. =#
trajs = filter(traj -> all(!iszero, traj), trajs)

# use these parameter fits to init the emission distributions
fit_mle(Normal, reduce(vcat, trajs); mu = 0.0)
fit_mle(Gamma, +filter(>(0), reduce(vcat, trajs)))
fit_mle(Gamma, -filter(<(0), reduce(vcat, trajs)))
fit_mle(Normal, +filter(>(0), reduce(vcat, trajs)))
fit_mle(Normal, -filter(<(0), reduce(vcat, trajs)))

# use these parameter fits to init the emission distributions
fit_mle(Normal, reduce(vcat, trajs); mu = 0.0)
fit_mle(Gamma, +filter(>(0), reduce(vcat, trajs)))
fit_mle(Gamma, -filter(<(0), reduce(vcat, trajs)))

mean(>(0), reduce(vcat, trajs))
mean(<(0), reduce(vcat, trajs))
mean(iszero, reduce(vcat, trajs))
hmm = ZebrafishHMM_G4_Sym(
    hmm_elife.pinit_turn,
    Matrix(transition_matrix(hmm_elife)),
    Normal(0.0, hmm_elife.σforw),
    hmm_elife.turn,
)
normalize_all!(hmm)
logdensityof(hmm, trajs, length(trajs))
logdensityof(hmm_elife, trajs, length(trajs))

(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 1000)
lL[end]
logdensityof(hmm, trajs, length(trajs))

stubborness_factor(hmm, 3)


normalize_all!(hmm)
(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)
lL[end]


hmm = ZebrafishHMM_TN3(
    rand(3),
    rand(3,3),
    Normal(0, 3),
    Normal(0, 50),
    1e-2
)
normalize_all!(hmm)
(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)
lL[end]
lL[1]

stubborness_factor(hmm, 3)

markov_equilibrium(hmm.transition_matrix)

logdensityof(hmm, trajs, length(trajs))
forward(hmm, trajs, length(trajs))

hmm.forw
hmm.turn

mean(hmm.forw), std(hmm.forw)
mean(hmm.turn), std(hmm.turn)
