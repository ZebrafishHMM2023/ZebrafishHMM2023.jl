using ZebrafishHMM2023: load_behaviour_free_swimming_data, normalize_all!, load_behaviour_free_swimming_trajs,
    ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4, markov_equilibrium, stubborness_factor,
    ZebrafishHMM_Elife2020
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, forward, viterbi,
    transition_matrix, initial_distribution
using Statistics: mean, std
using Distributions: Normal, Gamma, fit_mle
using LinearAlgebra

hmm = ZebrafishHMM_Elife2020(; pinit_turn=0.4, pturn=0.2, pflip=0.4, σturn=2, σforw=1)

#= Load trajectories. =#
trajs = load_behaviour_free_swimming_trajs(18)

(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)


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

hmm = ZebrafishHMM_TN04(
    rand(4),
    rand(4,4),
    Normal(0, 3),
    Normal(0, 50)
)
normalize_all!(hmm)
(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)
lL[end]


hmm = ZebrafishHMM_TN4(
    rand(4),
    rand(4,4),
    Normal(0, 3),
    Normal(0, 50),
    1e-2
)
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
