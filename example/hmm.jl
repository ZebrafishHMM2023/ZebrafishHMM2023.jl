using ZebrafishHMM2023: load_behaviour_free_swimming_data, normalize_all!, load_behaviour_free_swimming_trajs,
    ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4, markov_equilibrium
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, forward, viterbi
using Statistics: mean, std
using Distributions: Normal, Gamma, fit_mle
using LinearAlgebra

#= Load trajectories. =#
trajs = load_behaviour_free_swimming_trajs(22)

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

hmm = ZebrafishHMM_TN03(
    rand(3),
    rand(3,3),
    Normal(0, 3),
    Normal(0, 50)
)
normalize_all!(hmm)

(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 100)
lL

markov_equilibrium(hmm.transition_matrix)

logdensityof(hmm, trajs, length(trajs))
forward(hmm, trajs, length(trajs))

hmm.forw
hmm.turn

mean(hmm.forw), std(hmm.forw)
mean(hmm.turn), std(hmm.turn)

hmm.transition_matrix
hmm.initial_probs

viterbi(hmm, trajs, length(trajs))

using SpecialFunctions: erfcx
using ZebrafishHMM2023: half_normal_fit, half_normal_fit_optimization, half_normal_fit_optim
half_normal_fit_iter(sqrt(2/π), 1)
erfcx(0)

half_normal_fit_optimization(3.28205277499448947798906269824, 15.5641055499889789559781253965)
half_normal_fit_optim(3.28205277499448947798906269824, 15.5641055499889789559781253965)
