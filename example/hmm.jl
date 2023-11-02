using ZebrafishHMM2023: load_behaviour_free_swimming_data, normalize_all!, load_behaviour_free_swimming_trajs,
    ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, viterbi
using Statistics: mean, std
using Distributions: Normal, Gamma, fit_mle
using LinearAlgebra

#= Load trajectories. =#
trajs26 = load_behaviour_free_swimming_trajs(26)

#= Zero values give trouble with Gamma.
There are only 3 trajectories with a zero, so I'll just filter those out. =#
#trajs26 = filter(traj -> all(!iszero, traj), trajs26)

# use these parameter fits to init the emission distributions
fit_mle(Normal, reduce(vcat, trajs26); mu = 0.0)
fit_mle(Gamma, +filter(>(0), reduce(vcat, trajs26)))
fit_mle(Gamma, -filter(<(0), reduce(vcat, trajs26)))

# use these parameter fits to init the emission distributions
fit_mle(Normal, reduce(vcat, trajs26); mu = 0.0)
fit_mle(Gamma, +filter(>(0), reduce(vcat, trajs26)))
fit_mle(Gamma, -filter(<(0), reduce(vcat, trajs26)))

mean(>(0), reduce(vcat, trajs26))
mean(<(0), reduce(vcat, trajs26))
mean(iszero, reduce(vcat, trajs26))

hmm = ZebrafishHMM_G3(
    rand(3),
    rand(3,3),
    Normal(0, 10),
    Gamma(0.6, 32)
)
normalize_all!(hmm)

(hmm, lL) = baum_welch(
    hmm, trajs26, length(trajs26);
    max_iterations = 500
)
lL

hmm.forw
hmm.turn

mean(hmm.forw), std(hmm.forw)
mean(hmm.turn), std(hmm.turn)

hmm.transition_matrix
hmm.initial_probs

viterbi(hmm, trajs26, length(trajs26))
