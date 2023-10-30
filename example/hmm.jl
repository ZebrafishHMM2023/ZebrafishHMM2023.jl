using ZebrafishHMM2023: load_behaviour_free_swimming_data, ZebrafishHMM, normalize_all!,
    build_trajectories, ZebrafishHMM_FLR
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, viterbi
using Statistics: mean, std
using Distributions: Normal, Gamma, fit
using LinearAlgebra

data26 = load_behaviour_free_swimming_data(26)

#= zero values give Infinities with Gamma.
There are very few 0's, so I'll just replace them by `missing` =#
all_trajs = collect(eachcol(replace(data26.dtheta, NaN => missing, 0 => missing)))

# use these parameter fits to init the emission distributions
fit(Normal, collect(skipmissing(reduce(vcat, all_trajs))))
fit(Gamma, +filter(>(0), skipmissing(reduce(vcat, all_trajs))))
fit(Gamma, -filter(<(0), skipmissing(reduce(vcat, all_trajs))))

mean(>(0), skipmissing(data26.dtheta))
mean(<(0), skipmissing(data26.dtheta))
mean(iszero, skipmissing(data26.dtheta))

hmm = ZebrafishHMM(
    rand(4),
    rand(4, 4),
    Normal(0, 10),
    Gamma(0.6, 50)
)
normalize_all!(hmm)

(hmm, lL) = baum_welch(
    hmm, all_trajs, length(all_trajs);
    max_iterations = 500
)
lL

hmm.forw
hmm.turn

mean(hmm.forw), std(hmm.forw)
mean(hmm.turn), std(hmm.turn)

hmm.transition_matrix
hmm.initial_probs

eigvals(hmm.transition_matrix')
eigvecs(hmm.transition_matrix')[:,4] / sum(eigvecs(hmm.transition_matrix')[:,4])


viterbi(hmm, all_trajs, length(all_trajs))


#= FLR model =#

hmm_flr = ZebrafishHMM_FLR(
    rand(3),
    rand(3, 3),
    Normal(0, 10),
    Gamma(0.6, 50)
)
normalize_all!(hmm_flr)
(hmm_flr, lL_flr) = baum_welch(
    hmm_flr, all_trajs, length(all_trajs);
    max_iterations = 500
)
lL[end]
lL_flr[end]
