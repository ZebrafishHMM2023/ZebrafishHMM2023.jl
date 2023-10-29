using ZebrafishHMM2023: load_behaviour_free_swimming_data, ZebrafishHMM, normalize_all!,
    build_trajectories, logdensityof
using HiddenMarkovModels: baum_welch
using Statistics: mean, std
using Distributions: Normal, Gamma, fit

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
    Normal(0, 30),
    Gamma(0.6, 30)
)
normalize_all!(hmm)

(hmm, lL) = baum_welch(
    hmm, all_trajs, length(all_trajs);
    max_iterations = 100
)
lL

hmm.forw
hmm.turn

mean(hmm.forw), std(hmm.forw)
mean(hmm.turn), std(hmm.turn)

hmm.transition_matrix
hmm.initial_probs
