using ZebrafishHMM2023: load_behaviour_free_swimming_data, normalize_all!,
    normalize_transition_matrix!,
    load_behaviour_free_swimming_trajs,
    ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04, ZebrafishHMM_TN04_Sym,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4, markov_equilibrium, stubborness_factor,
    ZebrafishHMM_Elife2020, ZebrafishHMM_Elife2020_Gamma, ZebrafishHMM_G4_Sym
using HiddenMarkovModels: baum_welch, logdensityof, forward_backward, forward, viterbi,
    transition_matrix, initial_distribution
using Statistics: mean, std
using Distributions: Normal, Gamma, fit_mle
using LinearAlgebra

data = load_behaviour_free_swimming_data(18)
data.bouttime
data.
