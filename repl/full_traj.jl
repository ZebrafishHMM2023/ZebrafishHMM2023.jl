using ZebrafishHMM2023: ZebrafishHMM_G3_Sym_Full, load_full_obs, ATol, normalize_transition_matrix, ZebrafishHMM_G3_Sym_Full_Exp
using HiddenMarkovModels: baum_welch, logdensityof
using Distributions: Gamma, Exponential
using Statistics: mean, std

trajs = load_full_obs(18)

hmm = ZebrafishHMM_G3_Sym_Full_Exp(;
    pinit_turn=rand(), transition_matrix=normalize_transition_matrix(rand(3,3)),
    σforw=0.1, turn=Gamma(1.5, 20.0),
    forward_displacement=Gamma(1.5, 1.2), turn_displacement=Gamma(1.5, 1.2),
    forward_interboutinterval=Exponential(1.8), turn_interboutinterval=Exponential(1.8),
    min_alpha=1.0
)
(hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-7))
hmm

rand(hmm, 100)
