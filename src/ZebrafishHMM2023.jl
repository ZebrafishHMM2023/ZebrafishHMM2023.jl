module ZebrafishHMM2023

import HiddenMarkovModels
import StatsAPI
import DensityInterface
import Optimization
import OptimizationOptimJL
import Optim
using LazyArtifacts: LazyArtifacts, @artifact_str
using HDF5: h5open, attrs
using Distributions: Normal, Gamma, AffineDistribution, truncated, fit_mle
using StatsAPI: fit
using DensityInterface: logdensityof
using Random: AbstractRNG
using Statistics: mean, std
using SpecialFunctions: logerfcx

include("artifacts.jl")
include("missing.jl")
include("data.jl")
include("util.jl")
include("truncated_normal.jl")

include("models/gamma/hmm_g3.jl")
include("models/gamma/hmm_g4.jl")
include("models/trunc_norm_0/hmm_tn03.jl")
include("models/trunc_norm_0/hmm_tn04.jl")
include("models/trunc_norm/hmm_tn3.jl")
include("models/trunc_norm/hmm_tn4.jl")

end
