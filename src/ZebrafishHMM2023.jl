module ZebrafishHMM2023

import HiddenMarkovModels
import StatsAPI
import DensityInterface
import Optimization
import OptimizationOptimJL
import Optim
using LazyArtifacts: LazyArtifacts, @artifact_str
using HDF5: h5open, attrs
using Distributions: Normal, Gamma, AffineDistribution, truncated, fit_mle, params
using StatsAPI: fit
using DensityInterface: logdensityof
using Random: AbstractRNG
using Statistics: mean, std, middle
using SpecialFunctions: logerfcx, erfcx
using DampedUpdates: damp
using LinearAlgebra: eigvecs
using StaticArrays: SMatrix, @SMatrix, @SVector

include("artifacts.jl")
include("missing.jl")
include("data.jl")
include("util.jl")
include("truncated_normal.jl")
include("hmm.jl")
include("signed_gamma.jl")

include("models/gamma/hmm_g3.jl")
include("models/gamma/hmm_g4.jl")
include("models/gamma/hmm_g4_sym.jl")
include("models/trunc_norm_0/hmm_tn03.jl")
include("models/trunc_norm_0/hmm_tn04.jl")
include("models/trunc_norm/hmm_tn3.jl")
include("models/trunc_norm/hmm_tn4.jl")
include("models/elife2020/elife2020.jl")
include("models/elife2020/elife2020_gamma.jl")

end
