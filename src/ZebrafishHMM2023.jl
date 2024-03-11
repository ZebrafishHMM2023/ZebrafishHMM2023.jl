module ZebrafishHMM2023

import DensityInterface
import HiddenMarkovModels
import Optim
import Optimization
import OptimizationOptimJL
import StatsAPI
import NonlinearSolve

using DampedUpdates: damp
using DensityInterface: logdensityof
using Distributions: Normal, Gamma, AffineDistribution, truncated, fit_mle, params, Exponential
using HDF5: h5open, attrs
using LazyArtifacts: LazyArtifacts, @artifact_str
using LinearAlgebra: eigvecs, dot, normalize
using LogExpFunctions: logistic, log1pexp
using MAT: matread
using Random: AbstractRNG
using SpecialFunctions: logerfcx, erfcx
using StaticArrays: SMatrix, @SMatrix, @SVector
using Statistics: mean, std, middle
using StatsAPI: fit
using LogarithmicNumbers: LogFloat64

include("artifacts.jl")
include("missing.jl")
include("data.jl")
include("util.jl")
include("truncated_normal.jl")
include("hmm.jl")
include("signed_gamma.jl")
include("ATol.jl")
include("wolf_artr.jl")
include("wolf_eyes.jl")

include("models/gamma/hmm_g2_sym.jl")
include("models/gamma/hmm_g3.jl")
include("models/gamma/hmm_g3_sym.jl")
include("models/gamma/hmm_g4.jl")
include("models/gamma/hmm_g4_sym.jl")
include("models/trunc_norm_0/hmm_tn03.jl")
include("models/trunc_norm_0/hmm_tn04.jl")
include("models/trunc_norm_0/hmm_tn04_sym.jl")
include("models/trunc_norm/hmm_tn3.jl")
include("models/trunc_norm/hmm_tn4.jl")
include("models/elife2020/elife2020.jl")
include("models/elife2020/elife2020_gamma.jl")
include("models/artr/hmm_artr.jl")
include("models/artr/hmm_artr_log.jl")
include("models/artr/hmm_artr_3_sym.jl")
include("models/full_traj_models/hmm_g3_sym_full.jl")
include("models/full_traj_models/hmm_g3_sym_full_exp.jl")
include("models/artr/hmm_artr_log_freeze.jl")

end
