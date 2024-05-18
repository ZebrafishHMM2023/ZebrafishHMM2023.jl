module ZebrafishHMM2023

import CairoMakie
import DensityInterface
import HiddenMarkovModels
import LazyArtifacts
import Makie
import Random
import StatsAPI
using DensityInterface: logdensityof
using Distributions: AffineDistribution
using Distributions: Exponential
using Distributions: fit_mle
using Distributions: Gamma
using Distributions: logpdf
using Distributions: Normal
using Distributions: params
using Distributions: truncated
using HDF5: attrs
using HDF5: h5open
using LazyArtifacts: @artifact_str
using LinearAlgebra: dot
using LinearAlgebra: eigvecs
using LinearAlgebra: normalize
using LogarithmicNumbers: LogFloat64
using LogExpFunctions: log1pexp
using LogExpFunctions: logistic
using MAT: matread
using Random: AbstractRNG
using SpecialFunctions: erfcx
using SpecialFunctions: logerfcx
using StaticArrays: @SMatrix
using StaticArrays: @SVector
using StaticArrays: SMatrix
using Statistics: mean
using Statistics: middle
using Statistics: std
using StatsAPI: fit

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
include("models/artr/hmm_artr_m_diff.jl")
include("models/eyes/hmm_gaze.jl")
include("models/eyes/hmm_gaze_artr.jl")
include("models/eyes/hmm_eyes_artr_only.jl")

end
