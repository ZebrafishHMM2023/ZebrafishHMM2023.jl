module ZebrafishHMM2023

import HiddenMarkovModels
import StatsAPI
import DensityInterface
using LazyArtifacts: LazyArtifacts, @artifact_str
using HDF5: h5open, attrs
using Distributions: Normal, Gamma, AffineDistribution
using StatsAPI: fit
using DensityInterface: logdensityof
using Random: AbstractRNG

include("artifacts.jl")
include("missing.jl")
include("hmm.jl")
include("data.jl")

include("models/hmm_flr.jl")

end
