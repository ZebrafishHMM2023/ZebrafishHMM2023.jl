module ZebrafishHMM2023

import HiddenMarkovModels
import StatsAPI
import DensityInterface
using LazyArtifacts: LazyArtifacts, @artifact_str
using HDF5: h5open, attrs
using Distributions: Normal, Gamma, logpdf, AffineDistribution, UnivariateDistribution
using StatsAPI: fit
using NegativeGammaDistributions: NegativeGamma
using DensityInterface: logdensityof
using Random: AbstractRNG

include("artifacts.jl")
include("signed_gamma.jl")
include("missing.jl")
include("hmm.jl")
include("data.jl")

end
