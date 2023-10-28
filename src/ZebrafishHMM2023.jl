module ZebrafishHMM2023

import HiddenMarkovModels
using LazyArtifacts: LazyArtifacts, @artifact_str
using HDF5: h5open, attrs

include("artifacts.jl")
include("hmm.jl")

end
