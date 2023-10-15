import Aqua
import ZebrafishHMM2023
using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(ZebrafishHMM2023; ambiguities = false)
end
