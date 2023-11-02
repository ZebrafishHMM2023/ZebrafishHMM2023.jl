using Test: @test, @testset
using ZebrafishHMM2023: ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4,
    save_hmm, load_hmm
using Distributions: Normal, Gamma
using Statistics: mean, std

@testset "ZebrafishHMM_G3 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_G3(rand(3), rand(3,3), Normal(0, 10), Gamma(0.6, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_G3)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_G4 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_G4(rand(4), rand(4,4), Normal(0, 10), Gamma(0.6, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_G4)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_TN03 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_TN03(rand(3), rand(3,3), Normal(0, 10), Normal(0.0, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_TN03)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_TN04 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_TN04(rand(4), rand(4,4), Normal(0, 10), Normal(0.0, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_TN04)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_TN3 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_TN3(rand(3), rand(3,3), Normal(0, 10), Normal(0.5, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_TN3)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_TN4 I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_TN4(rand(4), rand(4,4), Normal(0, 10), Normal(0.5, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_TN4)
    @test hmm_loaded.initial_probs == hmm.initial_probs
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test mean(hmm_loaded.forw) == mean(hmm.forw)
    @test std(hmm_loaded.forw) == std(hmm.forw)
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end
