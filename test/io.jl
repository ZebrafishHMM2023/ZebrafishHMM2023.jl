using Test: @test, @testset
using ZebrafishHMM2023: ZebrafishHMM_G3, ZebrafishHMM_G4, ZebrafishHMM_TN03, ZebrafishHMM_TN04,
    ZebrafishHMM_TN3, ZebrafishHMM_TN4, ZebrafishHMM_G3_Sym, ZebrafishHMM_G4_Sym,
    save_hmm, load_hmm, HMM_ARTR_Log, normalize_transition_matrix
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

@testset "ZebrafishHMM_G3_Sym I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_G3_Sym(rand(), rand(3,3), 10.5, Gamma(0.6, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_G3_Sym)
    @test hmm_loaded.pinit_turn == hmm.pinit_turn
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test hmm_loaded.σforw == hmm.σforw
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "ZebrafishHMM_G4_Sym I/O" begin
    h5_file = tempname()
    hmm = ZebrafishHMM_G4_Sym(rand(), rand(4,4), 10.3, Gamma(0.6, 32))
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, ZebrafishHMM_G4_Sym)
    @test hmm_loaded.pinit_turn == hmm.pinit_turn
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test hmm_loaded.σforw == hmm.σforw
    @test mean(hmm_loaded.turn) == mean(hmm.turn)
    @test std(hmm_loaded.turn) == std(hmm.turn)
end

@testset "HMM_ARTR_Log I/O" begin
    h5_file = tempname()
    hmm = HMM_ARTR_Log(normalize_transition_matrix(rand(3,3)), randn(128, 3), 5.0)
    save_hmm(h5_file, hmm)
    hmm_loaded = load_hmm(h5_file, HMM_ARTR_Log)
    @test hmm_loaded.pinit == hmm.pinit
    @test hmm_loaded.transition_matrix == hmm.transition_matrix
    @test hmm_loaded.h == hmm.h
    @test hmm_loaded.pseudocount == hmm.pseudocount
end
