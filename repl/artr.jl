import CairoMakie
import Makie
import MAT
using HiddenMarkovModels: logdensityof, baum_welch, transition_matrix, initial_distribution, viterbi
using Statistics: mean
using Test: @test, @testset
using ZebrafishHMM2023: ATol
using ZebrafishHMM2023: easy_train_artr_hmm
using ZebrafishHMM2023: HMM_ARTR
using ZebrafishHMM2023: load_artr_wolf_2023
using ZebrafishHMM2023: normalize_transition_matrix
using ZebrafishHMM2023: viterbi_artr
using ZebrafishHMM2023: artr_wolf_2023_folder
using ZebrafishHMM2023: artr_wolf_2023_distances_folder
using ZebrafishHMM2023: artr_wolf_2023_distances_file
using ZebrafishHMM2023: artr_wolf_2023_distances
using ZebrafishHMM2023: artr_wolf_2023_temperatures
using ZebrafishHMM2023: artr_wolf_2023_fishes
using ZebrafishHMM2023: load_artr_wolf_2023




only(keys(MAT.matread(artr_wolf_2023_distances_file(; temperature=18, fish=14))))



for f = readdir(artr_wolf_2023_distances_folder())
    println(f)
end


MAT.matread(artr_wolf_2023_distances_file(; temperature=18, fish=14))["dist_"]
