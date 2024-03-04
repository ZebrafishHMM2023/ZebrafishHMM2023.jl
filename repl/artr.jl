import CairoMakie
import Makie
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

artr_wolf_2023_folder(18)

artr_wolf_2023_distances_folder()
