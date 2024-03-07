import CairoMakie
import Makie
import MAT
using HiddenMarkovModels: logdensityof, baum_welch, transition_matrix, initial_distribution, viterbi
using Statistics: mean
using Test: @test, @testset
using ZebrafishHMM2023: artr_wolf_2023_distances
using ZebrafishHMM2023: artr_wolf_2023_distances_file
using ZebrafishHMM2023: artr_wolf_2023_distances_folder
using ZebrafishHMM2023: artr_wolf_2023_fishes
using ZebrafishHMM2023: artr_wolf_2023_folder
using ZebrafishHMM2023: artr_wolf_2023_temperatures
using ZebrafishHMM2023: ATol
using ZebrafishHMM2023: dunn2016hbo_data_dir
using ZebrafishHMM2023: easy_train_artr_hmm
using ZebrafishHMM2023: HMM_ARTR
using ZebrafishHMM2023: load_artr_wolf_2023
using ZebrafishHMM2023: load_artr_wolf_2023
using ZebrafishHMM2023: normalize_transition_matrix
using ZebrafishHMM2023: viterbi_artr

MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "frame_turn.mat"))["frame_turn"]

MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "frame_turn.mat"))["frame_turn"]

MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140418", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "frame_turn.mat"))["frame_turn"]

MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_1", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_3", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_2", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140805", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140820", "frame_turn.mat"))["frame_turn"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140827", "frame_turn.mat"))["frame_turn"]



MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140418", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "L_HBO.mat"))["L_HBO"]

MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_1", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_3", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_2", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140805", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140820", "L_HBO.mat"))["L_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140827", "L_HBO.mat"))["L_HBO"]



MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140418", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "R_HBO.mat"))["R_HBO"]

MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_1", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141015_3", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_2", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140805", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140820", "R_HBO.mat"))["R_HBO"]
MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20140827", "R_HBO.mat"))["R_HBO"]



let fig = Makie.Figure()
    data = MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "frame_turn.mat"))["frame_turn"]
    #data = MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "frame_turn.mat"))["frame_turn"]

    ax = Makie.Axis(fig[1,1], width=400, height=400)
    Makie.hist!(ax, vec(data); bins=-0.2:0.001:0.2, normalization=:pdf)
    Makie.resize_to_layout!(fig)
    fig
end

mean(MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "L_HBO.mat"))["L_HBO"] .< 0)
mean(MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "R_HBO.mat"))["R_HBO"] .< 0)
mean(MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "L_HBO.mat"))["L_HBO"] .< 0)
mean(MAT.matread(joinpath(dunn2016hbo_data_dir(), "cyto", "20140604", "R_HBO.mat"))["R_HBO"] .< 0)



let fig = Makie.Figure()
    data_L = MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "L_HBO.mat"))["L_HBO"]
    data_R = MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "R_HBO.mat"))["R_HBO"]
    data_turn = MAT.matread(joinpath(dunn2016hbo_data_dir(), "nuc", "20141017_1", "frame_turn.mat"))["frame_turn"]

    ax = Makie.Axis(fig[1,1], width=400, height=200)
    Makie.lines!(ax, dropdims(mean(data_L; dims=1); dims=1); color=:blue)
    Makie.lines!(ax, dropdims(mean(data_R; dims=1); dims=1); color=:red)
    Makie.resize_to_layout!(fig)
    fig
end

only(keys(MAT.matread(artr_wolf_2023_distances_file(; temperature=18, fish=14))))



for f = readdir(artr_wolf_2023_distances_folder())
    println(f)
end


MAT.matread(artr_wolf_2023_distances_file(; temperature=18, fish=14))["dist_"]
