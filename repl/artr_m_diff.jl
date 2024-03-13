import CairoMakie
import Makie
using DelimitedFiles: writedlm
using HiddenMarkovModels: baum_welch
using HiddenMarkovModels: initial_distribution
using HiddenMarkovModels: logdensityof
using HiddenMarkovModels: transition_matrix
using HiddenMarkovModels: viterbi
using Statistics: mean
using Test: @test
using Test: @testset
using ZebrafishHMM2023: ATol
using ZebrafishHMM2023: HMM_ARTR_Log
using ZebrafishHMM2023: HMM_ARTR_m_diff
using ZebrafishHMM2023: load_artr_wolf_2023
using ZebrafishHMM2023: normalize_transition_matrix
using ZebrafishHMM2023: viterbi_artr

data = load_artr_wolf_2023(; temperature=18, fish=12)
traj = vec(mean(data.left; dims=1) - mean(data.right; dims=1))

hmm = HMM_ARTR_m_diff(normalize_transition_matrix(rand(3,3)), randn(3), ones(3))
(hmm, lL) = baum_welch(hmm, trajs; max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-7))

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=200)
    Makie.scatterlines!(ax, lL)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=150, xlabel="neuron", ylabel="h")
    Makie.scatter!(ax, hmm.h[:,1], color=:blue)
    Makie.scatter!(ax, hmm.h[:,3], color=:red)
    Makie.scatter!(ax, hmm.h[:,2], color=:black, marker=:cross)
    Makie.vlines!(ax, size(data.left, 1), color=:orange, linestyle=:dash, linewidth=3)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=300)
    Makie.lines!(ax, vec(mean(data.left; dims=1)))
    Makie.lines!(ax, vec(mean(data.right; dims=1)))
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 1), 0.9ones(count(viterbi(hmm, trajs) .== 1)), color=:blue)
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 2), 0.1ones(count(viterbi(hmm, trajs) .== 2)), color=:black)
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 3), 0.8ones(count(viterbi(hmm, trajs) .== 3)), color=:red)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=300)
    Makie.lines!(ax, vec(mean(data.left; dims=1)))
    Makie.lines!(ax, vec(mean(data.right; dims=1)))
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 1), zeros(count(viterbi(hmm, trajs) .== 1)), color=:blue)
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 2), ones(count(viterbi(hmm, trajs) .== 2)), color=:red)
    Makie.resize_to_layout!(fig)
    fig
end

findall(viterbi_artr(hmm, stack(trajs)) .≠ viterbi(hmm, trajs))


fig = Makie.Figure()
ax = Makie.Axis(fig[1,1], width=400, height=300, xlabel="neuron", ylabel="h")
Makie.lines!(ax, hmm.h[:,1], label="L")
Makie.lines!(ax, hmm.h[:,2], label="R")
Makie.vlines!(ax, size(data.left, 1), color=:black, linestyle=:dash)
Makie.axislegend(ax, position=:lb)
Makie.resize_to_layout!(fig)
fig



viterbi(hmm, trajs)
mean(data.left; dims=1)
