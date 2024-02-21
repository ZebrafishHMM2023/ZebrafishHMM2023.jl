import Makie
import CairoMakie

using HiddenMarkovModels: logdensityof, baum_welch, transition_matrix, initial_distribution, viterbi
using Statistics: mean
using Test: @test, @testset
using ZebrafishHMM2023: load_artr_wolf_2023, HMM_ARTR, normalize_transition_matrix, ATol, viterbi_artr

data = load_artr_wolf_2023(; temperature=18, fish=12)
trajs = collect(eachcol(vcat(data.left, data.right)))
hmm = HMM_ARTR([0.75 0.25; 0.25 0.75], randn(326, 2), 0.1)
(hmm, lL) = baum_welch(hmm, trajs; max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-7))

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=200)
    Makie.scatterlines!(ax, lL)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=300)
    Makie.lines!(ax, vec(mean(data.left; dims=1)))
    Makie.lines!(ax, vec(mean(data.right; dims=1)))
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 1), zeros(count(viterbi(hmm, trajs) .== 1)), color=:blue)
    Makie.scatter!(ax, findall(viterbi(hmm, trajs) .== 2), ones(count(viterbi(hmm, trajs) .== 2)), color=:red)

    Makie.scatter!(ax, findall(viterbi_artr(hmm, stack(trajs)) .== 1), 0.1ones(count(viterbi_artr(hmm, stack(trajs)) .== 1)), color=:cyan)
    Makie.scatter!(ax, findall(viterbi_artr(hmm, stack(trajs)) .== 2), 0.9ones(count(viterbi_artr(hmm, stack(trajs)) .== 2)), color=:orange)

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
