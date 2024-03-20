import CairoMakie
import Dates
import Makie
import MAT
import ZebrafishHMM2023
using HiddenMarkovModels: baum_welch
using HiddenMarkovModels: viterbi
using HiddenMarkovModels: logdensityof
using Statistics: mean
using Distributions: Normal
using ZebrafishHMM2023: ATol
using ZebrafishHMM2023: normalize_transition_matrix

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=700, height=200, xlabel="time", ylabel="gaze orientation")
    Makie.lines!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))
    Makie.lines!(ax, 1:20:36000,
        map(mean, Iterators.partition(vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]), 20))
    )
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    Makie.hist!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))
    Makie.resize_to_layout!(fig)
    fig
end

ZebrafishHMM2023.wolf_eyes_artr_data().left
vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"])


let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=200, xlabel="time", ylabel="mL, mR")
    Makie.lines!(ax, vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left; dims=1)))
    Makie.lines!(ax, vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right; dims=1)))
    Makie.xlims!(ax, 1000, 1200)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="mL", ylabel="mR")
    Makie.hexbin!(ax,
        vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left; dims=1)),
        vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right; dims=1)),
        bins=50
    )
    Makie.scatter!(ax,
        vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left; dims=1)),
        vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right; dims=1)),
        markersize=2, color=:black
    )
    Makie.xlims!(ax, 0, 1)
    Makie.ylims!(ax, 0, 1)
    Makie.resize_to_layout!(fig)
    fig
end


ZebrafishHMM2023.wolf_eyes_artr_data().left

diff(vec(ZebrafishHMM2023.wolf_eyes_data()["tframes"]))


ZebrafishHMM2023.wolf_eyes_data()["tframes"]

gaze_data = vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"])
gaze_data_subsampled = map(mean, Iterators.partition(gaze_data, 20))

nstates = 3
hmm = ZebrafishHMM2023.HMM_Gaze(normalize_transition_matrix(rand(nstates, nstates)), randn(nstates), ones(nstates))
hmm, lLs = baum_welch(hmm, gaze_data_subsampled)

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    Makie.lines!(ax, lLs)
    Makie.resize_to_layout!(fig)
    fig
end

states = viterbi(hmm, gaze_data_subsampled)

let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    _xs = -30:0.01:15
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    for s = 1:nstates
        Makie.hist!(ax, gaze_data_subsampled[states .== s], normalization=:pdf, color=(_colors[s], 0.6))
        Makie.lines!(ax, _xs, [exp(logdensityof(Normal(hmm.μ[s], hmm.σ[s]), x)) for x = _xs], color=_colors[s])
    end
    Makie.resize_to_layout!(fig)
    #Makie.save("temp/$(Dates.today())-gaze-viterbi.pdf", fig)
    fig
end


data = ZebrafishHMM2023.gaze_artr_data()
hmm = ZebrafishHMM2023.HMM_Gaze_ARTR(normalize_transition_matrix(rand(3,3)), randn(3), ones(3), randn(774,3), 1.0)
(hmm, lL) = baum_welch(hmm, data; max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-7))

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=200)
    Makie.scatterlines!(ax, lL)
    Makie.resize_to_layout!(fig)
    fig
end

artr_data = ZebrafishHMM2023.wolf_eyes_artr_data()

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=150, xlabel="neuron", ylabel="h")
    Makie.scatter!(ax, hmm.emit[1].logbias, color=:blue)
    Makie.scatter!(ax, hmm.emit[2].logbias, color=:red)
    Makie.scatter!(ax, hmm.emit[3].logbias, color=:black, marker=:cross)
    Makie.vlines!(ax, size(artr_data.left, 1), color=:orange, linestyle=:dash, linewidth=3)
    Makie.resize_to_layout!(fig)
    fig
end


let fig = Makie.Figure()
    states = viterbi(hmm, data)
    _colors = [:teal, :orange, :purple, :pink]
    _xs = -30:0.01:15
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    for s = 1:length(hmm)
        Makie.hist!(ax, map(first, data[states .== s]), normalization=:pdf, color=(_colors[s], 0.6))
        Makie.lines!(ax, _xs, [exp(logdensityof(hmm.emit[s].gaze, x)) for x = _xs], color=_colors[s])
    end
    Makie.resize_to_layout!(fig)
    #Makie.save("temp/$(Dates.today())-gaze-viterbi.pdf", fig)
    fig
end


let fig = Makie.Figure()
    states = viterbi(hmm, data)
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="mL", ylabel="mR")
    for s = 1:length(hmm)
        Makie.scatter!(ax,
            vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left[:, states .== s]; dims=1)),
            vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right[:, states .== s]; dims=1)),
            markersize=5, color=_colors[s], label="$s"
        )
    end
    Makie.xlims!(ax, 0, 0.6)
    Makie.ylims!(ax, 0, 0.6)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end


data = ZebrafishHMM2023.gaze_artr_data()
hmm = ZebrafishHMM2023.HMM_Eyes_ARTR_Only(normalize_transition_matrix(rand(3,3)), randn(774,3), 1.0)
(hmm, lL) = baum_welch(hmm, last.(data); max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-7))

let fig = Makie.Figure()
    states = viterbi(hmm, last.(data))
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="mL", ylabel="mR")
    for s = 1:length(hmm)
        Makie.scatter!(ax,
            vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left[:, states .== s]; dims=1)),
            vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right[:, states .== s]; dims=1)),
            markersize=5, color=_colors[s], label="$s"
        )
    end
    Makie.xlims!(ax, 0, 0.6)
    Makie.ylims!(ax, 0, 0.6)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end
