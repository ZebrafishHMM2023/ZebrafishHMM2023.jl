import CairoMakie
import Makie
import MAT
import ZebrafishHMM2023
using HiddenMarkovModels: baum_welch
using HiddenMarkovModels: viterbi
using HiddenMarkovModels: logdensityof
using Statistics: mean
using Distributions: Normal
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


ZebrafishHMM2023.wolf_eyes_artr_data().left


diff(vec(ZebrafishHMM2023.wolf_eyes_data()["tframes"]))


ZebrafishHMM2023.wolf_eyes_data()["tframes"]

nstates = 3
hmm = ZebrafishHMM2023.HMM_Gaze(normalize_transition_matrix(rand(nstates, nstates)), randn(nstates), ones(nstates))
hmm, lLs = baum_welch(hmm, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    Makie.lines!(ax, lLs)
    Makie.resize_to_layout!(fig)
    fig
end

states = viterbi(hmm, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))

let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    _xs = -35:0.01:15
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    for s = 1:nstates
        Makie.hist!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"])[states .== s], normalization=:pdf, color=(_colors[s], 0.6))
        Makie.lines!(ax, _xs, [exp(logdensityof(Normal(hmm.μ[s], hmm.σ[s]), x)) for x = _xs], color=_colors[s])
    end
    Makie.resize_to_layout!(fig)
    fig
end
