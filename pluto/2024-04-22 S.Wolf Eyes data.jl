### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 32deac33-32dd-454f-96fd-1276e03d1a68
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ eca1a5cb-7862-4afe-b5ba-efeb864da254
using Statistics: mean

# ╔═╡ 54f44afc-9a47-4f55-a4fe-0e09769e86ac
using Distributions: Normal

# ╔═╡ 5c714840-00a9-11ef-0a61-3de89cd98ac4
md"# Imports"

# ╔═╡ 7061506e-960a-4afd-a7eb-bd8a5dbada44
import CairoMakie

# ╔═╡ d27e9774-7c03-4fe0-a3fe-74c079174edd
import Makie

# ╔═╡ 847e1a28-9f2d-4312-bf9f-99aa97253288
import Dates

# ╔═╡ 3ca19369-0c47-46be-9f9f-f6aa3106f633
import MAT

# ╔═╡ 06ced9bc-1032-4d1b-9112-50fcc0e3942e
import ZebrafishHMM2023

# ╔═╡ 13391d05-6c9c-4ec8-8903-558c98ed50a0
import HiddenMarkovModels

# ╔═╡ fa653470-e407-4e4d-aabf-1ea50be0d588
md"# Functions"

# ╔═╡ 55fe0491-52e4-4bf8-aac8-0abe0a6269d3
md"# Analysis"

# ╔═╡ 998e04c9-fe23-49b1-9c40-80abeb366a06


# ╔═╡ b63ed9af-009a-4e24-a4d5-0bcfa793a53d
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=200, xlabel="time", ylabel="mL, mR")
    Makie.lines!(ax, vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().left; dims=1)))
    Makie.lines!(ax, vec(mean(ZebrafishHMM2023.wolf_eyes_artr_data().right; dims=1)))
    Makie.xlims!(ax, 1000, 1200)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 273641fa-3a42-4746-a66a-9ba5068a5488
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=1000, height=200, xlabel="time", ylabel="gaze orientation")
    Makie.lines!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))
    Makie.lines!(ax, 1:20:36000,
        map(mean, Iterators.partition(vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]), 20))
    )
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 674fffd0-b070-4d56-956a-a76076b590fa
hmm_num_states = 3

# ╔═╡ ebd6ea32-e0d4-474a-9009-fb96eb685e5c
raw_data = ZebrafishHMM2023.wolf_eyes_20240415_data()

# ╔═╡ 6ca8dccc-95e1-4428-99aa-3d7a0d3f44c0
num_neurons_artr = size(raw_data.left, 1) + size(raw_data.right, 1)

# ╔═╡ b3e898af-c598-46b6-af1e-14f57d71f19b
hmm_init = ZebrafishHMM2023.HMM_Eyes_ARTR_Only(
	ZebrafishHMM2023.normalize_transition_matrix(rand(hmm_num_states, hmm_num_states)),
	randn(num_neurons_artr, hmm_num_states), 1.0
)

# ╔═╡ ff64716d-4b7d-4f66-9e93-89c1ec56e224
hmm_trained, train_lls = HiddenMarkovModels.baum_welch(
	hmm_init, eachcol(vcat(raw_data.left, raw_data.right));
	max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7)
)

# ╔═╡ a704d8eb-1e1c-4924-beac-d10412d7a75e
viterbi_states = HiddenMarkovModels.viterbi(hmm_trained, eachcol(vcat(raw_data.left, raw_data.right)))

# ╔═╡ 9f76e032-77cf-408a-abe5-049c78433d88
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="mL", ylabel="mR")
    for s = 1:length(hmm_trained)
        Makie.scatter!(ax,
            vec(mean(raw_data.left[:, viterbi_states .== s]; dims=1)),
            vec(mean(raw_data.right[:, viterbi_states .== s]; dims=1));
            markersize=5, color=_colors[s], label="$s"
        )
    end
    Makie.xlims!(ax, -0.01, 0.7)
    Makie.ylims!(ax, -0.01, 0.7)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ Cell order:
# ╠═5c714840-00a9-11ef-0a61-3de89cd98ac4
# ╠═32deac33-32dd-454f-96fd-1276e03d1a68
# ╠═7061506e-960a-4afd-a7eb-bd8a5dbada44
# ╠═d27e9774-7c03-4fe0-a3fe-74c079174edd
# ╠═847e1a28-9f2d-4312-bf9f-99aa97253288
# ╠═3ca19369-0c47-46be-9f9f-f6aa3106f633
# ╠═06ced9bc-1032-4d1b-9112-50fcc0e3942e
# ╠═13391d05-6c9c-4ec8-8903-558c98ed50a0
# ╠═eca1a5cb-7862-4afe-b5ba-efeb864da254
# ╠═54f44afc-9a47-4f55-a4fe-0e09769e86ac
# ╠═fa653470-e407-4e4d-aabf-1ea50be0d588
# ╠═55fe0491-52e4-4bf8-aac8-0abe0a6269d3
# ╠═998e04c9-fe23-49b1-9c40-80abeb366a06
# ╠═b63ed9af-009a-4e24-a4d5-0bcfa793a53d
# ╠═273641fa-3a42-4746-a66a-9ba5068a5488
# ╠═674fffd0-b070-4d56-956a-a76076b590fa
# ╠═ebd6ea32-e0d4-474a-9009-fb96eb685e5c
# ╠═6ca8dccc-95e1-4428-99aa-3d7a0d3f44c0
# ╠═b3e898af-c598-46b6-af1e-14f57d71f19b
# ╠═ff64716d-4b7d-4f66-9e93-89c1ec56e224
# ╠═a704d8eb-1e1c-4924-beac-d10412d7a75e
# ╠═9f76e032-77cf-408a-abe5-049c78433d88
