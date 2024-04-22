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

# ╔═╡ 06ced9bc-1032-4d1b-9112-50fcc0e3942e
import ZebrafishHMM2023

# ╔═╡ 13391d05-6c9c-4ec8-8903-558c98ed50a0
import HiddenMarkovModels

# ╔═╡ fa653470-e407-4e4d-aabf-1ea50be0d588
md"# Functions"

# ╔═╡ 55fe0491-52e4-4bf8-aac8-0abe0a6269d3
md"# Analysis"

# ╔═╡ 674fffd0-b070-4d56-956a-a76076b590fa
hmm_num_states = 3

# ╔═╡ ebd6ea32-e0d4-474a-9009-fb96eb685e5c
raw_data = ZebrafishHMM2023.wolf_eyes_20240422_data()

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
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="mL", ylabel="mR")
    for s = 1:length(hmm_trained)
        Makie.scatter!(ax,
            vec(mean(raw_data.left[:, viterbi_states .== s]; dims=1)),
            vec(mean(raw_data.right[:, viterbi_states .== s]; dims=1));
            markersize=7, color=(_colors[s], 0.25), label="$s"
        )
    end
    Makie.xlims!(ax, -0.01, 0.5)
    Makie.ylims!(ax, -0.01, 0.5)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 2c62def0-c292-4391-92b1-9fadd03685ea
gaze_data_subsampled = map(mean, ZebrafishHMM2023.chunks(raw_data.gaze, size(raw_data.left, 2)))

# ╔═╡ 9b9456db-c034-4b9e-8c0c-6dfe448b528d
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=1500, height=300, xlabel="time", ylabel="gaze")
	Makie.lines!(ax, gaze_data_subsampled, color=:gray)
    for s = 1:length(hmm_trained)
		Makie.scatter!(ax, findall(viterbi_states .== s), gaze_data_subsampled[viterbi_states .== s], markersize=7, color=_colors[s], label="$s")
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 97a2c9b8-aa69-4a1a-8904-f1c220e4d4e6
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="gaze", ylabel="frequency")
    for s = 1:length(hmm_trained)
		Makie.stephist!(ax, gaze_data_subsampled[viterbi_states .== s], color=_colors[s], label="$s", normalization=:pdf)
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ ce887339-5a5a-4149-a493-37cf7aa8227d


# ╔═╡ Cell order:
# ╠═5c714840-00a9-11ef-0a61-3de89cd98ac4
# ╠═32deac33-32dd-454f-96fd-1276e03d1a68
# ╠═7061506e-960a-4afd-a7eb-bd8a5dbada44
# ╠═d27e9774-7c03-4fe0-a3fe-74c079174edd
# ╠═06ced9bc-1032-4d1b-9112-50fcc0e3942e
# ╠═13391d05-6c9c-4ec8-8903-558c98ed50a0
# ╠═eca1a5cb-7862-4afe-b5ba-efeb864da254
# ╠═54f44afc-9a47-4f55-a4fe-0e09769e86ac
# ╠═fa653470-e407-4e4d-aabf-1ea50be0d588
# ╠═55fe0491-52e4-4bf8-aac8-0abe0a6269d3
# ╠═674fffd0-b070-4d56-956a-a76076b590fa
# ╠═ebd6ea32-e0d4-474a-9009-fb96eb685e5c
# ╠═6ca8dccc-95e1-4428-99aa-3d7a0d3f44c0
# ╠═b3e898af-c598-46b6-af1e-14f57d71f19b
# ╠═ff64716d-4b7d-4f66-9e93-89c1ec56e224
# ╠═a704d8eb-1e1c-4924-beac-d10412d7a75e
# ╠═9f76e032-77cf-408a-abe5-049c78433d88
# ╠═2c62def0-c292-4391-92b1-9fadd03685ea
# ╠═9b9456db-c034-4b9e-8c0c-6dfe448b528d
# ╠═97a2c9b8-aa69-4a1a-8904-f1c220e4d4e6
# ╠═ce887339-5a5a-4149-a493-37cf7aa8227d
