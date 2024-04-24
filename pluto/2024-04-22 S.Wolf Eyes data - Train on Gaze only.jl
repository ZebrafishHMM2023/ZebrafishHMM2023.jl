### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ d9d420e7-9df0-4aa5-8b18-b9ec77a9021e
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ be486509-c91a-49e9-b8ec-65fde601dab5
using Statistics: mean

# ╔═╡ a22b7e35-84ca-424a-b888-b933d5637436
using Distributions: Normal

# ╔═╡ beb44192-483c-405c-832c-d6d6a5b99312
md"# Imports"

# ╔═╡ 59bee291-2110-4630-abc0-bcf07dcfabb8
import CairoMakie

# ╔═╡ 79d4c806-70e0-467a-975c-77dbb2cc1549
import Makie

# ╔═╡ d8890ba7-bbc7-4de0-9f29-eae2a980b6e1
import ZebrafishHMM2023

# ╔═╡ fd8a3f50-4900-4903-bea8-c8bc3ae2aee4
import HiddenMarkovModels

# ╔═╡ 288abaae-6e8e-4db2-b683-6de78cea5a4a
md"# Functions"

# ╔═╡ 7bfacac0-d6fd-4f69-b592-126149278453
md"# Analysis"

# ╔═╡ 3c3796a8-68c9-45ca-be0d-f03694f227fc
raw_data = ZebrafishHMM2023.wolf_eyes_20240422_data()

# ╔═╡ 463c30c9-c4c2-4042-80c9-6bc08f531b47
num_neurons_artr = size(raw_data.left, 1) + size(raw_data.right, 1)

# ╔═╡ 031f37e6-7db0-4043-a335-b0a5325b90c4
hmm_num_states = 2

# ╔═╡ 4081c0dd-85a0-4be2-af35-7bc2a00fa609
hmm_init = ZebrafishHMM2023.HMM_Gaze(
	ZebrafishHMM2023.normalize_transition_matrix(rand(hmm_num_states, hmm_num_states)),
	randn(hmm_num_states), 1 .+ rand(hmm_num_states)
)

# ╔═╡ 93c434e0-dd63-44b9-95e6-a1ec2a9c5fbe
hmm_trained, train_lls = HiddenMarkovModels.baum_welch(
	hmm_init, raw_data.gaze; max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7)
)

# ╔═╡ 6a331503-d152-4948-813f-0b2016c1f9d3
viterbi_states = HiddenMarkovModels.viterbi(hmm_trained, raw_data.gaze)

# ╔═╡ 43156786-b2d5-4a72-ae34-6316dfcce8ca
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :blue]
    ax = Makie.Axis(fig[1,1], width=1500, height=300, xlabel="time", ylabel="gaze")
	Makie.lines!(ax, raw_data.gaze, color=:gray)
    for s = 1:length(hmm_trained)
		Makie.scatter!(ax, findall(viterbi_states .== s), raw_data.gaze[viterbi_states .== s], markersize=7, color=_colors[s], label="$s")
    end
	Makie.ylims!(ax, -15, 15)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 72a1b761-3acc-48d5-b943-f2d217ad1ef9
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=500, xticks=1:hmm_num_states, yticks=1:hmm_num_states)
	plt = Makie.heatmap!(ax, float(hmm_trained.transition_matrix))
	Makie.Colorbar(fig[1,2], plt)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ e302a623-9e3e-4df2-967f-64425b444502
float(sum(hmm_trained.transition_matrix; dims=2))

# ╔═╡ f8a21e49-4cad-4066-a8ec-b9dd50cf98fd
float(hmm_trained.transition_matrix)

# ╔═╡ Cell order:
# ╠═beb44192-483c-405c-832c-d6d6a5b99312
# ╠═d9d420e7-9df0-4aa5-8b18-b9ec77a9021e
# ╠═59bee291-2110-4630-abc0-bcf07dcfabb8
# ╠═79d4c806-70e0-467a-975c-77dbb2cc1549
# ╠═d8890ba7-bbc7-4de0-9f29-eae2a980b6e1
# ╠═fd8a3f50-4900-4903-bea8-c8bc3ae2aee4
# ╠═be486509-c91a-49e9-b8ec-65fde601dab5
# ╠═a22b7e35-84ca-424a-b888-b933d5637436
# ╠═288abaae-6e8e-4db2-b683-6de78cea5a4a
# ╠═7bfacac0-d6fd-4f69-b592-126149278453
# ╠═3c3796a8-68c9-45ca-be0d-f03694f227fc
# ╠═463c30c9-c4c2-4042-80c9-6bc08f531b47
# ╠═031f37e6-7db0-4043-a335-b0a5325b90c4
# ╠═4081c0dd-85a0-4be2-af35-7bc2a00fa609
# ╠═93c434e0-dd63-44b9-95e6-a1ec2a9c5fbe
# ╠═6a331503-d152-4948-813f-0b2016c1f9d3
# ╠═43156786-b2d5-4a72-ae34-6316dfcce8ca
# ╠═72a1b761-3acc-48d5-b943-f2d217ad1ef9
# ╠═e302a623-9e3e-4df2-967f-64425b444502
# ╠═f8a21e49-4cad-4066-a8ec-b9dd50cf98fd
