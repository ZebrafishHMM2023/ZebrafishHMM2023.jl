### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 7752ef77-6a36-4f65-b9c9-40bb3f4ace3f
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 703dc739-beab-45ef-a68f-770f47144b60
using Statistics: mean, std, cor

# ╔═╡ 3fe7f379-9cc0-438f-8a35-b5d5431acb39
md"# Imports"

# ╔═╡ d94bf588-c150-4ca8-a1e3-5f564893ba61
import ZebrafishHMM2023

# ╔═╡ e584de87-ad25-400b-8efa-70c9e3b8b277
import HiddenMarkovModels

# ╔═╡ 71c69eed-6c6b-433a-89de-7af3cb2f8240
import Makie

# ╔═╡ 224094e2-419e-460b-8eb7-28a4c92fd78f
import CairoMakie

# ╔═╡ b0f8d461-56e7-4868-8f4a-aea8476e9ee6
import PlutoUI

# ╔═╡ b50b1846-99f4-4bc6-bc34-3b5a7238b44c
import CSV

# ╔═╡ f9e7b5b3-42e9-41dc-af88-a66324baba59
import HDF5

# ╔═╡ f9c9dd2a-98de-4e54-a055-6400f20f6015
import Distributions

# ╔═╡ 7324efa9-21b3-44f5-bd2d-86f5dd81b55f
PlutoUI.TableOfContents()

# ╔═╡ 56f93ecf-e877-422c-8553-5474e9023c77
md"# Data"

# ╔═╡ 004af453-2575-41ed-a00c-ae4fe83e125b
trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ 6cf66c0d-ea7f-4693-834b-7eb08c6d22ac
glob_all_trajs = [t for fish = trajs for t = fish]

# ╔═╡ 1692b2dc-0618-4529-b8c0-357e3a88b977
md"""
# Train global HMM

HMM trained on ALL trajectories (of all fish) together.
"""

# ╔═╡ a1143951-7387-40fd-b28c-1d7b31af5b1b
glob_hmm_init = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(
    rand(),
    rand(3,3),
    1.0,
    Distributions.Gamma(1.1, 15.0),
    1.0
)

# ╔═╡ 149547ad-f8a4-4fca-a4b4-6cf095d0153c
# HMM trained on ALL trajectories together
begin
	(glob_hmm, glob_lLs) = HiddenMarkovModels.baum_welch(glob_hmm_init, glob_all_trajs, length(glob_all_trajs); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	ZebrafishHMM2023.normalize_all!(glob_hmm)
end

# ╔═╡ 47672489-5c37-415e-a93e-021a9595d413
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=900, height=300)
	Makie.lines!(ax, glob_lLs)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ c3b6e192-26ac-455d-8aba-735ebe8573b6
md"""
# Train HMM on all trajectories of each fish
"""

# ╔═╡ e41e8570-ab5b-4237-9464-69099f538d26
begin
	global_hmms = ZebrafishHMM2023.ZebrafishHMM_G3_Sym[]
	global_lLs = Vector{Float64}[]
	for (n, fish) = enumerate(trajs)
	    println("Fitting fish $n of 18.")
		hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Distributions.Gamma(1.5, 20.0), 1.0))
	    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish, length(fish); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	    push!(global_hmms, hmm)
	    push!(global_lLs, lL)
	end
end

# ╔═╡ 51cf389e-87fd-4670-a599-f3a4964f9470
md"# Chunks"

# ╔═╡ 84b22eae-7fde-42e4-aa39-0b59e77d1f4d
n_chunks = 10

# ╔═╡ aabf41fc-4069-4bc6-af5a-20845d22f66a
function train_chunk_hmms(n_chunks::Int; fictitious::Bool = false)
	chunk_hmms = Vector{Vector{ZebrafishHMM2023.ZebrafishHMM_G3_Sym}}()
	chunk_lLs = Vector{Vector{Vector{Float64}}}()
	for (n, fish) = enumerate(trajs)
	    push!(chunk_hmms, ZebrafishHMM2023.ZebrafishHMM_G3_Sym[])
	    push!(chunk_lLs, Vector{Float64}[])
	    for (t, chunk) = enumerate(ZebrafishHMM2023.chunks(fish, n_chunks))
	        println("Fitting fish $n of 18, chunk $t (of length $(length(chunk))).")
			hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Distributions.Gamma(1.5, 20.0), 1.0))

			if fictitious
				# use same global HMM for all fish ... null model for lack of individuality
				my_data = [rand(glob_hmm, length(seq)).obs_seq for seq = chunk]
			else
				my_data = chunk
			end

			# my_data = chunk
			# @show length(my_data)
			
	        (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, my_data, length(my_data); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	        push!(chunk_hmms[n], hmm)
	        push!(chunk_lLs[n], lL)
	    end
	end
	return chunk_hmms, chunk_lLs
end

# ╔═╡ bde3cd11-ede4-409f-959b-a26267e51408
chunk_hmms, chunk_lLs = train_chunk_hmms(n_chunks; fictitious=false)

# ╔═╡ f2251c4d-9164-4b71-8a36-bdbbc6d16e0f
let fig = Makie.Figure()
	_opts = (; width=200, height=200, xlabel="global HMM", ylabel="<chunk HMMs>")

	ax = Makie.Axis(fig[1,1]; _opts..., title="std(forw)")
	Makie.errorbars!(ax, [hmm.σforw for hmm = global_hmms],
	    [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.σforw for hmm = fish) for fish = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [hmm.σforw for hmm = global_hmms], [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms])
	Makie.ylims!(ax, 1, 7)
	
	ax = Makie.Axis(fig[1,2]; _opts..., title="mean(turn)")
	Makie.errorbars!(ax, [mean(hmm.turn) for hmm = global_hmms], 
	    [mean(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms],
	    [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [mean(hmm.turn) for hmm = global_hmms], [mean(mean(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 20, 45)
	
	ax = Makie.Axis(fig[1,3]; _opts..., title="std(turn)")
	Makie.errorbars!(ax, [std(hmm.turn) for hmm = global_hmms], 
	    [mean(std(hmm.turn) for hmm = fish) for fish = chunk_hmms],
	    [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [std(hmm.turn) for hmm = global_hmms], [mean(std(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 13, 28)
	
	ax = Makie.Axis(fig[2,1]; _opts..., title="F->F")
	Makie.errorbars!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], [mean(hmm.transition_matrix[1,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 0, 0.7)
	
	ax = Makie.Axis(fig[2,2]; _opts..., title="F->L, F->R")
	Makie.errorbars!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], [mean(hmm.transition_matrix[1,2] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 0.2, 0.5)
	
	ax = Makie.Axis(fig[2,3]; _opts..., title="L->L, R->R")
	Makie.errorbars!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms],
	    [std(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 0.1, 0.6)
	
	ax = Makie.Axis(fig[2,4]; _opts..., title="L->F, R->F")
	Makie.errorbars!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms],
	    [std(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms] / 2,
	)
	Makie.scatter!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms])
	Makie.ylims!(ax, 0.2, 0.7)
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 94243ca7-eb43-49c2-9ddf-339e59d015fd
sum(global_hmms[1].transition_matrix; dims=2)

# ╔═╡ c129498c-558b-4b03-b69e-e4e7b91f42a4
md"# Autocorrelation of chunk HMMs"

# ╔═╡ 3c3eb078-36da-4398-9c3d-b3c1cbdb253d
σforw_all_autocors = [
	[cor([fish_hmms[t].σforw for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].σforw for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 7ea525cd-dda1-4d0c-bdf9-996dde28f4ee
turn_mean_all_autocors = [
	[cor([mean(fish_hmms[t].turn) for t = 1:length(fish_hmms) - Δ + 1], [mean(fish_hmms[t].turn) for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 9d93f74d-d7eb-4661-9db4-b37a99bb304c
turn_std_all_autocors = [
	[cor([std(fish_hmms[t].turn) for t = 1:length(fish_hmms) - Δ + 1], [std(fish_hmms[t].turn) for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 649a66fb-d4ce-4edc-bdfb-451aa7e6ce6b
FtoF_all_autocors = [
	[cor([fish_hmms[t].transition_matrix[1,1] for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].transition_matrix[1,1] for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ e096de8c-210c-4422-9896-83ccf9d36d93
FtoT_all_autocors = [
	[cor([fish_hmms[t].transition_matrix[1,2] for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].transition_matrix[1,2] for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 3f6af6ed-d3ad-47a5-8e57-d4358211e188
TtoF_all_autocors = [
	[cor([fish_hmms[t].transition_matrix[2,1] for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].transition_matrix[2,1] for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 0a046ed7-281c-47b3-aa55-66bf7e6f4d32
TtoT_all_autocors = [
	[cor([fish_hmms[t].transition_matrix[2,2] for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].transition_matrix[2,2] for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 0f127df4-5fa5-4dc2-acbc-6ab7c00a8de8
LtoR_all_autocors = [
	[cor([fish_hmms[t].transition_matrix[2,3] for t = 1:length(fish_hmms) - Δ + 1], [fish_hmms[t].transition_matrix[2,3] for t = Δ:length(fish_hmms)]) for Δ = 1:floor(Int, length(fish_hmms)) - 2] for fish_hmms = chunk_hmms
]

# ╔═╡ 25fc0377-b52b-4588-a39f-f7970aad20a2
avg_σforw_autocors = [mean(a[Δ] for a = σforw_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, σforw_all_autocors))]

# ╔═╡ 37aa671c-eb79-41cd-b704-0370065f368b
avg_turn_mean_autocors = [mean(a[Δ] for a = turn_mean_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, turn_mean_all_autocors))]

# ╔═╡ 24e1815d-f87b-4bb2-af2a-2b9ae70c1b50
avg_turn_std_autocors = [mean(a[Δ] for a = turn_std_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, turn_std_all_autocors))]

# ╔═╡ a051c3bf-ebee-47fb-b27d-85c2ab8d63c8
FtoF_autocors = [mean(a[Δ] for a = FtoF_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, FtoF_all_autocors))]

# ╔═╡ 315c0697-07c9-4ef8-a3d7-4694349cc7cb
FtoT_autocors = [mean(a[Δ] for a = FtoT_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, FtoT_all_autocors))]

# ╔═╡ e27af890-3076-4bd6-b4de-27f50f660e6c
TtoF_autocors = [mean(a[Δ] for a = TtoF_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, TtoF_all_autocors))]

# ╔═╡ fabd3af9-4983-4ff9-8d58-dede2728f182
TtoT_autocors = [mean(a[Δ] for a = TtoT_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, TtoT_all_autocors))]

# ╔═╡ 3453a770-56d5-42fb-94fc-2b5b906afc7d
LtoR_autocors = [mean(a[Δ] for a = LtoR_all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, LtoR_all_autocors))]

# ╔═╡ 52b8362b-e828-44b5-83ba-61b9cfbc8cc6
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=200, height=200, title="forw std", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(avg_σforw_autocors)

	ax = Makie.Axis(fig[1,2], width=200, height=200, title="turn mean", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(avg_turn_mean_autocors)
	
	ax = Makie.Axis(fig[1,3], width=200, height=200, title="turn std", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(avg_turn_std_autocors)

	ax = Makie.Axis(fig[2,1], width=200, height=200, title="F -> F", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(FtoF_autocors)
	
	ax = Makie.Axis(fig[2,2], width=200, height=200, title="F -> L, R", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(FtoT_autocors)

	ax = Makie.Axis(fig[2,3], width=200, height=200, title="L, R -> F", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(TtoF_autocors)

	ax = Makie.Axis(fig[2,4], width=200, height=200, title="L -> L, R -> R", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(TtoT_autocors)

	ax = Makie.Axis(fig[2,5], width=200, height=200, title="L -> R, R -> L", xlabel="chunk", ylabel="autocor.")
	Makie.scatterlines!(LtoR_autocors)

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ Cell order:
# ╠═3fe7f379-9cc0-438f-8a35-b5d5431acb39
# ╠═7752ef77-6a36-4f65-b9c9-40bb3f4ace3f
# ╠═d94bf588-c150-4ca8-a1e3-5f564893ba61
# ╠═e584de87-ad25-400b-8efa-70c9e3b8b277
# ╠═71c69eed-6c6b-433a-89de-7af3cb2f8240
# ╠═224094e2-419e-460b-8eb7-28a4c92fd78f
# ╠═b0f8d461-56e7-4868-8f4a-aea8476e9ee6
# ╠═b50b1846-99f4-4bc6-bc34-3b5a7238b44c
# ╠═f9e7b5b3-42e9-41dc-af88-a66324baba59
# ╠═f9c9dd2a-98de-4e54-a055-6400f20f6015
# ╠═703dc739-beab-45ef-a68f-770f47144b60
# ╠═7324efa9-21b3-44f5-bd2d-86f5dd81b55f
# ╠═56f93ecf-e877-422c-8553-5474e9023c77
# ╠═004af453-2575-41ed-a00c-ae4fe83e125b
# ╠═6cf66c0d-ea7f-4693-834b-7eb08c6d22ac
# ╠═1692b2dc-0618-4529-b8c0-357e3a88b977
# ╠═a1143951-7387-40fd-b28c-1d7b31af5b1b
# ╠═149547ad-f8a4-4fca-a4b4-6cf095d0153c
# ╠═47672489-5c37-415e-a93e-021a9595d413
# ╠═c3b6e192-26ac-455d-8aba-735ebe8573b6
# ╠═e41e8570-ab5b-4237-9464-69099f538d26
# ╠═51cf389e-87fd-4670-a599-f3a4964f9470
# ╠═84b22eae-7fde-42e4-aa39-0b59e77d1f4d
# ╠═aabf41fc-4069-4bc6-af5a-20845d22f66a
# ╠═bde3cd11-ede4-409f-959b-a26267e51408
# ╠═f2251c4d-9164-4b71-8a36-bdbbc6d16e0f
# ╠═94243ca7-eb43-49c2-9ddf-339e59d015fd
# ╠═c129498c-558b-4b03-b69e-e4e7b91f42a4
# ╠═3c3eb078-36da-4398-9c3d-b3c1cbdb253d
# ╠═7ea525cd-dda1-4d0c-bdf9-996dde28f4ee
# ╠═9d93f74d-d7eb-4661-9db4-b37a99bb304c
# ╠═649a66fb-d4ce-4edc-bdfb-451aa7e6ce6b
# ╠═e096de8c-210c-4422-9896-83ccf9d36d93
# ╠═3f6af6ed-d3ad-47a5-8e57-d4358211e188
# ╠═0a046ed7-281c-47b3-aa55-66bf7e6f4d32
# ╠═0f127df4-5fa5-4dc2-acbc-6ab7c00a8de8
# ╠═25fc0377-b52b-4588-a39f-f7970aad20a2
# ╠═37aa671c-eb79-41cd-b704-0370065f368b
# ╠═24e1815d-f87b-4bb2-af2a-2b9ae70c1b50
# ╠═a051c3bf-ebee-47fb-b27d-85c2ab8d63c8
# ╠═315c0697-07c9-4ef8-a3d7-4694349cc7cb
# ╠═e27af890-3076-4bd6-b4de-27f50f660e6c
# ╠═fabd3af9-4983-4ff9-8d58-dede2728f182
# ╠═3453a770-56d5-42fb-94fc-2b5b906afc7d
# ╠═52b8362b-e828-44b5-83ba-61b9cfbc8cc6
