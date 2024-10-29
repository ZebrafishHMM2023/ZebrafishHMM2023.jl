### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 66a6d5b1-729f-49ea-8404-9c0ce48d9973
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 9674b801-5acf-4f76-b494-4aa8cb8ab012
using Statistics: mean, std

# ╔═╡ dca0d89a-744f-11ef-1dde-71bceb3d228e
md"# Imports"

# ╔═╡ 47c05d6b-4488-4c2e-8efe-a1fc47e41376
import ZebrafishHMM2023

# ╔═╡ c4a6db37-bedb-438c-8d66-96bc42e7d752
import HiddenMarkovModels

# ╔═╡ bd5e2351-8fe0-49c9-8d6e-014560b59563
import Makie

# ╔═╡ 2ecf574a-ed55-4e03-b4e0-3639a3328871
import CairoMakie

# ╔═╡ 99b4612b-04fa-4248-bcef-81ae8ad21cff
import PlutoUI

# ╔═╡ 0f94e06d-c8bf-4204-a5a3-3e67d5374a99
import CSV

# ╔═╡ 13917bd0-67fc-4454-84b1-4f5ff6004b0f
import HDF5

# ╔═╡ 8ab55b4d-5ea8-4e12-959e-21301d76770a
import Distributions

# ╔═╡ 89f60f4e-be24-4531-b80a-7789bbf26aeb
PlutoUI.TableOfContents()

# ╔═╡ 56a17d8b-d9d0-4c3d-bd9f-97ac614074d9
md"# Data"

# ╔═╡ 19aed50e-284d-42d3-ad8c-8aa382bb6e76
trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ b7b955d0-023d-43a0-9ad5-5b02a61ff266
glob_all_trajs = [t for fish = trajs for t = fish]

# ╔═╡ 1824f4e3-fe84-4263-8bea-33e185121055
md"""
# Train global HMM

HMM trained on ALL trajectories (of all fish) together.
"""

# ╔═╡ 1737bae3-7527-47c5-81de-c841dd3dacad
glob_hmm_init = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(
    rand(),
    rand(3,3),
    1.0,
    Distributions.Gamma(1.1, 15.0),
    1.0
)

# ╔═╡ 0244c15e-1d24-4b40-adb6-f71c3466f65d
# HMM trained on ALL trajectories together
begin
	(glob_hmm, glob_lLs) = HiddenMarkovModels.baum_welch(glob_hmm_init, glob_all_trajs, length(glob_all_trajs); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	ZebrafishHMM2023.normalize_all!(glob_hmm)
end

# ╔═╡ b45a1679-4967-4ce0-88e5-d9e40eee0fa2
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=900, height=300)
	Makie.lines!(ax, glob_lLs)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 9b385b4f-f087-499d-bce5-650962be31ff
md"""
# Train HMM on all trajectories of each fish
"""

# ╔═╡ 06c159a2-c59e-40fa-b2ba-24d974310299
begin
	global_hmms = ZebrafishHMM2023.ZebrafishHMM_G4_Sym[]
	global_lLs = Vector{Float64}[]
	for (n, fish) = enumerate(trajs)
	    println("Fitting fish $n of 18.")
	    hmm = ZebrafishHMM2023.ZebrafishHMM_G4_Sym(
	        rand(),
	        rand(4,4),
	        1.0,
	        Distributions.Gamma(0.5, 15.0)
	    )
	    ZebrafishHMM2023.normalize_all!(hmm)
	    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish, length(fish); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	    ZebrafishHMM2023.FL_FR_canon!(hmm)
	    push!(global_hmms, hmm)
	    push!(global_lLs, lL)
	end
end

# ╔═╡ b4cb7f2f-3a52-4371-ba97-a81c00ed6c38
md"# Chunks"

# ╔═╡ c8fc8255-f68f-4bac-85e2-1fd00615181e
n_chunks = 10

# ╔═╡ 93f887d2-8913-42c4-adf0-d751b9f85eb9
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

# ╔═╡ 159663b0-a16a-4653-9d7e-6bae6ea3aabf
chunk_hmms, chunk_lLs = train_chunk_hmms(n_chunks; fictitious=true)

# ╔═╡ 791b67c2-e826-47e4-9bcb-bbe3f736094c
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

# ╔═╡ Cell order:
# ╠═dca0d89a-744f-11ef-1dde-71bceb3d228e
# ╠═66a6d5b1-729f-49ea-8404-9c0ce48d9973
# ╠═47c05d6b-4488-4c2e-8efe-a1fc47e41376
# ╠═c4a6db37-bedb-438c-8d66-96bc42e7d752
# ╠═bd5e2351-8fe0-49c9-8d6e-014560b59563
# ╠═2ecf574a-ed55-4e03-b4e0-3639a3328871
# ╠═99b4612b-04fa-4248-bcef-81ae8ad21cff
# ╠═0f94e06d-c8bf-4204-a5a3-3e67d5374a99
# ╠═13917bd0-67fc-4454-84b1-4f5ff6004b0f
# ╠═8ab55b4d-5ea8-4e12-959e-21301d76770a
# ╠═9674b801-5acf-4f76-b494-4aa8cb8ab012
# ╠═89f60f4e-be24-4531-b80a-7789bbf26aeb
# ╠═56a17d8b-d9d0-4c3d-bd9f-97ac614074d9
# ╠═19aed50e-284d-42d3-ad8c-8aa382bb6e76
# ╠═b7b955d0-023d-43a0-9ad5-5b02a61ff266
# ╠═1824f4e3-fe84-4263-8bea-33e185121055
# ╠═1737bae3-7527-47c5-81de-c841dd3dacad
# ╠═0244c15e-1d24-4b40-adb6-f71c3466f65d
# ╠═b45a1679-4967-4ce0-88e5-d9e40eee0fa2
# ╠═9b385b4f-f087-499d-bce5-650962be31ff
# ╠═06c159a2-c59e-40fa-b2ba-24d974310299
# ╠═b4cb7f2f-3a52-4371-ba97-a81c00ed6c38
# ╠═c8fc8255-f68f-4bac-85e2-1fd00615181e
# ╠═93f887d2-8913-42c4-adf0-d751b9f85eb9
# ╠═159663b0-a16a-4653-9d7e-6bae6ea3aabf
# ╠═791b67c2-e826-47e4-9bcb-bbe3f736094c
