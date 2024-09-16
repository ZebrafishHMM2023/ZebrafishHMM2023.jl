### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 66a6d5b1-729f-49ea-8404-9c0ce48d9973
import Pkg, Revise; Pkg.activate(Base.current_project())

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

HMM trained on ALL trajectories together
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

# ╔═╡ b4cb7f2f-3a52-4371-ba97-a81c00ed6c38
md"# Chunks"

# ╔═╡ c8fc8255-f68f-4bac-85e2-1fd00615181e
n_chunks = 10

# ╔═╡ faf3d121-ca0d-4630-8f96-8ac042a961fb
length(trajs[1])

# ╔═╡ dd6d42aa-9723-4d14-a109-730a29f67210
for (n, fish) = enumerate(trajs)
	for (t, chunk) = enumerate(ZebrafishHMM2023.chunks(fish, n_chunks))
		@show chunk
		@show length(chunk)
		break
	end
	break
end

# ╔═╡ 93f887d2-8913-42c4-adf0-d751b9f85eb9
function train_chunk_hmms(n_chunks::Int; fictitious::Bool = false)
	chunk_hmms = Vector{Vector{ZebrafishHMM2023.ZebrafishHMM_G4_Sym}}()
	chunk_lLs = Vector{Vector{Vector{Float64}}}()
	for (n, fish) = enumerate(trajs)
	    push!(chunk_hmms, ZebrafishHMM2023.ZebrafishHMM_G4_Sym[])
	    push!(chunk_lLs, Vector{Float64}[])
	    for (t, chunk) = enumerate(ZebrafishHMM2023.chunks(fish, n_chunks))
	        println("Fitting fish $n of 18, chunk $t (of length $(length(chunk))).")
	        hmm = ZebrafishHMM2023.ZebrafishHMM_G4_Sym(
	            rand(),
	            rand(4,4),
	            1.0,
	            Distributions.Gamma(0.5, 15.0)
	        )
	        ZebrafishHMM2023.normalize_all!(hmm)

			# if fictitious
			# 	my_data = ...
			# else
			# 	my_data = chunk
			# end

			my_data = chunk
			@show length(my_data)
			
	        (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, my_data, length(my_data); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	        ZebrafishHMM2023.FL_FR_canon!(hmm)
	        push!(chunk_hmms[n], hmm)
	        push!(chunk_lLs[n], lL)
	    end
	end
	return chunk_hmms, chunk_lLs
end

# ╔═╡ 159663b0-a16a-4653-9d7e-6bae6ea3aabf
chunk_hmms, chunk_lLs = train_chunk_hmms(n_chunks)

# ╔═╡ 29849ffc-c262-4325-b918-acb52a6bbc0a
[t for fish = trajs for t = fish]

# ╔═╡ 791b67c2-e826-47e4-9bcb-bbe3f736094c
glob_hmm

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
# ╠═89f60f4e-be24-4531-b80a-7789bbf26aeb
# ╠═56a17d8b-d9d0-4c3d-bd9f-97ac614074d9
# ╠═19aed50e-284d-42d3-ad8c-8aa382bb6e76
# ╠═b7b955d0-023d-43a0-9ad5-5b02a61ff266
# ╠═1824f4e3-fe84-4263-8bea-33e185121055
# ╠═1737bae3-7527-47c5-81de-c841dd3dacad
# ╠═0244c15e-1d24-4b40-adb6-f71c3466f65d
# ╠═b45a1679-4967-4ce0-88e5-d9e40eee0fa2
# ╠═b4cb7f2f-3a52-4371-ba97-a81c00ed6c38
# ╠═c8fc8255-f68f-4bac-85e2-1fd00615181e
# ╠═faf3d121-ca0d-4630-8f96-8ac042a961fb
# ╠═dd6d42aa-9723-4d14-a109-730a29f67210
# ╠═93f887d2-8913-42c4-adf0-d751b9f85eb9
# ╠═159663b0-a16a-4653-9d7e-6bae6ea3aabf
# ╠═29849ffc-c262-4325-b918-acb52a6bbc0a
# ╠═791b67c2-e826-47e4-9bcb-bbe3f736094c
