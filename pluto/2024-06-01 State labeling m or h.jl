### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 167996be-274e-44b7-8cb0-2dc6bebaa57c
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ a36894e5-b139-4d75-8856-f868be1b329a
using Statistics: mean

# ╔═╡ 8e80e780-817d-4486-b74d-f00b7dec33be
using Statistics: cov

# ╔═╡ 8eb7749e-4db6-4189-951d-b324448d36a7
using Statistics: cor

# ╔═╡ af3dfce8-3f5e-428b-8c96-e81f19d21124
using Statistics: middle

# ╔═╡ a53d0019-aa21-4f19-8a33-73ba3649d772
using LinearAlgebra: eigen

# ╔═╡ faeccd57-0ce0-4e4b-ba16-3a4071ddc7c5
using HiddenMarkovModels: logdensityof

# ╔═╡ c66a9f60-7c05-4cdc-a792-44b846164dcc
using HiddenMarkovModels: initial_distribution

# ╔═╡ e93f60af-7eb8-4066-99a1-98cbb97f91c6
using HiddenMarkovModels: viterbi

# ╔═╡ 6917f424-2e35-4008-ba89-bd6d71069a14
using Makie: @L_str

# ╔═╡ 5520e6cd-73ec-466f-b3c6-62a5179d2d7c
using ZebrafishHMM2023: load_artr_wolf_2023

# ╔═╡ f92e68c8-3ddf-45ef-993f-c79673139b01
using ZebrafishHMM2023: HMM_ARTR_Log

# ╔═╡ dc071769-8251-4a4e-b6ac-1a895a2a2e25
using ZebrafishHMM2023: artr_wolf_2023_temperatures

# ╔═╡ 33eb7e7c-4f01-4fc6-b67e-7c1d18adaba0
using ZebrafishHMM2023: artr_wolf_2023_fishes

# ╔═╡ 8a6d98c1-1a96-4be9-a26c-f67b342abad0
using ZebrafishHMM2023: easy_train_artr_hmm

# ╔═╡ c596cdfa-5914-42ef-97d0-31379c583652
using LogExpFunctions: logistic

# ╔═╡ ea9dbc64-2018-11ef-2925-cb2372b98475
md"# Imports"

# ╔═╡ 8ebdad26-7d70-40a2-adf2-40c2909f9bbb
import ZebrafishHMM2023

# ╔═╡ e0846124-c85c-424b-8628-4e607b2eeed0
import Makie

# ╔═╡ 8e9e2b9b-1785-4bb1-b757-c5fa27d1d96d
import CairoMakie

# ╔═╡ ab1ea645-1004-4bd0-9cc1-6612ea8dddd0
md"# Train all HMMs"

# ╔═╡ 6325481f-0906-44a8-8d0f-f0f4df9af101
hmms = Dict((; temperature, fish) => easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature))

# ╔═╡ c14c376f-9b89-4787-b02f-98bbf05b1a32
function ordered_activity_matrix(; temperature, fish)
	data = load_artr_wolf_2023(; temperature, fish)
	hmm = hmms[(; temperature, fish)].hmm

	Nleft = size(data.left, 1)
	Nright = size(data.right, 2)

	_states_perm = sortperm([mean(logistic.(hmm.h[1:Nleft, s])) for s = 1:3])

	#trajs = collect(eachcol(vcat(data.left, data.right)))
	#states = viterbi(hmm, trajs)
	#Δm_per_state = [mean(Δm[states .== s]) for s = 1:nstates]
	
	fullmat = vcat(data.left, data.right)
	mat = reduce(hcat, fullmat[:, states .== s] for s = _states_perm)
	return (; mat, Nleft, Nright)
end

# ╔═╡ 026c05f2-dcb4-435a-ba72-d1c1a9a53b76
activity_matrices = Dict((; temperature, fish) => ordered_activity_matrix(; temperature, fish) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature))

# ╔═╡ 58b1111b-3ae9-4708-a1b7-3bb6c7b55ed1
let fig = Makie.Figure()
	_idx = 0
	for (row, temperature) = enumerate(artr_wolf_2023_temperatures()), (col, fish) = enumerate(artr_wolf_2023_fishes(temperature))
		# _idx += 1
		# row, col = fldmod1(_idx, 4)

		# _min, _max = extrema(activity_matrices[(; temperature, fish)].mat)
		# if !(_min < 0 < _max)
		# 	@info "Failed for T=$temperature, fish=$fish. Got _min=$_min, _max=$_max"
		# else

		# 	_mat = zero(activity_matrices[(; temperature, fish)].mat)
		# 	for i = eachindex(_mat)
		# 		x = activity_matrices[(; temperature, fish)].mat[i]
		# 		if x ≥ 0
		# 			_mat[i] = x ./ abs(_max)
		# 		else
		# 			_mat[i] = x ./ abs(_min)
		# 		end
		# 	end
			
		# end
		
		ax = Makie.Axis(fig[row, col], width=200, height=200, xlabel="frame", ylabel="neuron", title="T=$temperature, fish=$fish", xticks=0:3000:20000, yticks=0:100:1000)
		Makie.image!(ax, activity_matrices[(; temperature, fish)].mat'; colormap=:bam, colorrange=(0, 1))
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ c2949166-eb86-4ed2-b300-b9bf34753503
extrema(randn(10, 4))

# ╔═╡ Cell order:
# ╠═ea9dbc64-2018-11ef-2925-cb2372b98475
# ╠═167996be-274e-44b7-8cb0-2dc6bebaa57c
# ╠═8ebdad26-7d70-40a2-adf2-40c2909f9bbb
# ╠═e0846124-c85c-424b-8628-4e607b2eeed0
# ╠═8e9e2b9b-1785-4bb1-b757-c5fa27d1d96d
# ╠═a36894e5-b139-4d75-8856-f868be1b329a
# ╠═8e80e780-817d-4486-b74d-f00b7dec33be
# ╠═8eb7749e-4db6-4189-951d-b324448d36a7
# ╠═af3dfce8-3f5e-428b-8c96-e81f19d21124
# ╠═a53d0019-aa21-4f19-8a33-73ba3649d772
# ╠═faeccd57-0ce0-4e4b-ba16-3a4071ddc7c5
# ╠═c66a9f60-7c05-4cdc-a792-44b846164dcc
# ╠═e93f60af-7eb8-4066-99a1-98cbb97f91c6
# ╠═6917f424-2e35-4008-ba89-bd6d71069a14
# ╠═5520e6cd-73ec-466f-b3c6-62a5179d2d7c
# ╠═f92e68c8-3ddf-45ef-993f-c79673139b01
# ╠═dc071769-8251-4a4e-b6ac-1a895a2a2e25
# ╠═33eb7e7c-4f01-4fc6-b67e-7c1d18adaba0
# ╠═8a6d98c1-1a96-4be9-a26c-f67b342abad0
# ╠═c596cdfa-5914-42ef-97d0-31379c583652
# ╠═ab1ea645-1004-4bd0-9cc1-6612ea8dddd0
# ╠═6325481f-0906-44a8-8d0f-f0f4df9af101
# ╠═c14c376f-9b89-4787-b02f-98bbf05b1a32
# ╠═026c05f2-dcb4-435a-ba72-d1c1a9a53b76
# ╠═58b1111b-3ae9-4708-a1b7-3bb6c7b55ed1
# ╠═c2949166-eb86-4ed2-b300-b9bf34753503
