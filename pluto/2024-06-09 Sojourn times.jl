### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ bfb8b4b5-eee8-47a5-8f55-e23ad3ddbaa9
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 2bcdddad-4227-4c56-b452-cff43bec2d41
using Makie: @L_str

# ╔═╡ 76c368ee-0aa9-4d8e-b815-d006ec9ae3db
using DataFrames: DataFrame

# ╔═╡ 6d94d933-8409-47ec-9e42-9bb0816f0d2f
using Statistics: mean

# ╔═╡ 71ee1b22-360b-44fa-a1d3-d03bdf2b13da
using Statistics: cov

# ╔═╡ 83bf9b96-f048-418a-b712-b240002f7f0f
using Statistics: cor

# ╔═╡ fbed3295-c9d4-4f08-800e-8e795c2599e9
using Statistics: middle

# ╔═╡ a08cd4f3-e666-4b45-980f-ea7805dbbb2d
using LinearAlgebra: eigen

# ╔═╡ 05bfd1b2-43f0-4259-adb8-a378549e9a71
using Distributions: Gamma

# ╔═╡ d3289663-20d4-46a9-9795-4c5fe5cd30b7
using HiddenMarkovModels: baum_welch

# ╔═╡ 7fb964e4-cb6c-4fc9-be84-f5bac9bd3da9
using HiddenMarkovModels: logdensityof

# ╔═╡ 7aeb8ad6-7e66-4178-91dd-329e893ba3a8
using HiddenMarkovModels: initial_distribution

# ╔═╡ 6fdd522d-4521-437c-bc5b-bd7681ed65f5
using HiddenMarkovModels: viterbi

# ╔═╡ c9147cdd-0ea9-439c-9c9a-339143617e96
using ZebrafishHMM2023: load_artr_wolf_2023

# ╔═╡ 92a7342e-3013-4fd5-89a0-3681d930c590
using ZebrafishHMM2023: HMM_ARTR_Log

# ╔═╡ b2ee75a2-2eb8-47c0-8d84-1d322f486259
using ZebrafishHMM2023: artr_wolf_2023_temperatures

# ╔═╡ 15ea5507-5476-4ab2-ab81-0285054a27a9
using ZebrafishHMM2023: artr_wolf_2023_fishes

# ╔═╡ 19b21fd1-8760-4c97-b96c-c79dea14a4d6
using ZebrafishHMM2023: easy_train_artr_hmm

# ╔═╡ 456636bb-5b78-4082-b64a-9a29b888b4da
using ZebrafishHMM2023: behaviour_free_swimming_temperatures

# ╔═╡ 09962b17-ba93-4c3a-a1b9-bca9453c2b5b
using ZebrafishHMM2023: load_behaviour_free_swimming_data

# ╔═╡ c4fffaf7-f7dc-4474-9bd1-47f2d03b21b6
using ZebrafishHMM2023: load_behaviour_free_swimming_trajs

# ╔═╡ df51bac9-f692-416b-98a5-8dfe13853a70
using ZebrafishHMM2023: ZebrafishHMM_G3_Sym

# ╔═╡ e6fcc43f-f118-4282-9b4c-c8a029e72319
using ZebrafishHMM2023: find_repeats

# ╔═╡ 9189b20d-d70f-417d-994d-2fc1488965fa
using ZebrafishHMM2023: normalize_all!

# ╔═╡ 91a45bdf-321d-4302-8de3-338bb10bc4fb
using ZebrafishHMM2023: ATol

# ╔═╡ ae9a7803-3ddf-4fcc-9b15-6b970797b4cb
md"# Imports"

# ╔═╡ 4285c8bf-7a44-4c56-a86e-f15dafd67b4b
import ZebrafishHMM2023

# ╔═╡ d3731804-a255-42a7-bad2-31c834be5d7a
import Makie

# ╔═╡ 5d9d1329-8faf-46f6-aac3-aee7bd9b8065
import CairoMakie

# ╔═╡ 010bfb30-d9d0-4b23-ae9b-063c7f798cc9
md"# Swimming sojourn times"

# ╔═╡ d557e6ea-6794-4b45-b859-b6e4b9361019
function swimming_train_hmm(T)
	@info "Training model for swimming at temperature = $T"
    trajs = load_behaviour_free_swimming_trajs(T)

    #trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes
    hmm = normalize_all!(ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ATol(1e-6))

	seqs = viterbi(hmm, trajs, length(trajs))

	all_times = [filter(!isnan, t) for t = eachcol(load_behaviour_free_swimming_data(T).bouttime)]

	times_F = Float64[]
	times_L = Float64[]
	times_R = Float64[]
	
	for (s, t) = zip(seqs, all_times)
		@assert length(s) + 1 == length(t)
        reps = find_repeats(s)
		for r = reps
			@assert allequal(s[r])
		end

        append!(times_F, [t[last(r) + 1] - t[first(r)] for r = reps if s[first(r)] == 1])
        append!(times_L, [t[last(r) + 1] - t[first(r)] for r = reps if s[first(r)] == 2])
        append!(times_R, [t[last(r) + 1] - t[first(r)] for r = reps if s[first(r)] == 3])
    end

    return (; hmm, lL, times_F, times_L, times_R)
end

# ╔═╡ 4b4f2b29-34c9-4846-b545-5ab103d1e0fc
swimming_hmms = Dict(T => swimming_train_hmm(T) for T = behaviour_free_swimming_temperatures())

# ╔═╡ 277b0cb0-a3de-4fc5-8720-9d5e8846e81f
md"# ARTR HMMs"

# ╔═╡ ebefc691-9bf9-4376-a74e-bf56e5ddb065
function artr_train_hmm(; temperature, fish)
	# Train HMM
	hmm, lL = easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true)

	# load ARTR data
	data = load_artr_wolf_2023(; temperature, fish)

	# numbers of neurons in left and right ARTR regions
	Nleft, Nright = size(data.left, 1), size(data.right, 1)
	Nneurons = Nleft + Nright # and total number of neurons
	
	traj = collect(eachcol(vcat(data.left, data.right)))
	seq = viterbi(hmm, traj)
	reps = find_repeats(seq)
	t = [0.0; data.time]

	times_F = [t[last(r) + 1] - t[first(r)] for r = reps if seq[first(r)] == 1]
	times_L = [t[last(r) + 1] - t[first(r)] for r = reps if seq[first(r)] == 2]
	times_R = [t[last(r) + 1] - t[first(r)] for r = reps if seq[first(r)] == 3]

	return (; hmm, lL, times_F, times_L, times_R)
end

# ╔═╡ c5c12295-e495-480e-899b-22745ad9f4e6
artr_hmms = Dict(
	artr_train_hmm(; temperature, fish)
	for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)
)

# ╔═╡ aa26f84c-49d3-4f14-bc32-e3c3858f4b39
md"# Comparison of sojourn times"

# ╔═╡ a8a8c8aa-3def-4b5c-a7c4-cbb09f0d63ad


# ╔═╡ Cell order:
# ╠═ae9a7803-3ddf-4fcc-9b15-6b970797b4cb
# ╠═bfb8b4b5-eee8-47a5-8f55-e23ad3ddbaa9
# ╠═4285c8bf-7a44-4c56-a86e-f15dafd67b4b
# ╠═d3731804-a255-42a7-bad2-31c834be5d7a
# ╠═5d9d1329-8faf-46f6-aac3-aee7bd9b8065
# ╠═2bcdddad-4227-4c56-b452-cff43bec2d41
# ╠═76c368ee-0aa9-4d8e-b815-d006ec9ae3db
# ╠═6d94d933-8409-47ec-9e42-9bb0816f0d2f
# ╠═71ee1b22-360b-44fa-a1d3-d03bdf2b13da
# ╠═83bf9b96-f048-418a-b712-b240002f7f0f
# ╠═fbed3295-c9d4-4f08-800e-8e795c2599e9
# ╠═a08cd4f3-e666-4b45-980f-ea7805dbbb2d
# ╠═05bfd1b2-43f0-4259-adb8-a378549e9a71
# ╠═d3289663-20d4-46a9-9795-4c5fe5cd30b7
# ╠═7fb964e4-cb6c-4fc9-be84-f5bac9bd3da9
# ╠═7aeb8ad6-7e66-4178-91dd-329e893ba3a8
# ╠═6fdd522d-4521-437c-bc5b-bd7681ed65f5
# ╠═c9147cdd-0ea9-439c-9c9a-339143617e96
# ╠═92a7342e-3013-4fd5-89a0-3681d930c590
# ╠═b2ee75a2-2eb8-47c0-8d84-1d322f486259
# ╠═15ea5507-5476-4ab2-ab81-0285054a27a9
# ╠═19b21fd1-8760-4c97-b96c-c79dea14a4d6
# ╠═456636bb-5b78-4082-b64a-9a29b888b4da
# ╠═09962b17-ba93-4c3a-a1b9-bca9453c2b5b
# ╠═c4fffaf7-f7dc-4474-9bd1-47f2d03b21b6
# ╠═df51bac9-f692-416b-98a5-8dfe13853a70
# ╠═e6fcc43f-f118-4282-9b4c-c8a029e72319
# ╠═9189b20d-d70f-417d-994d-2fc1488965fa
# ╠═91a45bdf-321d-4302-8de3-338bb10bc4fb
# ╠═010bfb30-d9d0-4b23-ae9b-063c7f798cc9
# ╠═d557e6ea-6794-4b45-b859-b6e4b9361019
# ╠═4b4f2b29-34c9-4846-b545-5ab103d1e0fc
# ╠═277b0cb0-a3de-4fc5-8720-9d5e8846e81f
# ╠═ebefc691-9bf9-4376-a74e-bf56e5ddb065
# ╠═c5c12295-e495-480e-899b-22745ad9f4e6
# ╠═aa26f84c-49d3-4f14-bc32-e3c3858f4b39
# ╠═a8a8c8aa-3def-4b5c-a7c4-cbb09f0d63ad
