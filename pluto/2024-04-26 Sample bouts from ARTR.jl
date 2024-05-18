### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ b5cd5128-01f9-4d63-a757-1ee9882a0f54
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ d41ec7d3-479b-4509-b9a2-659836462f10
using Statistics: mean, cov, cor

# ╔═╡ 33afc340-77f5-4090-9e4b-b60d6abd0035
using LinearAlgebra: eigen

# ╔═╡ 954538b5-3eb9-4cba-842f-f729537818d5
using HiddenMarkovModels: logdensityof, initial_distribution, viterbi

# ╔═╡ 20fc0aff-0a2b-4dd9-98cf-d09ca514b81c
using ZebrafishHMM2023: load_artr_wolf_2023, HMM_ARTR, HMM_ARTR_Log,
    artr_wolf_2023_temperatures, artr_wolf_2023_fishes, split_into_repeated_subsequences, easy_train_artr_hmm

# ╔═╡ e5b56118-03ab-11ef-039b-9bbceb9fbd96
md"# Imports"

# ╔═╡ 7628ddb6-420b-4b9c-98b6-73e3e5cfcac1
import ZebrafishHMM2023

# ╔═╡ 911e5e73-ec3d-43e9-a807-0cd6c25f5d00
import Makie, CairoMakie

# ╔═╡ d9b31996-0b41-407e-b017-f41a662f4bf8
md"# Functions"

# ╔═╡ d123ee64-6d14-4d16-9336-a24c0327530c
function sample_behavioral_states_from_artr(; temperature::Int, fish::Int)
	# Load behavior data at a given temperature
	behavior_data_full = ZebrafishHMM2023.load_full_obs(temperature)

	# load ARTR data
	data = load_artr_wolf_2023(; temperature, fish)

	# numbers of neurons in left and right ARTR regions
	Nleft, Nright = size(data.left, 1), size(data.right, 1)
	Nneurons = Nleft + Nright # and total number of neurons

	# neural concatenated trajectories
	trajs = collect(eachcol(vcat(data.left, data.right)))

	# neural HMM
	hmm, lL = easy_train_artr_hmm(; temperature, fish) # ARTR data time unit = 0.2s

	# sample a state sequence
	sampled_state_seq = rand(hmm, 1000000)

	# subsample using bout time intervals from behavioral data (ARTR data time unit = 0.2s)
	subsampling_times = cumsum(rand([bout.t for traj = behavior_data_full for bout = traj], 10000)) / 0.2
	subsampled_state_seq = sampled_state_seq.state_seq[round.(Int, subsampling_times)]
end

# ╔═╡ 35c75e6e-ea0d-446c-b0c3-00b88574876a
md"# Analysis"

# ╔═╡ 3c2d6501-c223-43db-9939-b92b67e575c7
temperature = 18

# ╔═╡ b125dd99-3d7a-4b0b-9bee-1c0903917c35
fish = 12

# ╔═╡ 25dd03f2-5f5e-4f74-b571-c61d77c2292e
# Load behavior data at a given temperature
behavior_data_full = ZebrafishHMM2023.load_full_obs(temperature)

# ╔═╡ 953ed213-690e-435e-b32b-8ad4d067ab84
data = load_artr_wolf_2023(; temperature, fish)

# ╔═╡ ef1c1633-636e-466c-96bf-b3cc34bc45dc
Nleft, Nright = size(data.left, 1), size(data.right, 1)

# ╔═╡ 0874055d-f810-45f2-874c-fe075e7702b7
Nneurons = Nleft + Nright;

# ╔═╡ 66409bff-4a5b-4fef-a945-42c6e1964b9b
trajs = collect(eachcol(vcat(data.left, data.right)))

# ╔═╡ 0e0e953b-8323-4007-9be7-6355e476502a
hmm, lL = easy_train_artr_hmm(; temperature, fish) # time unit = 0.2s

# ╔═╡ e6c88f1d-d714-478d-b057-fd7042f1075e
sampled_state_seq = rand(hmm, 1000000)

# ╔═╡ 2282fa05-2238-48f4-9425-24d00a287d3b
# time unit = 0.2s
subsampling_times = cumsum(rand([bout.t for traj = behavior_data_full for bout = traj], 10000)) / 0.2

# ╔═╡ 6c26d5dc-e8e6-4830-b516-b999a308685e
subsampled_state_seq = sampled_state_seq.state_seq[round.(Int, subsampling_times)]

# ╔═╡ 1e7e0d4a-3ac6-4fab-974a-da9cd8d8a36a
subsampled_state_seq_split_by_repeats = [subsampled_state_seq[idx] for idx = ZebrafishHMM2023.find_repeats(subsampled_state_seq)]

# ╔═╡ 440bd454-425d-4488-8042-ec20165abc12
subsampled_transition_counts = [length([
	i
	 for i = 1:length(subsampled_state_seq) - 1
	 if first(subsampled_state_seq[i]) == s_prev && 
	 first(subsampled_state_seq[i+1]) == s_next
]) for s_next = 1:3, s_prev = 1:3]

# ╔═╡ dc67d6df-f7d8-4785-8a60-3d69eaf02e7a
subsampled_transition_counts ./ sum(subsampled_transition_counts; dims=1)

# ╔═╡ bc3d6dff-6b4d-42b5-bcc9-2d2cbd4676d7
[length(rep) for rep = subsampled_state_seq_split_by_repeats if first(rep) == 1]

# ╔═╡ e358f599-5ce7-4c83-b029-109efaf8a4a6
sample_behavioral_states_from_artr(; temperature, fish)

# ╔═╡ Cell order:
# ╠═e5b56118-03ab-11ef-039b-9bbceb9fbd96
# ╠═b5cd5128-01f9-4d63-a757-1ee9882a0f54
# ╠═7628ddb6-420b-4b9c-98b6-73e3e5cfcac1
# ╠═911e5e73-ec3d-43e9-a807-0cd6c25f5d00
# ╠═d41ec7d3-479b-4509-b9a2-659836462f10
# ╠═33afc340-77f5-4090-9e4b-b60d6abd0035
# ╠═954538b5-3eb9-4cba-842f-f729537818d5
# ╠═20fc0aff-0a2b-4dd9-98cf-d09ca514b81c
# ╠═d9b31996-0b41-407e-b017-f41a662f4bf8
# ╠═d123ee64-6d14-4d16-9336-a24c0327530c
# ╠═35c75e6e-ea0d-446c-b0c3-00b88574876a
# ╠═3c2d6501-c223-43db-9939-b92b67e575c7
# ╠═b125dd99-3d7a-4b0b-9bee-1c0903917c35
# ╠═25dd03f2-5f5e-4f74-b571-c61d77c2292e
# ╠═953ed213-690e-435e-b32b-8ad4d067ab84
# ╠═ef1c1633-636e-466c-96bf-b3cc34bc45dc
# ╠═0874055d-f810-45f2-874c-fe075e7702b7
# ╠═66409bff-4a5b-4fef-a945-42c6e1964b9b
# ╠═0e0e953b-8323-4007-9be7-6355e476502a
# ╠═e6c88f1d-d714-478d-b057-fd7042f1075e
# ╠═2282fa05-2238-48f4-9425-24d00a287d3b
# ╠═6c26d5dc-e8e6-4830-b516-b999a308685e
# ╠═1e7e0d4a-3ac6-4fab-974a-da9cd8d8a36a
# ╠═440bd454-425d-4488-8042-ec20165abc12
# ╠═dc67d6df-f7d8-4785-8a60-3d69eaf02e7a
# ╠═bc3d6dff-6b4d-42b5-bcc9-2d2cbd4676d7
# ╠═e358f599-5ce7-4c83-b029-109efaf8a4a6
