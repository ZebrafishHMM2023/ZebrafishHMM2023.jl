### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ b5cd5128-01f9-4d63-a757-1ee9882a0f54
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ d53d040f-5998-40cc-9b7f-69da402f1cce
using DataFrames: DataFrame

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

	# we will subsample using bout time intervals from behavioral data (ARTR data time unit = 0.2s)
	subsampling_times = cumsum(rand([bout.t for traj = behavior_data_full for bout = traj], 10000)) / 0.2
	subsampling_indices = round.(Int, subsampling_times)

	# sample a state sequence
	sampled_state_seq = rand(hmm, maximum(subsampling_indices))
	subsampled_state_seq = sampled_state_seq.state_seq[subsampling_indices]

	return subsampled_state_seq
end

# ╔═╡ 1e539f83-4dd0-4ff3-a71f-471e4cfb2c1e
function compute_transition_matrix_from_state_seq(state_seq::AbstractVector{Int}; nstates::Int=3)
	# count transitions
	transition_counts = [
		count([state_seq[i] == s_prev && state_seq[i+1] == s_next for i = 1:length(state_seq) - 1]) 
		for s_next = 1:nstates, s_prev = 1:nstates
	]
	
	# normalize
	return transition_counts ./ sum(transition_counts; dims=1)
end

# ╔═╡ 35c75e6e-ea0d-446c-b0c3-00b88574876a
md"# Analysis"

# ╔═╡ 3c2d6501-c223-43db-9939-b92b67e575c7
temperature = 18

# ╔═╡ b125dd99-3d7a-4b0b-9bee-1c0903917c35
fish = 12

# ╔═╡ cc158fa3-cf45-4045-bfc1-bcc5dfc9c716
df = let df = DataFrame()

	for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(temperature)
	    data = load_artr_wolf_2023(; temperature, fish)
	    trajs = collect(eachcol(vcat(data.left, data.right)))
	
	    time_unit = mean(diff(data.time)) # convert time bins to seconds
	
	    for nstates = 1:5
	        println("temperature=$temperature, fish=$fish, nstates=$nstates ...")
	        hmm = load_hmm(
				"/data/cossio/projects/2023/zebrafish_hmm/Zebrafish_HMM/2024-02-20 ARTR/Saved HMMs/ARTR HMM $nstates states temperature=$(temperature), fish=$(fish).hd5", HMM_ARTR_Log
			)
	        ll = logdensityof(hmm, trajs) / (data.time[end] - data.time[begin])
	        push!(df, (; temperature, fish, nstates, ll))
	    end
	end
	df
end

# ╔═╡ d92b05c1-9469-4cd7-84de-ad63ef7bc070
for fish = 

# ╔═╡ 6c26d5dc-e8e6-4830-b516-b999a308685e
subsampled_state_seq = sample_behavioral_states_from_artr(; temperature, fish)

# ╔═╡ d40b21e3-3a24-4b72-8638-8d42d05429fd
function split_by_repeats(state_seq::AbstractVector{Int})
	return [state_seq[idx] for idx = ZebrafishHMM2023.find_repeats(subsampled_state_seq)]
end

# ╔═╡ 0b536ae7-b231-4718-a675-25804d9d30e3
function compute_sojourns_from_state_seq(state_seq::AbstractVector{Int}; nstates::Int=3)
	chunks = split_by_repeats(state_seq)
	sojourns = Dict(n => Int[] for n = 1:nstates)
	for (n, t) = zip(map(first, chunks), map(length, chunks))
		push!(sojourns[n], t)
	end
	return sojourns
end

# ╔═╡ 01115683-f37c-4b04-b233-9b91d9c0251f
compute_transition_matrix_from_state_seq(subsampled_state_seq)

# ╔═╡ a43ec3e5-34c7-4537-8bff-0b329c6dd139
compute_sojourns_from_state_seq(subsampled_state_seq)

# ╔═╡ Cell order:
# ╠═e5b56118-03ab-11ef-039b-9bbceb9fbd96
# ╠═b5cd5128-01f9-4d63-a757-1ee9882a0f54
# ╠═7628ddb6-420b-4b9c-98b6-73e3e5cfcac1
# ╠═911e5e73-ec3d-43e9-a807-0cd6c25f5d00
# ╠═d53d040f-5998-40cc-9b7f-69da402f1cce
# ╠═d41ec7d3-479b-4509-b9a2-659836462f10
# ╠═33afc340-77f5-4090-9e4b-b60d6abd0035
# ╠═954538b5-3eb9-4cba-842f-f729537818d5
# ╠═20fc0aff-0a2b-4dd9-98cf-d09ca514b81c
# ╠═d9b31996-0b41-407e-b017-f41a662f4bf8
# ╠═d123ee64-6d14-4d16-9336-a24c0327530c
# ╠═1e539f83-4dd0-4ff3-a71f-471e4cfb2c1e
# ╠═d40b21e3-3a24-4b72-8638-8d42d05429fd
# ╠═0b536ae7-b231-4718-a675-25804d9d30e3
# ╠═35c75e6e-ea0d-446c-b0c3-00b88574876a
# ╠═3c2d6501-c223-43db-9939-b92b67e575c7
# ╠═b125dd99-3d7a-4b0b-9bee-1c0903917c35
# ╠═cc158fa3-cf45-4045-bfc1-bcc5dfc9c716
# ╠═d92b05c1-9469-4cd7-84de-ad63ef7bc070
# ╠═6c26d5dc-e8e6-4830-b516-b999a308685e
# ╠═01115683-f37c-4b04-b233-9b91d9c0251f
# ╠═a43ec3e5-34c7-4537-8bff-0b329c6dd139
