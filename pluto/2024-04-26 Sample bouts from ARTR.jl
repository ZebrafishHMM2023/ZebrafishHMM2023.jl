### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ b5cd5128-01f9-4d63-a757-1ee9882a0f54
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 16867f62-2fd0-4e52-80c9-5962e615ee98
using Makie: @L_str

# ╔═╡ d53d040f-5998-40cc-9b7f-69da402f1cce
using DataFrames: DataFrame

# ╔═╡ 92e5f7e3-3f4d-4f78-92c1-c701ea663b38
using Statistics: mean

# ╔═╡ 923e7940-c88b-471f-8454-454629ae632f
using Statistics: cov

# ╔═╡ 9ca878bb-5f40-4c25-aa5c-fba350b37e76
using Statistics: cor

# ╔═╡ 955d0b51-d267-4f9d-a773-db841daece52
using Statistics: middle

# ╔═╡ 33afc340-77f5-4090-9e4b-b60d6abd0035
using LinearAlgebra: eigen

# ╔═╡ 16a74c00-d4f6-4b6c-a7bf-6096333ccb9d
using Distributions: Gamma

# ╔═╡ 9c5c3dc2-154b-4893-9292-98c945e7ed86
using HiddenMarkovModels: baum_welch

# ╔═╡ 3eb94a23-113e-4c2b-a966-452b33caadf1
using HiddenMarkovModels: logdensityof

# ╔═╡ 00ebdd2d-d7a6-444a-a548-285b228af912
using HiddenMarkovModels: initial_distribution

# ╔═╡ 0a4ea7ef-7372-4c6b-ad74-2e18db46694e
using ZebrafishHMM2023: load_artr_wolf_2023

# ╔═╡ 2ed3be3a-f935-4a28-95e1-ab3f209cb494
using ZebrafishHMM2023: HMM_ARTR_Log

# ╔═╡ a6ff3c4e-f9d5-416b-84e8-f02ccda113dd
using ZebrafishHMM2023: artr_wolf_2023_temperatures

# ╔═╡ 787ca473-b238-499f-a3b2-317680341b3d
using ZebrafishHMM2023: artr_wolf_2023_fishes

# ╔═╡ 1d8a6197-f25c-4c4f-aea7-87f261bf892a
using ZebrafishHMM2023: easy_train_artr_hmm

# ╔═╡ ae32a09d-4b39-456a-9b17-b72b14da2457
using ZebrafishHMM2023: behaviour_free_swimming_temperatures

# ╔═╡ 2ade4518-1ef4-4f59-a8c8-a90531ee421e
using ZebrafishHMM2023: load_behaviour_free_swimming_data

# ╔═╡ 2b0d13b1-cff1-4183-ad08-79c83e9cc2a5
using ZebrafishHMM2023: load_behaviour_free_swimming_trajs

# ╔═╡ ebcfb5a4-2aa7-477c-a911-15cf60c6a35e
using ZebrafishHMM2023: ZebrafishHMM_G3_Sym

# ╔═╡ ff056014-160a-4228-9a1e-bfb0c85653e9
using ZebrafishHMM2023: find_repeats

# ╔═╡ 582c74a3-0ec8-4706-99dd-410fa0bd2a5a
using ZebrafishHMM2023: normalize_all!

# ╔═╡ f394a010-23ae-43fb-8cab-c0b9228c37cd
using ZebrafishHMM2023: ATol

# ╔═╡ e5b56118-03ab-11ef-039b-9bbceb9fbd96
md"# Imports"

# ╔═╡ 7628ddb6-420b-4b9c-98b6-73e3e5cfcac1
import ZebrafishHMM2023

# ╔═╡ 259f6749-37a8-45c3-a32d-fa678cb81fe5
import Makie

# ╔═╡ d2dbc9b0-1ace-4c90-8706-4fc28fbe3a0c
import CairoMakie

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
	hmm, lL = easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true) # ARTR data time unit = 0.2s

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

# ╔═╡ d40b21e3-3a24-4b72-8638-8d42d05429fd
function split_by_repeats(state_seq::AbstractVector{Int})
	return [state_seq[idx] for idx = ZebrafishHMM2023.find_repeats(state_seq)]
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

# ╔═╡ 35c75e6e-ea0d-446c-b0c3-00b88574876a
md"# ARTR resampled sojourn times (in bouts)"

# ╔═╡ cc158fa3-cf45-4045-bfc1-bcc5dfc9c716
df = let df = DataFrame()
	for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(temperature)
		subsampled_state_seq = sample_behavioral_states_from_artr(; temperature, fish)

		Ω = compute_transition_matrix_from_state_seq(subsampled_state_seq)
		ts = compute_sojourns_from_state_seq(subsampled_state_seq)

		push!(df, 
			(; 
				temperature, fish,
			
				# transition matrix
				FtoF = Ω[1,1],
				FtoT = middle(Ω[2,1], Ω[3,1]), 
				TtoF = middle(Ω[1,2], Ω[1,3]),
				TtoTsame = middle(Ω[2,2], Ω[3,3]),
				TtoTdiff = middle(Ω[3,2], Ω[2,3]),
			
				# sojourn times
				sojourn_F = ts[1],
				sojourn_T = vcat(ts[2], ts[3])
			)
		)
	end
	df
end

# ╔═╡ 01115683-f37c-4b04-b233-9b91d9c0251f
let fig = Makie.Figure()
	_sz = 200
	
	ax = Makie.Axis(fig[1,1], width=_sz, height=_sz, title=L"$F$ $\rightarrow$ $F$", xticks=collect(artr_wolf_2023_temperatures()), xlabel="temperature", ylabel="transition rate (1/bout)")
	Makie.scatter!(df.temperature, df.FtoF)
	Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.FtoF[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])
	Makie.ylims!(ax, 0, 1)
	
	ax = Makie.Axis(fig[1,2], width=_sz, height=_sz, title=L"$F$ $\rightarrow$ $L$ or $R$", xticks=collect(artr_wolf_2023_temperatures()), xlabel="temperature", ylabel="transition rate (1/bout)")
	Makie.scatter!(df.temperature, df.FtoT)
	Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.FtoT[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])
	Makie.ylims!(ax, 0, 1)

	ax = Makie.Axis(fig[1,3], width=_sz, height=_sz, title=L"$L$ or $R$ $\rightarrow$ $F$", xticks=collect(artr_wolf_2023_temperatures()), xlabel="temperature", ylabel="transition rate (1/bout)")
	Makie.scatter!(df.temperature, df.TtoF)
	Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.TtoF[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])
	Makie.ylims!(ax, 0, 1)

	ax = Makie.Axis(fig[1,4], width=_sz, height=_sz, title=L"$L$ $\rightarrow$ $L$ or $R$ $\rightarrow$ $R$", xticks=collect(artr_wolf_2023_temperatures()), xlabel="temperature", ylabel="transition rate (1/bout)")
	Makie.scatter!(df.temperature, df.TtoTsame)
	Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.TtoTsame[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])
	Makie.ylims!(ax, 0, 1)

	ax = Makie.Axis(fig[1,5], width=_sz, height=_sz, title=L"$L$ $\rightarrow$ $R$ or $R$ $\rightarrow$ $L$", xticks=collect(artr_wolf_2023_temperatures()), xlabel="temperature", ylabel="transition rate (1/bout)")
	Makie.scatter!(df.temperature, df.TtoTdiff)
	Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.TtoTdiff[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])
	Makie.ylims!(ax, 0, 1)

	# ax = Makie.Axis(fig[2,2], width=_sz, height=_sz, title=L"$L$ $\rightarrow$ $R$ or $R$ $\rightarrow$ $L$ / $L$ or $R$ $\rightarrow$ $F$", xticks=collect(artr_wolf_2023_temperatures()))
	# Makie.scatter!(df.temperature, df.TtoTdiff ./ df.FtoT)
	# Makie.lines!(collect(artr_wolf_2023_temperatures()), [mean(df.TtoTdiff[df.temperature .== T]) / mean(df.TtoF[df.temperature .== T]) for T = artr_wolf_2023_temperatures()])

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ daf9a4f4-4ef2-4f65-b27b-2e88956133c5
let fig = Makie.Figure()
	for (n, T) = enumerate(artr_wolf_2023_temperatures())
	    ax = Makie.Axis(fig[1,n], width=200, height=200, xlabel="sojourn (bouts)", ylabel="frequency", yscale=log10, title="$T °C")
	    Makie.stephist!(ax, reduce(vcat, df.sojourn_F[df.temperature .== T]), normalization=:pdf, bins=0:5:50, color=:black, label="F")
	    Makie.stephist!(ax, reduce(vcat, df.sojourn_T[df.temperature .== T]), normalization=:pdf, bins=0:5:50, linewidth=2, color=:blue, label="L/R", linestyle=:dash)
	    Makie.ylims!(ax, 1e-5, 1)
	    if n == length(artr_wolf_2023_temperatures())
			Makie.axislegend(ax)
		end
	end
	Makie.resize_to_layout!(fig)
	# Makie.save("Sel. Figures/2024-02-22 ARTR sojourn_times.pdf", fig)
	fig
end

# ╔═╡ Cell order:
# ╠═e5b56118-03ab-11ef-039b-9bbceb9fbd96
# ╠═b5cd5128-01f9-4d63-a757-1ee9882a0f54
# ╠═7628ddb6-420b-4b9c-98b6-73e3e5cfcac1
# ╠═259f6749-37a8-45c3-a32d-fa678cb81fe5
# ╠═d2dbc9b0-1ace-4c90-8706-4fc28fbe3a0c
# ╠═16867f62-2fd0-4e52-80c9-5962e615ee98
# ╠═d53d040f-5998-40cc-9b7f-69da402f1cce
# ╠═92e5f7e3-3f4d-4f78-92c1-c701ea663b38
# ╠═923e7940-c88b-471f-8454-454629ae632f
# ╠═9ca878bb-5f40-4c25-aa5c-fba350b37e76
# ╠═955d0b51-d267-4f9d-a773-db841daece52
# ╠═33afc340-77f5-4090-9e4b-b60d6abd0035
# ╠═16a74c00-d4f6-4b6c-a7bf-6096333ccb9d
# ╠═9c5c3dc2-154b-4893-9292-98c945e7ed86
# ╠═3eb94a23-113e-4c2b-a966-452b33caadf1
# ╠═00ebdd2d-d7a6-444a-a548-285b228af912
# ╠═0a4ea7ef-7372-4c6b-ad74-2e18db46694e
# ╠═2ed3be3a-f935-4a28-95e1-ab3f209cb494
# ╠═a6ff3c4e-f9d5-416b-84e8-f02ccda113dd
# ╠═787ca473-b238-499f-a3b2-317680341b3d
# ╠═1d8a6197-f25c-4c4f-aea7-87f261bf892a
# ╠═ae32a09d-4b39-456a-9b17-b72b14da2457
# ╠═2ade4518-1ef4-4f59-a8c8-a90531ee421e
# ╠═2b0d13b1-cff1-4183-ad08-79c83e9cc2a5
# ╠═ebcfb5a4-2aa7-477c-a911-15cf60c6a35e
# ╠═ff056014-160a-4228-9a1e-bfb0c85653e9
# ╠═582c74a3-0ec8-4706-99dd-410fa0bd2a5a
# ╠═f394a010-23ae-43fb-8cab-c0b9228c37cd
# ╠═d9b31996-0b41-407e-b017-f41a662f4bf8
# ╠═d123ee64-6d14-4d16-9336-a24c0327530c
# ╠═1e539f83-4dd0-4ff3-a71f-471e4cfb2c1e
# ╠═d40b21e3-3a24-4b72-8638-8d42d05429fd
# ╠═0b536ae7-b231-4718-a675-25804d9d30e3
# ╠═35c75e6e-ea0d-446c-b0c3-00b88574876a
# ╠═cc158fa3-cf45-4045-bfc1-bcc5dfc9c716
# ╠═01115683-f37c-4b04-b233-9b91d9c0251f
# ╠═daf9a4f4-4ef2-4f65-b27b-2e88956133c5
