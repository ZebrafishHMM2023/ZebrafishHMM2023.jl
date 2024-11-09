### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 7f95a967-426a-465f-bad4-671372ea1092
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ c797f3ce-6d16-47e1-8728-82399f2133a9
using DataFrames: DataFrame

# ╔═╡ 7d355591-dddd-4463-97b1-d53e223bd6b3
using Statistics: mean

# ╔═╡ 2b6c2c33-1168-49b1-82a3-10f9f6478188
using Statistics: std

# ╔═╡ fbb0e774-cf43-4cea-8057-81c36c8d3d04
using Statistics: var

# ╔═╡ 8398c332-235b-4f6c-ab1e-8b5fd62d2d5a
using Distributions: Gamma

# ╔═╡ 0aae9cb6-85c1-4792-8c81-2d772b2f3b19
using Distributions: Exponential

# ╔═╡ 9be8b699-c392-45b2-a80c-e5893b1ecc53
using Random: shuffle

# ╔═╡ 5b3604a6-3dcc-11ef-1ff1-a71b51c7d8e9
md"# Imports"

# ╔═╡ f9ed9420-cf6a-4c65-9e46-6e3e8fc54476
import ZebrafishHMM2023

# ╔═╡ 7571c197-a647-4c2e-8510-a05b7cbc447f
import HiddenMarkovModels

# ╔═╡ 2116822a-7e14-426a-90d3-bbc2496939ff
import Makie

# ╔═╡ 8caf193d-44a8-4e5a-8af0-0f29e81469ea
import CairoMakie

# ╔═╡ 44b6d472-e1e2-4e6b-970a-182bbe07f72f
import PlutoUI

# ╔═╡ 23f071a3-f499-4330-9685-9860e06b04ae
import CSV

# ╔═╡ 08ba1609-d6c6-4d47-a970-d68bf3771ad8
import HDF5

# ╔═╡ 4f8b021f-7917-4689-a95e-02e20a766b3a
PlutoUI.TableOfContents()

# ╔═╡ a67c5bfa-02b5-41af-8bbe-5ca350bfb58f
md"# Label ARTR states in the data"

# ╔═╡ ee9ae8b1-e589-4c71-8450-0b9ea915df21
artr_data = Dict((; temperature, fish) =>
	ZebrafishHMM2023.load_artr_wolf_2023(; temperature, fish)
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ c1fa7427-1167-446b-a8e7-50f0b321ae2d
artr_hmms = Dict((; temperature, fish) => 
	first(ZebrafishHMM2023.easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ a6c708f3-7f96-416e-8aca-1a7c02a21e98
artr_trajs = Dict((; temperature, fish) =>
	collect(eachcol(vcat(artr_data[(; temperature, fish)].left, artr_data[(; temperature, fish)].right)))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ ac410781-0712-45dd-b3eb-713b5629c329
artr_viterbi_states = Dict((; temperature, fish) =>
	HiddenMarkovModels.viterbi(artr_hmms[(; temperature, fish)], artr_trajs[(; temperature, fish)])
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 7ae067c1-8c0c-4f91-b8e1-8c39a32c821a
for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures(), fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	@assert all(diff(artr_data[(; temperature, fish)].time) .≈ mean(diff(artr_data[(; temperature, fish)].time)))
end

# ╔═╡ 3b6b3cc3-d29f-469b-ad3a-e0cc50017b02
artr_time_unit = Dict(
	(; temperature, fish) => mean(diff(artr_data[(; temperature=26, fish=6)].time))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 37ab290b-e7b6-42b2-bccf-7d73c1916557
md"# Train behavior models"

# ╔═╡ 88d1dfc7-e0c6-47c0-8170-787e290fc2c8
function train_bouts_hmm(T::Int)
	@info "Training models for T = $T"

    trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)

    trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes
    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ 0601dac5-cef9-4b2b-83ce-d37064e4d9e1
bouts_hmms = Dict(temperature => train_bouts_hmm(temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ cbf99327-4e73-43d8-aa64-05555b8ebe99
function train_full_behavior_hmm(bouts_hmm, T::Int)
	@info "Training models for T = $T"
	
	hmm = ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Exp(;
        pinit_turn=rand(), 
		transition_matrix=ZebrafishHMM2023.normalize_transition_matrix(rand(3,3)),
        σforw=0.1, turn=Gamma(1.5, 20.0),
        forward_displacement=Gamma(1.5, 1.2),
		turn_displacement=Gamma(1.5, 1.2),
        forward_interboutinterval=Exponential(1.8),
		turn_interboutinterval=Exponential(1.8),
        min_alpha = bouts_hmm.min_turn_alpha,
        only_train_spacetime=true
    )

    hmm.pinit_turn = bouts_hmm.pinit_turn
    hmm.transition_matrix .= bouts_hmm.transition_matrix
    hmm.σforw = bouts_hmm.σforw
    hmm.turn = bouts_hmm.turn

    trajs = ZebrafishHMM2023.load_full_obs(T)
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ a72cad49-5d10-453b-81c3-af9bb4b0fab7
behavior_full_hmms = Dict(temperature => train_full_behavior_hmm(bouts_hmms[temperature], temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ f3750f97-ac22-491e-99f1-d9fc1a99e423
#behavior_long_hmm = Dict(temperature => train_full_behavior_hmm(bouts_hmms[temperature], temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ 369d6c55-4239-4eeb-b23b-4c3910d791fd
md"# Sample behavior from ARTR HMM (not from ARTR data)"

# ╔═╡ edd47451-04d6-4fe1-ab6c-61ca175e9e4f
# function sample_behavior_states_from_artr(; temperature::Int, fish::Int, λ::Real)
# 	all_bout_times = [obs.t for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]
# 	time_scaling_factor = λ / artr_time_unit[(; temperature, fish)]
# 	rescaled_bout_times = all_bout_times * time_scaling_factor

# 	#all_states = artr_viterbi_states[(; temperature, fish)]
# 	all_states = rand(artr_hmms[(; temperature, fish)], 100_000).state_seq
# 	# instead of using the Viterbi labeled ARTR data, we will use the HMM to sample really long trajectories

# 	selected_bout_times = Float64[]
# 	selected_bout_times_total = 0.0
# 	while round(Int, selected_bout_times_total) < length(all_states)
# 		t = rand(rescaled_bout_times)
# 		push!(selected_bout_times, t)
# 		selected_bout_times_total += t
# 	end
	
# 	selected_indices = filter(≤(length(all_states)), round.(Int, cumsum(selected_bout_times)))

# 	return all_states[selected_indices], selected_bout_times[1:length(selected_indices)] / time_scaling_factor
# end

# ╔═╡ 51d4b383-4afc-4827-a253-5fbcb8b56e16
function sample_behavior_states_from_artr_v2(; temperature::Int, fish::Int, λ::Real, traj_length = 100_000)
	all_bout_times = [obs.t for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]

	# instead of using the Viterbi labeled ARTR data, we will use the HMM to sample really long trajectories
	#artr_states = artr_viterbi_states[(; temperature, fish)]
	artr_states = rand(artr_hmms[(; temperature, fish)], traj_length).state_seq

	artr_recording_duration = length(artr_states) * artr_time_unit[(; temperature, fish)]
	artr_recording_times = (1:length(artr_states)) .* artr_time_unit[(; temperature, fish)]

	selected_times = Float64[]
	selected_times_total = 0.0
	while true
		Δt = rand(all_bout_times)
		if (selected_times_total + Δt) * λ < artr_recording_duration
			selected_times_total += Δt
			push!(selected_times, Δt)
		else
			break
		end
	end
	@assert selected_times_total * λ < artr_recording_duration

	#selected_indices = [argmin(abs.(t * λ .- artr_recording_times)) for t = cumsum(selected_times)]
	selected_indices = [findfirst(artr_recording_times .- t * λ .≥ 0) for t = cumsum(selected_times)]
	return artr_states[selected_indices], selected_times
end

# ╔═╡ 51fc8534-b2f4-46fa-85d1-b49d252f2112
function sample_full_behavior_from_artr(; temperature::Int, fish::Int, λ::Real)
	states, times = sample_behavior_states_from_artr_v2(; temperature, fish, λ)
	return (; obs_seq = [rand(HiddenMarkovModels.obs_distribution(behavior_full_hmms[temperature], s)) for s = states], state_seq = states, times)
end

# ╔═╡ 2291f23b-b69f-4a37-9c51-945c1d0de050
function sample_full_behavior_from_artr_times_corrected(; temperature::Int, fish::Int, λ::Real)
	samples_0 = sample_full_behavior_from_artr(; temperature, fish, λ)
	@assert length(samples_0.obs_seq) == length(samples_0.times) == length(samples_0.state_seq)
	obs_seq = [ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Obs(s.θ, s.d, t) for (s, t) = zip(samples_0.obs_seq, samples_0.times)]
	return (; obs_seq, samples_0.state_seq)
end

# ╔═╡ 29371878-01c9-4f1e-b472-d224bf3e5504
my_λ = 1 / 0.44
#my_λ = 0.44 # try inverted λ

# ╔═╡ 4c2e9ba7-fc82-416e-8e14-4d5ce138d2be
let fig = Makie.Figure()
	for (n, T) = enumerate((18, 26, 33))
		fish = rand(ZebrafishHMM2023.artr_wolf_2023_fishes(T))
	    sample = sample_full_behavior_from_artr(; temperature=T, fish, λ=my_λ)
	    path = ZebrafishHMM2023.to_spatiotemporal_trajectory(sample.obs_seq)
		#path[:, 3] .= times
		
	    ax = Makie.Axis(fig[1,n], width=200, height=200, title="$T C", xlabel="mm", ylabel="mm")
	    Makie.scatterlines!(ax, path[:, 1:2], markercolor=(sample.state_seq .== 1), markersize=3)
	    Makie.xlims!(ax, -50, 50)
	    Makie.ylims!(ax, -50, 50)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ bced5da9-0584-474d-8dd2-4025eb040e67
let fig = Makie.Figure()
	_sz = 100
	_lw = 2

	λ = my_λ

	fishes = Dict(18 => 14, 26 => 7, 33 => 17)

	for (n, T) = enumerate((18, 26, 33))
		#fish = rand(ZebrafishHMM2023.artr_wolf_2023_fishes(T))
		fish = fishes[T]
		
		samples = [s for _ = 1:100 for s = sample_full_behavior_from_artr_times_corrected(; temperature=T, fish, λ).obs_seq]
		data = [obs for traj = ZebrafishHMM2023.load_full_obs(T) for obs = traj]

		_bins = -200:200
	    ax = Makie.Axis(fig[1,n], width=_sz, height=_sz, title="$T C", yscale=log10, xlabel="bout angle (deg.)", ylabel="frequency", xticks=[-150,0,150])
	    Makie.hist!(ax, [obs.θ for obs = data], normalization=:pdf, bins=_bins, color=:gray, label="data")
	    Makie.stephist!(ax, [obs.θ for obs = samples], normalization=:pdf, bins=_bins, color=:red, linewidth=_lw, label="gen.")
	    Makie.ylims!(ax, 1e-4, 0.1)
	
	    _bins = 0:0.1:12
	    ax = Makie.Axis(fig[2,n], width=_sz, height=_sz, yscale=log10, xlabel="bout distance (mm)", ylabel="frequency")
	    Makie.hist!(ax, [obs.d for obs = data], normalization=:pdf, bins=_bins, color=:gray, label="data")
	    Makie.stephist!(ax, [obs.d for obs = samples], normalization=:pdf, bins=_bins, color=:red, linewidth=_lw, label="gen.")
	    Makie.ylims!(ax, 1e-4, 1)
	    
	    _bins = 0:0.5:50
	    ax = Makie.Axis(fig[3,n], width=_sz, height=_sz, yscale=log10, xlabel="interbout time (s)", ylabel="frequency")
	    Makie.hist!(ax, [obs.t for obs = data], normalization=:pdf, bins=_bins, color=:gray, label="data")
	    Makie.stephist!(ax, [obs.t for obs = samples], normalization=:pdf, bins=_bins, color=:red, linewidth=_lw, label="gen.")
	    Makie.ylims!(ax, 1e-4, 1)
	
	    n == 3 && Makie.axislegend(ax)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 6ac94515-4e81-4eb5-88d7-eb8eb193ac76
my_my_λ = 1/0.2

# ╔═╡ 1cf0fbf5-35de-4437-82d2-e5d7ebeadac7
md"# Save data for Matteo (Behavior from Neural HMMs)"

# ╔═╡ 90f17950-5f5f-4406-ae9e-d5d81f28c8b1
_tmpdir = mktempdir()

# ╔═╡ 3b9353ab-85ea-42a7-837c-cdf313e476b0
my_λ

# ╔═╡ c179f81c-fffc-4310-8101-e9d344bc2c82
for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures(), fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	#λ = 2.775
	λ = my_λ
	for rep = 1:100
		sample = sample_full_behavior_from_artr_times_corrected(; temperature, fish, λ)
		CSV.write(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), (; bout_angle = [s.θ for s = sample.obs_seq], bout_time = [s.t for s = sample.obs_seq], bout_dist = [s.d for s = sample.obs_seq], state = sample.state_seq))
	end
	@info "temperature = $temperature, fish = $fish DONE"
end

# ╔═╡ bb4fd9ad-247c-405d-911b-2d36545e837c
md"# MSR"

# ╔═╡ ae172866-6924-46ed-b8e5-d74563a4377c
function mean_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
	#std_trajs = [(traj .- mean(traj)) / std(traj)) for traj = trajs]
	std_trajs = [traj .- mean(traj) for traj = trajs]
	#std_trajs = trajs
    return mean(abs2(sum(traj[i:(i + q)])) for traj = std_trajs for i = 1:length(traj) - q)
end

# ╔═╡ ff68a3d4-27d9-4ee4-9dec-a1ea82a03ec4
function std_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
	#std_trajs = [(traj .- mean(traj)) / std(traj) for traj = trajs]
	std_trajs = [traj .- mean(traj) for traj = trajs]
	#std_trajs = trajs
    return std(abs2(sum(traj[i:(i + q)])) for traj = trajs for i = 1:length(traj) - q) / sqrt(sum(length(traj) - q for traj = std_trajs))
end

# ╔═╡ 5ccc9a64-afff-4076-b599-6b81a79ba400
function MSR_analyses(trajs::AbstractVector{<:AbstractVector{<:Real}}; qmax = 100)
	real_msr_mean = [mean_MSR(trajs, q) for q = 0:qmax]
	real_msr_std = [std_MSR(trajs, q) for q = 0:qmax]

	shuffled_trajs = [rand(traj, length(traj)) for traj = trajs]
	
	shuffled_msr_mean = [mean_MSR(shuffled_trajs, q) for q = 0:qmax]
	shuffled_msr_std = [std_MSR(shuffled_trajs, q) for q = 0:qmax]

	return (; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std)
end

# ╔═╡ 267cf637-c931-4088-9a68-2a38e914b950
# ╠═╡ disabled = true
#=╠═╡
let fig = Makie.Figure()
	qmax = 30
	for (col, temperature) = enumerate(ZebrafishHMM2023.artr_wolf_2023_temperatures())
		
		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		
		ax = Makie.Axis(fig[1,col], width=200, height=200, title="temperature = $temperature")

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
		
			(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
			# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
			# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
			Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:blue, 0.6), linewidth=0.5)
		end

		δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature) for rep = 1:100]

		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
		Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, label="ARTR HMM")
	
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_dat; qmax)
		Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:gray, 0.5))
		Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:red, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, label="Data")
		Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="Data shuffled")
		Makie.xlims!(ax, 0, 15)
		#Makie.ylims!(ax, 0, 4e4)
		if col == 1
			Makie.axislegend(ax)
		end
	end
	Makie.resize_to_layout!(fig)
	fig
end
  ╠═╡ =#

# ╔═╡ d3f1ffa1-708d-4f0e-860b-5a839dcf1b54
[Float64[]]

# ╔═╡ e7babee3-87f5-4cff-a60c-5006f8c92cd9
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 10

	MSR_for_bar_plot_hmm = [Float64[] for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSR_for_bar_plot_dat = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSR_for_bar_plot_dat_shuffled = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	
	for (col, temperature) = enumerate(ZebrafishHMM2023.artr_wolf_2023_temperatures())
		
		ax = Makie.Axis(fig[1,col], width=170, height=170, title="temperature = $temperature", xgridvisible=false, ygridvisible=false)


		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
		
			(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
			# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
			# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, linewidth=1)
			end

			push!(MSR_for_bar_plot_hmm[col], real_msr_mean[q_for_barplot])
		end

		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_dat; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:gray, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:red, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		Makie.ylims!(ax, 0, 4e4)

		MSR_for_bar_plot_dat[col] = real_msr_mean[q_for_barplot]
		MSR_for_bar_plot_dat_shuffled[col] = shuffled_msr_mean[q_for_barplot]
		
		if col == 1
			Makie.axislegend(ax)
		end
	end

	ax = Makie.Axis(fig[2,1:2], width=2 * 170, height=170, xgridvisible=false, ygridvisible=false)

	Makie.barplot!(ax, 1:5, map(mean, MSR_for_bar_plot_hmm), color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:5, map(mean, MSR_for_bar_plot_hmm), map(std, MSR_for_bar_plot_hmm) ./ map(sqrt ∘ length, MSR_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5
	)
	for t = 1:5
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat[t:t]; color=:black, linewidth=4)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=:dash)
	end
	# Makie.lines!(ax, [0.7, 1.3], MSR_for_bar_plot_dat[1:1]; color=:black, linewidth=4)
	# Makie.lines!(ax, [1.7, 2.3], MSR_for_bar_plot_dat[2:2]; color=:black, linewidth=4)
	# Makie.lines!(ax, [2.7, 3.3], MSR_for_bar_plot_dat[3:3]; color=:black, linewidth=4)
	# Makie.lines!(ax, [3.7, 4.3], MSR_for_bar_plot_dat[4:4]; color=:black, linewidth=4)
	# Makie.lines!(ax, [4.7, 5.3], MSR_for_bar_plot_dat[5:5]; color=:black, linewidth=4)
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ f6963397-4e09-4cfc-8d33-64a5a37cad96
[length([obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]) for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]

# ╔═╡ eb54ac49-2607-4590-bbc1-84cc7cca9d52
[mean(length(CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature) for rep = 1:100) for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]

# ╔═╡ 159900f7-1b9f-4b0f-8b89-c7db803137a3
md"# Long trajectories"

# ╔═╡ 63c2663b-4cf6-4b90-96d1-ad7f840b02ff
trajs_long = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ 550b1a3a-ba25-48f7-aec4-509704ffe21c
let fig = Makie.Figure()
	qmax = 30
	
	ax = Makie.Axis(fig[1,1], width=400, height=400)

	for fish = trajs_long
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(fish; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=:lightgreen)
		Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:lightpink)
	end
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ ffdb4a8d-6a75-441f-b6a5-c6c3321d37b7
let fig = Makie.Figure()
	qmax = 30
	
	all_fish_MSR = [MSR_analyses(fish; qmax) for fish = trajs_long]
	
	ax = Makie.Axis(fig[1,1], width=400, height=400)

	for fish = all_fish_MSR
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = fish
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=:lightgreen)
		Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:lightpink)
	end

	# Makie.lines!(ax, 0:qmax, dropdims(mean(stack(fish.real_msr_mean for fish = all_fish_MSR); dims=2); dims=2); color=:green, linewidth=4)
	# Makie.lines!(ax, 0:qmax, dropdims(mean(stack(fish.shuffled_msr_mean for fish = all_fish_MSR); dims=2); dims=2); color=:red, linewidth=4)

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 7d179bcd-5dd6-40e3-b96a-5b4d0e7a30a7
let fig = Makie.Figure()
	qmax = 40
	temperature = 26 # temperature of the long trajectory

	q_for_inset = 20

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)

	MSR_real_fish = Float64[]
	MSR_hmm_fish = Float64[]
	MSR_hmm_shuffled_fish = Float64[]

	for (n, fish) = enumerate(trajs_long)
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(fish; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:gray, 0.7), linewidth=1)
		#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:lightpink)

		push!(MSR_real_fish, real_msr_mean[q_for_inset + 1])
	end


	δθ_dat = [traj for fish = trajs_long for traj = fish]
	(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_dat; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, linewidth=5, linestyle=:dash, label="data")

	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
		δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
	
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:blue, 0.9), linewidth=1)
		#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=(:red, 0.9), linewidth=1)

		push!(MSR_hmm_fish, real_msr_mean[q_for_inset + 1])
		push!(MSR_hmm_shuffled_fish, shuffled_msr_mean[q_for_inset + 1])
	end
	
	δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature) for rep = 1:100]

	(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
	# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
	# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, label="ARTR HMM", linewidth=5, linestyle=:dash)
	#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="ARTR HMM (shuffled)", linewidth=5, linestyle=:dash)

	# δθ_dat = [[obs.θ for obs = traj] for traj = ZebrafishHMM2023.load_full_obs(temperature)]
	# (; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_dat; qmax)
	# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:gray, 0.5))
	# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:red, 0.5))
	# Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, label="Data")
	# Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="Data shuffled")
	
	Makie.xlims!(ax, 0, qmax)
	Makie.ylims!(ax, 0, 4e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)

	ax = Makie.Axis(fig[1,2], width=300, height=300)
	Makie.barplot!(1:1, [mean(MSR_real_fish)]; color=:lightgray, strokecolor=:black, strokewidth=5)
	Makie.barplot!(2:2, [mean(MSR_hmm_fish)]; color=:lightblue, strokecolor=:blue)
	Makie.barplot!(3:3, [mean(MSR_hmm_shuffled_fish)]; color=:pink, strokecolor=:red)

	Makie.errorbars!(1:1, [mean(MSR_real_fish) - std(MSR_real_fish)/2], [mean(MSR_real_fish) + std(MSR_real_fish)/2]; color=:black)
	Makie.errorbars!(2:2, [mean(MSR_hmm_fish) - std(MSR_hmm_fish)/2], [mean(MSR_hmm_fish) + std(MSR_hmm_fish)/2]; color=:blue)
	Makie.errorbars!(3:3, [mean(MSR_hmm_shuffled_fish) - std(MSR_hmm_shuffled_fish)/2], [mean(MSR_hmm_shuffled_fish) + std(MSR_hmm_shuffled_fish)/2]; color=:red)
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 5d3b3335-d787-44b9-a308-b5f73e6cc75c
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
let fig = Makie.Figure()
	qmax = 30
	temperature = 18 # temperature of the long trajectory
				
	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	for (n, fish) = enumerate(trajs_long)
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(fish; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		if n == 1
			Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:gray, 0.7), linewidth=1, label="data")
		else
			Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:gray, 0.7), linewidth=1)
		end
		#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:lightpink)
	end

	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
		δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
	
		(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
		# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
		# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:blue, 0.9), linewidth=1)
		Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=(:red, 0.9), linewidth=1)
	end
	
	δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature) for rep = 1:100]

	(; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_hmm; qmax)
	# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:blue, 0.5))
	# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:blue, 0.5))
	# Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, label="ARTR HMM", linewidth=5)
	# Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="ARTR HMM (shuffled)", linewidth=5)

	# δθ_dat = [[obs.θ for obs = traj] for traj = ZebrafishHMM2023.load_full_obs(temperature)]
	# (; real_msr_mean, real_msr_std, shuffled_msr_mean, shuffled_msr_std) = MSR_analyses(δθ_dat; qmax)
	# Makie.band!(ax, 0:qmax, real_msr_mean - real_msr_std, real_msr_mean + real_msr_std; color=(:gray, 0.5))
	# Makie.band!(ax, 0:qmax, shuffled_msr_mean - shuffled_msr_std, shuffled_msr_mean + shuffled_msr_std, color=(:red, 0.5))
	# Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, label="Data")
	# Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=:red, label="Data shuffled")
	
	Makie.xlims!(ax, 0, qmax)
	Makie.ylims!(ax, 0, 4e4)

	Makie.axislegend(ax, position=:lt)
	Makie.resize_to_layout!(fig)
	fig
end
  ╠═╡ =#

# ╔═╡ d40d0d61-46dc-4f0b-ab43-d6d7d4ddb318
var([obs.θ for traj = ZebrafishHMM2023.load_full_obs(26) for obs = traj])

# ╔═╡ 22744bd3-1792-41bc-a064-f855dfaf5297
var([obs.θ for traj = ZebrafishHMM2023.load_full_obs(18) for obs = traj])

# ╔═╡ 7b96c684-d1cf-4702-816e-2bf7cb1561bb
mean(map(length, ZebrafishHMM2023.load_full_obs(18)))

# ╔═╡ 0ab3e8bb-ef4d-4937-8e84-77f6acb2acd0
map(length, ZebrafishHMM2023.load_full_obs(18))

# ╔═╡ 7456ec34-1b23-4f4d-b4bb-3d9d41864a67
md"# HMMs trained on long trajectories of individual fish"

# ╔═╡ d47a91f7-6c68-4f13-a6c8-a17f54713504
function my_train_for_long_trajs(fish_trajs)
	hmm = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0)
	hmm = ZebrafishHMM2023.normalize_all!(hmm)
	(hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish_trajs, length(fish_trajs); max_iterations=5000, check_loglikelihood_increasing=false, atol = ZebrafishHMM2023.ATol(1e-5))
	return hmm
end

# ╔═╡ 7a785046-99a2-44dc-a229-134d32f85655
long_hmms = [my_train_for_long_trajs(fish) for fish = trajs_long]

# ╔═╡ abb3ac26-dacc-4534-beba-b656ae06401f
function my_train_full_behavior_hmm_for_long_trajs(bouts_hmm, fish_trajs)	
	hmm = ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Exp(;
        pinit_turn=rand(), 
		transition_matrix=ZebrafishHMM2023.normalize_transition_matrix(rand(3,3)),
        σforw=0.1, turn=Gamma(1.5, 20.0),
        forward_displacement=Gamma(1.5, 1.2),
		turn_displacement=Gamma(1.5, 1.2),
        forward_interboutinterval=Exponential(1.8),
		turn_interboutinterval=Exponential(1.8),
        min_alpha = bouts_hmm.min_turn_alpha,
        only_train_spacetime=true
    )

    hmm.pinit_turn = bouts_hmm.pinit_turn
    hmm.transition_matrix .= bouts_hmm.transition_matrix
    hmm.σforw = bouts_hmm.σforw
    hmm.turn = bouts_hmm.turn

    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish_trajs, length(fish_trajs); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))

    return hmm
end

# ╔═╡ 9ab9ebcb-9722-46e8-a3a1-3b43d41ad472
long_full_hmms = [my_train_full_behavior_hmm_for_long_trajs(bouts_hmm, fish_trajs) for (bouts_hmm, fish_trajs) = zip(long_hmms, ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())]

# ╔═╡ 0153111a-db86-473e-bb98-ee3d8d3a8607
ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[1][1][1].θ

# ╔═╡ f2d75003-f756-4c12-873e-d095dac0cc4c
var(x for fish = trajs_long for traj = fish for x = traj)

# ╔═╡ 3d407a1d-371b-405e-ac10-2dfb1cc1e017
[var(x for traj = fish for x = traj) for fish = trajs_long]

# ╔═╡ f0d7606f-18ab-453a-8deb-618e0c9680eb
let fig = Makie.Figure()
	for (n, fish) = enumerate(trajs_long)
		row, col = fldmod1(n, 6)
		ax = Makie.Axis(fig[row, col], width=100, height=100)
		Makie.hist!(ax, [x for traj = fish for x = traj]; bins=-100:5:100, normalization=:pdf)
		Makie.ylims!(ax, 0, 0.07)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ Cell order:
# ╠═5b3604a6-3dcc-11ef-1ff1-a71b51c7d8e9
# ╠═7f95a967-426a-465f-bad4-671372ea1092
# ╠═f9ed9420-cf6a-4c65-9e46-6e3e8fc54476
# ╠═7571c197-a647-4c2e-8510-a05b7cbc447f
# ╠═2116822a-7e14-426a-90d3-bbc2496939ff
# ╠═8caf193d-44a8-4e5a-8af0-0f29e81469ea
# ╠═44b6d472-e1e2-4e6b-970a-182bbe07f72f
# ╠═23f071a3-f499-4330-9685-9860e06b04ae
# ╠═08ba1609-d6c6-4d47-a970-d68bf3771ad8
# ╠═c797f3ce-6d16-47e1-8728-82399f2133a9
# ╠═7d355591-dddd-4463-97b1-d53e223bd6b3
# ╠═2b6c2c33-1168-49b1-82a3-10f9f6478188
# ╠═fbb0e774-cf43-4cea-8057-81c36c8d3d04
# ╠═8398c332-235b-4f6c-ab1e-8b5fd62d2d5a
# ╠═0aae9cb6-85c1-4792-8c81-2d772b2f3b19
# ╠═9be8b699-c392-45b2-a80c-e5893b1ecc53
# ╠═4f8b021f-7917-4689-a95e-02e20a766b3a
# ╠═a67c5bfa-02b5-41af-8bbe-5ca350bfb58f
# ╠═ee9ae8b1-e589-4c71-8450-0b9ea915df21
# ╠═c1fa7427-1167-446b-a8e7-50f0b321ae2d
# ╠═a6c708f3-7f96-416e-8aca-1a7c02a21e98
# ╠═ac410781-0712-45dd-b3eb-713b5629c329
# ╠═7ae067c1-8c0c-4f91-b8e1-8c39a32c821a
# ╠═3b6b3cc3-d29f-469b-ad3a-e0cc50017b02
# ╠═37ab290b-e7b6-42b2-bccf-7d73c1916557
# ╠═88d1dfc7-e0c6-47c0-8170-787e290fc2c8
# ╠═0601dac5-cef9-4b2b-83ce-d37064e4d9e1
# ╠═cbf99327-4e73-43d8-aa64-05555b8ebe99
# ╠═a72cad49-5d10-453b-81c3-af9bb4b0fab7
# ╠═f3750f97-ac22-491e-99f1-d9fc1a99e423
# ╠═369d6c55-4239-4eeb-b23b-4c3910d791fd
# ╠═edd47451-04d6-4fe1-ab6c-61ca175e9e4f
# ╠═51d4b383-4afc-4827-a253-5fbcb8b56e16
# ╠═51fc8534-b2f4-46fa-85d1-b49d252f2112
# ╠═2291f23b-b69f-4a37-9c51-945c1d0de050
# ╠═29371878-01c9-4f1e-b472-d224bf3e5504
# ╠═4c2e9ba7-fc82-416e-8e14-4d5ce138d2be
# ╠═bced5da9-0584-474d-8dd2-4025eb040e67
# ╠═6ac94515-4e81-4eb5-88d7-eb8eb193ac76
# ╠═1cf0fbf5-35de-4437-82d2-e5d7ebeadac7
# ╠═90f17950-5f5f-4406-ae9e-d5d81f28c8b1
# ╠═3b9353ab-85ea-42a7-837c-cdf313e476b0
# ╠═c179f81c-fffc-4310-8101-e9d344bc2c82
# ╠═bb4fd9ad-247c-405d-911b-2d36545e837c
# ╠═ae172866-6924-46ed-b8e5-d74563a4377c
# ╠═ff68a3d4-27d9-4ee4-9dec-a1ea82a03ec4
# ╠═5ccc9a64-afff-4076-b599-6b81a79ba400
# ╠═267cf637-c931-4088-9a68-2a38e914b950
# ╠═d3f1ffa1-708d-4f0e-860b-5a839dcf1b54
# ╠═e7babee3-87f5-4cff-a60c-5006f8c92cd9
# ╠═f6963397-4e09-4cfc-8d33-64a5a37cad96
# ╠═eb54ac49-2607-4590-bbc1-84cc7cca9d52
# ╠═159900f7-1b9f-4b0f-8b89-c7db803137a3
# ╠═63c2663b-4cf6-4b90-96d1-ad7f840b02ff
# ╠═550b1a3a-ba25-48f7-aec4-509704ffe21c
# ╠═ffdb4a8d-6a75-441f-b6a5-c6c3321d37b7
# ╠═7d179bcd-5dd6-40e3-b96a-5b4d0e7a30a7
# ╠═5d3b3335-d787-44b9-a308-b5f73e6cc75c
# ╠═d40d0d61-46dc-4f0b-ab43-d6d7d4ddb318
# ╠═22744bd3-1792-41bc-a064-f855dfaf5297
# ╠═7b96c684-d1cf-4702-816e-2bf7cb1561bb
# ╠═0ab3e8bb-ef4d-4937-8e84-77f6acb2acd0
# ╠═7456ec34-1b23-4f4d-b4bb-3d9d41864a67
# ╠═d47a91f7-6c68-4f13-a6c8-a17f54713504
# ╠═7a785046-99a2-44dc-a229-134d32f85655
# ╠═abb3ac26-dacc-4534-beba-b656ae06401f
# ╠═9ab9ebcb-9722-46e8-a3a1-3b43d41ad472
# ╠═0153111a-db86-473e-bb98-ee3d8d3a8607
# ╠═f2d75003-f756-4c12-873e-d095dac0cc4c
# ╠═3d407a1d-371b-405e-ac10-2dfb1cc1e017
# ╠═f0d7606f-18ab-453a-8deb-618e0c9680eb
