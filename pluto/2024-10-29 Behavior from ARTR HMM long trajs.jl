### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 673c2edc-3667-4834-b489-cdc1d8eec407
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 9fcb9477-1365-42e5-b29a-a7e20feddd41
using DataFrames: DataFrame

# ╔═╡ ef3f2f53-d4de-4878-95c2-027d2d34ae9e
using Statistics: mean

# ╔═╡ 1563ced7-a36a-4447-8119-efb6b235d45d
using Statistics: std

# ╔═╡ 023a473d-9da9-4f91-ae12-2cf43ead960b
using Statistics: var

# ╔═╡ 12904676-18eb-42bc-95a8-05832c7c7663
using Distributions: Gamma

# ╔═╡ 027ee889-e209-46c8-bc29-7f44954ead09
using Distributions: Exponential

# ╔═╡ 320ea063-d861-4bc9-947d-e54ad08fd92b
using Random: shuffle

# ╔═╡ 30ba0901-8e9c-4263-b664-5f2ebc7a1c7a
using LinearAlgebra: I, inv, det, eigvals, eigen

# ╔═╡ 051569b7-54de-4af0-aeed-135958d24081
md"# Imports"

# ╔═╡ 52d5317f-f2d1-4b8f-8104-eed727415ea2
import ZebrafishHMM2023

# ╔═╡ cff51f25-e94d-4a9a-bb68-f3b2f70af381
import HiddenMarkovModels

# ╔═╡ 71c4b7c7-a290-49c0-9168-1d6fec43bdb5
import Makie

# ╔═╡ cfc99da8-eb5e-4715-9a21-78918da0d01c
import CairoMakie

# ╔═╡ 31df8958-e175-4429-97f3-7fb38b5ce898
import PlutoUI

# ╔═╡ 896ce97c-b64b-4e2b-abc4-e2403b82bcbb
import CSV

# ╔═╡ 204fd97c-4be3-4841-bd74-d2d80605cf00
import HDF5

# ╔═╡ 0ae0436c-6db4-4013-8db0-2b4f90c568e4
PlutoUI.TableOfContents()

# ╔═╡ 268c7d21-8cfe-4d90-9590-6e4a08efc2c6
temperature = 26 # temperature of the long trajectory
# Everything here is at this temperature

# ╔═╡ 99879535-3957-45f6-a150-8baa82ce780f
md"# Label ARTR states in the data"

# ╔═╡ 85bee758-0273-4244-bc32-03ce527a267a
artr_data = Dict(fish => ZebrafishHMM2023.load_artr_wolf_2023(; temperature, fish) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))

# ╔═╡ 3186c6a6-4517-4f57-b801-6080c77fc877
artr_hmms = Dict(fish => first(ZebrafishHMM2023.easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true)) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))

# ╔═╡ 825f3959-b20d-4885-a985-9000bb1a5305
artr_trajs = Dict(fish => collect(eachcol(vcat(artr_data[fish].left, artr_data[fish].right))) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))

# ╔═╡ 1d4072ba-351b-4b3e-9b78-ec537d8e4257
artr_viterbi_states = Dict(fish => HiddenMarkovModels.viterbi(artr_hmms[fish], artr_trajs[fish]) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))

# ╔═╡ d6b7b2f2-d886-4776-a75b-2cdcd06a81c0
for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	@assert all(diff(artr_data[fish].time) .≈ mean(diff(artr_data[fish].time)))
end

# ╔═╡ 67a29f4b-da79-4961-9cc3-adc652f4972b
artr_time_unit = Dict(fish => mean(diff(artr_data[fish].time)) for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))

# ╔═╡ c15f8e4e-964b-4374-8816-f7c29a9578eb
md"# MSR computations"

# ╔═╡ 72bb23ac-c329-4266-b598-8e7e9b0b1188
function mean_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
    return mean(abs2(sum(traj[i:(i + q)])) for traj = trajs for i = 1:length(traj) - q)
end

# ╔═╡ 868491ca-35c3-481c-a964-2cbe6bdd1744
function std_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
    return std(abs2(sum(traj[i:(i + q)])) for traj = trajs for i = 1:length(traj) - q) / sqrt(sum(length(traj) - q for traj = trajs))
end

# ╔═╡ 4eeb105d-8c88-40a8-8bce-789e4adb98e7
md"# MSR of the long trajectory data"

# ╔═╡ a47b3dfb-d668-49b0-92b7-8d901c6006e6
let fig = Makie.Figure()
	qmax = 15

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="Streak length (# bouts)", ylabel="MSR (deg^2)", xgridvisible=false, ygridvisible=false)
	
	for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()
		δθ_dat = [[obs.θ for obs = traj] for traj = fish]

		# normalize
		μ = mean(δθ for traj = δθ_dat for δθ = traj)
		σ = std(δθ for traj = δθ_dat for δθ = traj)
		δθ_dat = [[(δθ - μ) / σ for δθ = traj] for traj = δθ_dat]
		
		msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="data")
	end
	Makie.lines!(ax, 0:qmax, 0:qmax, color=:red, linewidth=4, linestyle=:dash)
	
	#Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ f7b4589a-e444-43e1-bde1-5fb15ab14ceb
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=400, height=400)
	Makie.hist!(ax, [length(traj) for t = ZebrafishHMM2023.artr_wolf_2023_temperatures() for traj = ZebrafishHMM2023.load_full_obs(t)]; normalization=:pdf, bins=0:5:800, label="short")
	Makie.hist!(ax, [length(traj) for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs() for traj = fish]; normalization=:pdf, bins=0:5:800, label="long")
	Makie.axislegend(ax)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 315eb13a-ee2c-4973-8a31-46d12cd856d8
md"# HMMs trained on long trajectories of individual fish"

# ╔═╡ 6f46df80-a0ee-497f-a1e5-480f21aa169f
all_long_trajs_together = [traj for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs() for traj = fish]

# ╔═╡ 6c8538f5-6538-4a22-8c37-b879547e0073
function my_train_bouts_hmm(full_fish_trajs)
	trajs_bouts_only = [[obs.θ for obs = traj] for traj = full_fish_trajs]

	hmm_bouts = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.1, 15.0), 1.0)
	hmm_bouts = ZebrafishHMM2023.normalize_all!(hmm_bouts)

	@assert !any(isnan, hmm_bouts.transition_matrix)
	@assert hmm_bouts.min_turn_alpha > 0

	(hmm_bouts, lL) = HiddenMarkovModels.baum_welch(hmm_bouts, trajs_bouts_only, length(trajs_bouts_only); max_iterations=5000, check_loglikelihood_increasing=false, atol=ZebrafishHMM2023.ATol(1e-5))

	return hmm_bouts
end

# ╔═╡ 566f3242-3a1c-4362-bd20-625e65ce5079
hmm_all_together = my_train_bouts_hmm(all_long_trajs_together)

# ╔═╡ a6f7f26a-1b16-4960-9946-990dc1a93eb7
function my_train_full_behavior_hmm_for_long_trajs(full_fish_trajs)
	@info "Training bouts only HMM ..."
	@info length(full_fish_trajs)

	hmm_bouts = my_train_bouts_hmm(full_fish_trajs)
	
	@assert !isnan(hmm_bouts.pinit_turn)
	@assert !any(isnan, hmm_bouts.transition_matrix)

	@info "Training full HMM ..."
	
	hmm = ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Exp(;
        pinit_turn=rand(), 
		transition_matrix = ZebrafishHMM2023.normalize_transition_matrix(rand(3,3)),
        σforw = 0.1,
		turn = Gamma(1.5, 20.0),
        forward_displacement = Gamma(1.5, 1.2),
		turn_displacement = Gamma(1.5, 1.2),
        forward_interboutinterval = Exponential(1.8),
		turn_interboutinterval = Exponential(1.8),
        min_alpha = hmm_bouts.min_turn_alpha,
        only_train_spacetime = true
    )

    hmm.pinit_turn = hmm_bouts.pinit_turn
    hmm.transition_matrix .= hmm_bouts.transition_matrix
    hmm.σforw = hmm_bouts.σforw
    hmm.turn = hmm_bouts.turn

    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, full_fish_trajs, length(full_fish_trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ e525baca-de09-4fe9-9af1-42018cf08ab8
full_hmms = [my_train_full_behavior_hmm_for_long_trajs(fish_trajs) for fish_trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]

# ╔═╡ da0e0a6a-7e00-41eb-820f-ab077356a810
md" # Sample behavior from ARTR HMM"

# ╔═╡ eb62d267-d5ad-4bfb-8ff0-d84fbbe17604
function sample_behavior_states_from_artr(; fish::Int, λ::Real, traj_length::Int = 100_000)
	all_bout_times = [obs.t for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]

	# instead of using the Viterbi labeled ARTR data, we will use the HMM to sample really long trajectories
	#artr_states = artr_viterbi_states[(; temperature, fish)]
	artr_states = rand(artr_hmms[fish], traj_length).state_seq

	artr_recording_duration = length(artr_states) * artr_time_unit[fish]
	artr_recording_times = (1:length(artr_states)) .* artr_time_unit[fish]

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

# ╔═╡ 0e482409-4d17-415e-a1a3-185d07c5c9f1
function sample_full_behavior_from_artr(; fish::Int, λ::Real, behavior_fish::Int, traj_length::Int = 100_000)
	states, times = sample_behavior_states_from_artr(; fish, λ, traj_length)
	return (; obs_seq = [rand(HiddenMarkovModels.obs_distribution(full_hmms[behavior_fish], s)) for s = states], state_seq = states, times)
end

# ╔═╡ 74bf7fd8-c134-4283-a51b-5cbf50a85f48
function sample_full_behavior_from_artr_times_corrected(; fish::Int, λ::Real, behavior_fish::Int, traj_length::Int = 100_000)
	samples_0 = sample_full_behavior_from_artr(; fish, λ, behavior_fish, traj_length)
	@assert length(samples_0.obs_seq) == length(samples_0.times) == length(samples_0.state_seq)
	obs_seq = [ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Obs(s.θ, s.d, t) for (s, t) = zip(samples_0.obs_seq, samples_0.times)]
	return (; obs_seq, samples_0.state_seq)
end

# ╔═╡ af1ac20a-e002-4de0-8a4d-9498e1343b03
my_λ = 1 / 0.44

# ╔═╡ 615af0e8-d1e8-4eb1-8177-cf4ae7469fd9
_tmpdir = mktempdir()

# ╔═╡ faf75044-6a3a-47f9-8b54-81bbe6a2426a
for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	break
	#λ = 2.775
	λ = my_λ
	for rep = 1:100
		sample = sample_full_behavior_from_artr_times_corrected(; fish, λ, behavior_fish=1, traj_length=100_000)
		CSV.write(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), (; bout_angle = [s.θ for s = sample.obs_seq], bout_time = [s.t for s = sample.obs_seq], bout_dist = [s.d for s = sample.obs_seq], state = sample.state_seq))
	end
	break
	@info "temperature = $temperature, fish = $fish DONE"
end

# ╔═╡ b729df36-39a0-47e6-8b32-b13e68008052
let fig = Makie.Figure()
	qmax = 15
	behavior_fish = 2

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	# null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	# Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)
	for (n, fish) = enumerate(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
		hmm_sample = sample_full_behavior_from_artr_times_corrected(; fish, λ=my_λ, behavior_fish, traj_length=10000)
		δθ_hmm = [[obs.θ for obs = hmm_sample.obs_seq]]
		msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=(:blue, 0.7), linewidth=1)
	end

	δθ_dat = [[obs.θ for obs = traj] for traj = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[behavior_fish]]
	msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
	Makie.lines!(ax, 0:qmax, msr, color=:black, linewidth=5, linestyle=:dash, label="data")

	Makie.lines!(ax, 0:qmax, var(δθ for traj = δθ_dat for δθ = traj) .* (0:qmax), color=:red)
	#Makie.lines!(ax, 0:qmax, 0:qmax, color=:red)

	#Makie.xlims!(ax, 0, qmax)
	#Makie.ylims!(ax, 0, 20e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ a0ed51ef-481f-46fe-bcac-b74cf25a216a
let fig = Makie.Figure()
	qmax = 40
	behavior_fish = 3

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	# null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	# Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)
	for (n, fish) = enumerate(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
		hmm_sample = sample_full_behavior_from_artr_times_corrected(; fish, λ=my_λ, behavior_fish, traj_length=10000)
		δθ_hmm = [[obs.θ for obs = hmm_sample.obs_seq]]
		msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=(:blue, 0.7), linewidth=1)
	end

	δθ_dat = [[obs.θ for obs = traj] for traj = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[behavior_fish]]
	msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
	Makie.lines!(ax, 0:qmax, msr, color=:black, linewidth=5, linestyle=:dash, label="data")

	#Makie.lines!(ax, 0:qmax, var(obs.θ for traj = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[behavior_fish] for obs = traj) .* (0:qmax), color=:red)
	Makie.lines!(ax, 0:qmax, 0:qmax, color=:red)

	#Makie.xlims!(ax, 0, qmax)
	#Makie.ylims!(ax, 0, 20e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 50b8ab07-f05a-40f3-9fff-b3d47ffe7744
let fig = Makie.Figure()
	qmax = 40
	behavior_fish = 3

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	# null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	# Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)
	for (n, fish) = enumerate(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
		hmm_sample = sample_full_behavior_from_artr_times_corrected(; fish, λ=my_λ, behavior_fish, traj_length=10000)
		real_msr_mean = MSR_analyses([[obs.θ for obs = hmm_sample.obs_seq]]; qmax)
		Makie.lines!(ax, 0:qmax, real_msr_mean, color=(:blue, 0.7), linewidth=1)
	end

	real_msr_mean = MSR_analyses([[obs.θ for obs = traj] for traj = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[behavior_fish]]; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, linewidth=5, linestyle=:dash, label="data")

	#Makie.xlims!(ax, 0, qmax)
	#Makie.ylims!(ax, 0, 4e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 928d18b4-e0c0-41d6-8aa3-2b0239145a98
let fig = Makie.Figure()
	qmax = 15
	behavior_fish = 10

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	# null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	# Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)
	hmm_samples = [rand(full_hmms[behavior_fish], 10000) for _ = 1:10]
	real_msr_mean = MSR_analyses([[obs.θ for obs = hmm_sample.obs_seq] for hmm_sample = hmm_samples]; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, linewidth=1, label="HMM")

	# shuffled_trajs = [rand(traj, length(traj)) for traj = trajs]
	# shuffled_msr_mean = [mean_MSR(shuffled_trajs, q) for q = 0:qmax]
	# return (; real_msr_mean, shuffled_msr_mean)

	real_msr_mean = MSR_analyses([[obs.θ for obs = traj] for traj = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[behavior_fish]]; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, linewidth=5, linestyle=:dash, label="data")
	#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=(:red, 0.7), linewidth=1)

	Makie.xlims!(ax, 0, qmax)
	#Makie.ylims!(ax, 0, 4e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 17330e44-cd12-4fb1-a7b5-94fac18b563d
let fig = Makie.Figure()
	qmax = 15

	ax = Makie.Axis(fig[1,1], width=300, height=300, xlabel="q", ylabel="MSR(q)", xgridvisible=false, ygridvisible=false)

	# null_coeff = var([δθ for fish = trajs_long for traj = fish for δθ = traj])
	# Makie.lines!(ax, 0:qmax, null_coeff .* (0:qmax), color=:red, linewidth=4, linestyle=:dash)
	hmm_samples = [rand(hmm, 10000) for hmm = full_hmms]
	real_msr_mean = MSR_analyses_v2([[obs.θ for obs = hmm_sample.obs_seq] for hmm_sample = hmm_samples]; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:blue, linewidth=1, label="HMM")

	real_msr_mean = MSR_analyses_v2([[obs.θ for obs = traj] for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs() for traj = fish]; qmax)
	Makie.lines!(ax, 0:qmax, real_msr_mean, color=:black, linewidth=5, linestyle=:dash, label="data")
	#Makie.lines!(ax, 0:qmax, shuffled_msr_mean, color=(:red, 0.7), linewidth=1)

	Makie.xlims!(ax, 0, qmax)
	#Makie.ylims!(ax, 0, 4e4)

	Makie.axislegend(ax; position=:lt, framevisible=false)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ d58dc85f-48ae-4de0-bbfd-69ef447af559
map(length, ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()[1]) |> sum

# ╔═╡ 4f88a6e3-0a93-458c-8a20-61fcd4bbd10e
eigvals(I - full_hmms[1].transition_matrix)

# ╔═╡ 0ca030f4-f6e0-4e0f-a1ab-77bab03507a6
sum(full_hmms[1].transition_matrix^t for t = 0:1000)

# ╔═╡ 7609d04e-77e4-4665-b4ac-bcc406a0a8ef
eigvals(full_hmms[1].transition_matrix)

# ╔═╡ 1e60ed43-4109-4cb7-9623-3a63bc8bf790
[length(fish) for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]

# ╔═╡ 199ad7c1-7941-496b-ac9f-cfcfbb32b51e
for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()
	println(length(fish), " trajs for this fish, of lengths:")
	for traj = fish
		print(length(traj), ", ")
	end
	print("\n")
end

# ╔═╡ Cell order:
# ╠═051569b7-54de-4af0-aeed-135958d24081
# ╠═673c2edc-3667-4834-b489-cdc1d8eec407
# ╠═52d5317f-f2d1-4b8f-8104-eed727415ea2
# ╠═cff51f25-e94d-4a9a-bb68-f3b2f70af381
# ╠═71c4b7c7-a290-49c0-9168-1d6fec43bdb5
# ╠═cfc99da8-eb5e-4715-9a21-78918da0d01c
# ╠═31df8958-e175-4429-97f3-7fb38b5ce898
# ╠═896ce97c-b64b-4e2b-abc4-e2403b82bcbb
# ╠═204fd97c-4be3-4841-bd74-d2d80605cf00
# ╠═9fcb9477-1365-42e5-b29a-a7e20feddd41
# ╠═ef3f2f53-d4de-4878-95c2-027d2d34ae9e
# ╠═1563ced7-a36a-4447-8119-efb6b235d45d
# ╠═023a473d-9da9-4f91-ae12-2cf43ead960b
# ╠═12904676-18eb-42bc-95a8-05832c7c7663
# ╠═027ee889-e209-46c8-bc29-7f44954ead09
# ╠═320ea063-d861-4bc9-947d-e54ad08fd92b
# ╠═30ba0901-8e9c-4263-b664-5f2ebc7a1c7a
# ╠═0ae0436c-6db4-4013-8db0-2b4f90c568e4
# ╠═268c7d21-8cfe-4d90-9590-6e4a08efc2c6
# ╠═99879535-3957-45f6-a150-8baa82ce780f
# ╠═85bee758-0273-4244-bc32-03ce527a267a
# ╠═3186c6a6-4517-4f57-b801-6080c77fc877
# ╠═825f3959-b20d-4885-a985-9000bb1a5305
# ╠═1d4072ba-351b-4b3e-9b78-ec537d8e4257
# ╠═d6b7b2f2-d886-4776-a75b-2cdcd06a81c0
# ╠═67a29f4b-da79-4961-9cc3-adc652f4972b
# ╠═c15f8e4e-964b-4374-8816-f7c29a9578eb
# ╠═72bb23ac-c329-4266-b598-8e7e9b0b1188
# ╠═868491ca-35c3-481c-a964-2cbe6bdd1744
# ╠═4eeb105d-8c88-40a8-8bce-789e4adb98e7
# ╠═a47b3dfb-d668-49b0-92b7-8d901c6006e6
# ╠═f7b4589a-e444-43e1-bde1-5fb15ab14ceb
# ╠═315eb13a-ee2c-4973-8a31-46d12cd856d8
# ╠═6f46df80-a0ee-497f-a1e5-480f21aa169f
# ╠═566f3242-3a1c-4362-bd20-625e65ce5079
# ╠═6c8538f5-6538-4a22-8c37-b879547e0073
# ╠═a6f7f26a-1b16-4960-9946-990dc1a93eb7
# ╠═e525baca-de09-4fe9-9af1-42018cf08ab8
# ╠═da0e0a6a-7e00-41eb-820f-ab077356a810
# ╠═eb62d267-d5ad-4bfb-8ff0-d84fbbe17604
# ╠═0e482409-4d17-415e-a1a3-185d07c5c9f1
# ╠═74bf7fd8-c134-4283-a51b-5cbf50a85f48
# ╠═af1ac20a-e002-4de0-8a4d-9498e1343b03
# ╠═615af0e8-d1e8-4eb1-8177-cf4ae7469fd9
# ╠═faf75044-6a3a-47f9-8b54-81bbe6a2426a
# ╠═b729df36-39a0-47e6-8b32-b13e68008052
# ╠═a0ed51ef-481f-46fe-bcac-b74cf25a216a
# ╠═50b8ab07-f05a-40f3-9fff-b3d47ffe7744
# ╠═928d18b4-e0c0-41d6-8aa3-2b0239145a98
# ╠═17330e44-cd12-4fb1-a7b5-94fac18b563d
# ╠═d58dc85f-48ae-4de0-bbfd-69ef447af559
# ╠═4f88a6e3-0a93-458c-8a20-61fcd4bbd10e
# ╠═0ca030f4-f6e0-4e0f-a1ab-77bab03507a6
# ╠═7609d04e-77e4-4665-b4ac-bcc406a0a8ef
# ╠═1e60ed43-4109-4cb7-9623-3a63bc8bf790
# ╠═199ad7c1-7941-496b-ac9f-cfcfbb32b51e
