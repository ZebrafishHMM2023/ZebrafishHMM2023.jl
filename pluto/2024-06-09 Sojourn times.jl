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

# ╔═╡ ba1e4268-63c9-4cc8-b233-c088ecd1117a
using Statistics: mean

# ╔═╡ f6ce0fe7-d97a-4926-9151-607d12e05b44
using Statistics: var

# ╔═╡ 6d94d933-8409-47ec-9e42-9bb0816f0d2f
using Statistics: std

# ╔═╡ 71ee1b22-360b-44fa-a1d3-d03bdf2b13da
using Statistics: cov

# ╔═╡ 83bf9b96-f048-418a-b712-b240002f7f0f
using Statistics: cor

# ╔═╡ fbed3295-c9d4-4f08-800e-8e795c2599e9
using Statistics: middle

# ╔═╡ 86f6cf49-ac45-42dd-b974-e04ba90d4a8c
using StatsBase: corspearman

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

# ╔═╡ 17926ffb-8b9d-4d55-9bc6-12d1e564c530
import HDF5

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

	time_unit = mean(diff(data.time))

	return (; hmm, lL, times_F, times_L, times_R, time_unit)
end

# ╔═╡ c5c12295-e495-480e-899b-22745ad9f4e6
artr_hmms = Dict(
	(; temperature, fish) => artr_train_hmm(; temperature, fish)
	for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 314fc183-f07c-44b1-a93c-4bd761246ef1
md"# Save HMMs for Matteo"

# ╔═╡ b4430725-a9aa-45fa-a806-506074619c9a
for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(temperature)
	path = "/data/cossio/projects/2023/zebrafish_hmm/data/20240620_ARTR_HMMs_for_Matteo/artr_temperature=$temperature-fish=$fish.hdf5"
	HDF5.h5open(path, "w") do h5
		d = artr_hmms[(; temperature, fish)]
		write(h5, "type", "HMM_ARTR_Log")
		write(h5, "transition_matrix", float(d.hmm.transition_matrix))
		write(h5, "h", float(d.hmm.h))
		write(h5, "pinit", float(d.hmm.pinit))
		write(h5, "pseudocount", [d.hmm.pseudocount])

		write(h5, "sojourn_times_F", d.times_F)
		write(h5, "sojourn_times_L", d.times_L)
		write(h5, "sojourn_times_R", d.times_R)
		write(h5, "time_unit", d.time_unit)
	end
end

# ╔═╡ aa26f84c-49d3-4f14-bc32-e3c3858f4b39
md"# Comparison of sojourn times"

# ╔═╡ 9b77049b-9c3b-4ed8-838f-df958224763d
let fig = Makie.Figure()
	bins = 0:1:40
	for (i, T) = enumerate(artr_wolf_2023_temperatures())
		ylabel = i == 1 ? "Forw." : ""
		ax = Makie.Axis(fig[1,i]; width=200, height=150, yscale=log10, title="Temp. $T", xlabel="Sojourn time (sec.)", ylabel)
		Makie.stephist!(ax, swimming_hmms[T].times_F; bins, normalization=:pdf, label="Swim")
		Makie.stephist!(ax, [t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F]; bins, normalization=:pdf, label="ARTR")
		Makie.ylims!(ax, 1e-5, 1)

		ylabel = i == 1 ? "Turn" : ""
		ax = Makie.Axis(fig[2,i]; width=200, height=150, yscale=log10, xlabel="Sojourn time (sec.)", ylabel)
		Makie.stephist!(ax, [swimming_hmms[T].times_L; swimming_hmms[T].times_R]; bins, normalization=:pdf, label="Swim")
		Makie.stephist!(ax, [t for fish = artr_wolf_2023_fishes(T) for ts = [artr_hmms[(; temperature=T, fish)].times_L, artr_hmms[(; temperature=T, fish)].times_R] for t = ts]; bins, normalization=:pdf, label="ARTR")
		Makie.ylims!(ax, 1e-5, 1)

		if i == 5
			Makie.axislegend(ax, position=:rt)
		end
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ cc947aef-3e96-4d66-a374-209330b25f77
let fig = Makie.Figure()
	bins = 0:20
	for (i, T) = enumerate(artr_wolf_2023_temperatures())
		#λ = mean(swimming_hmms[T].times_F) / mean(t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F)
		λ = std(swimming_hmms[T].times_F) / std(t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F)

		ylabel = i == 1 ? "Forw." : ""
		ax = Makie.Axis(fig[1,i]; width=200, height=150, yscale=log10, title="Temp. $T", xlabel="Sojourn time (sec.)", ylabel)
		Makie.stephist!(ax, swimming_hmms[T].times_F; bins, normalization=:pdf, label="Swim")
		Makie.stephist!(ax, [λ * t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F]; bins, normalization=:pdf, label="ARTR")
		Makie.ylims!(ax, 1e-4, 2)

		ylabel = i == 1 ? "Turn" : ""
		ax = Makie.Axis(fig[2,i]; width=200, height=150, yscale=log10, xlabel="Sojourn time (sec.)", ylabel)
		Makie.stephist!(ax, swimming_hmms[T].times_L; bins, normalization=:pdf, label="Swim")
		Makie.stephist!(ax, [λ * t for fish = artr_wolf_2023_fishes(T) for ts = [artr_hmms[(; temperature=T, fish)].times_L, artr_hmms[(; temperature=T, fish)].times_R] for t = ts]; bins, normalization=:pdf, label="ARTR")
		Makie.ylims!(ax, 1e-4, 2)

		if i == 5
			Makie.axislegend(ax, position=:rt)
		end
	end
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 22f1ad7f-0e60-41c1-8d80-cb401f1002a0
times_F_swim = [t for T = artr_wolf_2023_temperatures() for t = swimming_hmms[T].times_F]

# ╔═╡ 408298dc-da4a-431e-8c13-581786f6bccb
times_F_artr = [t for T = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F]

# ╔═╡ cf4cf749-dcb4-4f23-80d3-c4e848ec8230
times_L_swim = [t for T = artr_wolf_2023_temperatures() for t = swimming_hmms[T].times_L]

# ╔═╡ 232a378f-f300-49e4-ae3f-8f3fc7511ba3
times_L_artr = [t for T = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_L]

# ╔═╡ e90dc938-34cf-4b1c-9ac5-424c0ba123b9
times_R_swim = [t for T = artr_wolf_2023_temperatures() for t = swimming_hmms[T].times_R]

# ╔═╡ f73ff171-dedd-4c0c-9ec3-c02143c0f4cd
times_R_artr = [t for T = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_R]

# ╔═╡ 9716ddea-4ddb-4893-85c7-f318751b1092
common_λ = std([times_F_artr; times_L_artr; times_R_artr]) / mean([times_F_swim; times_L_swim; times_R_swim])
#common_λ = 5

# ╔═╡ 9697aea0-0501-48ce-8a85-5e488395fa11
common_λ / 0.2

# ╔═╡ 3100cdac-66cc-4407-bb4b-698344d33986
let fig = Makie.Figure()
	bins = 0:50

	ax = Makie.Axis(fig[1,1]; width=250, height=200, yscale=log10, title="Forw.", xlabel="Sojourn time (sec.)", ylabel="Frequency")
	Makie.stephist!(ax, times_F_swim * common_λ; bins, normalization=:pdf, label="Swim")
	Makie.stephist!(ax, times_F_artr; bins, normalization=:pdf, label="ARTR")
	Makie.ylims!(ax, 1e-4, 2)

	ax = Makie.Axis(fig[1,2]; width=250, height=200, yscale=log10, title="Left", xlabel="Sojourn time (sec.)", ylabel="Frequency")
	Makie.stephist!(ax, times_L_swim * common_λ; bins, normalization=:pdf, label="Swim")
	Makie.stephist!(ax, times_L_artr; bins, normalization=:pdf, label="ARTR")
	Makie.ylims!(ax, 1e-4, 2)

	ax = Makie.Axis(fig[1,3]; width=250, height=200, yscale=log10, title="Right", xlabel="Sojourn time (sec.)", ylabel="Frequency")
	Makie.stephist!(ax, times_R_swim * common_λ; bins, normalization=:pdf, label="Swim")
	Makie.stephist!(ax, times_R_artr; bins, normalization=:pdf, label="ARTR")
	Makie.axislegend(ax)
	Makie.ylims!(ax, 1e-4, 2)

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 9e0f76ea-a3da-46f9-9bc2-7fa79b2e8b60
let fig = Makie.Figure()
	scaling_factors = [
		mean(t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F) / mean(swimming_hmms[T].times_F)
		for T = artr_wolf_2023_temperatures()
	]
	scaling_factors_std = [
		std(t for fish = artr_wolf_2023_fishes(T) for t = artr_hmms[(; temperature=T, fish)].times_F) / std(swimming_hmms[T].times_F)
		for T = artr_wolf_2023_temperatures()
	]

	ax = Makie.Axis(fig[1,1]; width=400, height=300, xlabel="Temperature", ylabel="Scaling factor (ARTR / Swim)", xticks=collect(artr_wolf_2023_temperatures()))	
	Makie.scatterlines!(ax, collect(artr_wolf_2023_temperatures()), scaling_factors, label="mean")
	Makie.scatterlines!(ax, collect(artr_wolf_2023_temperatures()), scaling_factors_std, label="std")
	Makie.axislegend(ax)
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 9a9c5dee-ac20-4828-a6ac-586a5fec69cf
artr_transition_matrices = Dict(
	temperature => mean([artr_hmms[(; temperature, fish)].hmm.transition_matrix for fish = artr_wolf_2023_fishes(temperature)]) for temperature = artr_wolf_2023_temperatures()
)

# ╔═╡ 258bac77-b65d-49c4-9c1a-bd397ad7ed6a
swimming_transition_matrices = Dict(
	temperature => swimming_hmms[temperature].hmm.transition_matrix for temperature = behaviour_free_swimming_temperatures()
)

# ╔═╡ 1ddca540-72db-43d2-bd80-3c64699029f2
[artr_transition_matrices[18]; artr_transition_matrices[22]]

# ╔═╡ cc268bdd-6c1b-4455-bcf9-7856e100a307
let fig = Makie.Figure()
	colors = [:blue, :teal, :green, :orange, :red]

	ax = Makie.Axis(fig[1,1], xlabel="ARTR", ylabel="Swim", width=400, height=400)

	all_swim_trans = Float64[]
	all_artr_trans = Float64[]
	
	for (i, T) = enumerate(artr_wolf_2023_temperatures())
		swim_trans = float(swimming_hmms[T].hmm.transition_matrix)
		artr_trans = mean(artr_hmms[(; temperature=T, fish)].hmm.transition_matrix for fish = artr_wolf_2023_fishes(T))

		append!(all_swim_trans, vec(swim_trans))
		append!(all_artr_trans, vec(artr_trans))
		
		Makie.scatter!(ax, vec(artr_trans), vec(swim_trans), color=colors[i])
	end
	Makie.xlims!(ax, 0, 1)
	Makie.ylims!(ax, 0, 1)
	Makie.text!(ax, 0.1, 0.9; text="Spearman corr. $(round(corspearman(all_swim_trans, all_artr_trans); digits=4))")


	ax = Makie.Axis(fig[1,2], xlabel="ARTR (one scale per temp.)", ylabel="Swim", width=400, height=400)

	all_swim_trans = Float64[]
	all_artr_trans = Float64[]

	for (i, T) = enumerate(artr_wolf_2023_temperatures())
		my_λ = (
			std(t for fish = artr_wolf_2023_fishes(T) for t = [artr_hmms[(; temperature=T, fish)].times_F; artr_hmms[(; temperature=T, fish)].times_L; artr_hmms[(; temperature=T, fish)].times_R]) /
			std([swimming_hmms[T].times_F; swimming_hmms[T].times_L; swimming_hmms[T].times_R])
		)
		
		swim_trans = float(swimming_hmms[T].hmm.transition_matrix)
		artr_trans = mean([float(artr_hmms[(; temperature=T, fish)].hmm.transition_matrix)^(my_λ / artr_hmms[(; temperature=T, fish)].time_unit) for fish = artr_wolf_2023_fishes(T)])

		append!(all_swim_trans, vec(real(swim_trans)))
		append!(all_artr_trans, vec(real(artr_trans)))

		Makie.lines!(ax, [0,1], [0,1], color=:gray)
		Makie.scatter!(ax, vec(real(artr_trans)), vec(real(swim_trans)), color=colors[i])
	end
	Makie.xlims!(ax, 0, 1)
	Makie.ylims!(ax, 0, 1)
	Makie.text!(ax, 0.1, 0.9; text="Spearman corr. $(round(corspearman(all_swim_trans, all_artr_trans); digits=4))")

	all_swim_trans = Float64[]
	all_artr_trans = Float64[]

	ax = Makie.Axis(fig[1,3], xlabel="ARTR (one scale for all temp.)", ylabel="Swim", width=400, height=400)
	for (i, T) = enumerate(artr_wolf_2023_temperatures())		
		swim_trans = float(swimming_hmms[T].hmm.transition_matrix)
		artr_trans = mean([
			float(artr_hmms[(; temperature=T, fish)].hmm.transition_matrix)^(common_λ / artr_hmms[(; temperature=T, fish)].time_unit) for fish = artr_wolf_2023_fishes(T)
		])

		append!(all_swim_trans, vec(real(swim_trans)))
		append!(all_artr_trans, vec(real(artr_trans)))

		Makie.lines!(ax, [0,1], [0,1], color=:gray)
		Makie.scatter!(ax, vec(real(artr_trans)), vec(real(swim_trans)), color=colors[i])
	end
	Makie.text!(ax, 0.1, 0.9; text="Spearman corr. $(round(corspearman(all_swim_trans, all_artr_trans); digits=4))")

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ Cell order:
# ╠═ae9a7803-3ddf-4fcc-9b15-6b970797b4cb
# ╠═bfb8b4b5-eee8-47a5-8f55-e23ad3ddbaa9
# ╠═4285c8bf-7a44-4c56-a86e-f15dafd67b4b
# ╠═d3731804-a255-42a7-bad2-31c834be5d7a
# ╠═5d9d1329-8faf-46f6-aac3-aee7bd9b8065
# ╠═17926ffb-8b9d-4d55-9bc6-12d1e564c530
# ╠═2bcdddad-4227-4c56-b452-cff43bec2d41
# ╠═76c368ee-0aa9-4d8e-b815-d006ec9ae3db
# ╠═ba1e4268-63c9-4cc8-b233-c088ecd1117a
# ╠═f6ce0fe7-d97a-4926-9151-607d12e05b44
# ╠═6d94d933-8409-47ec-9e42-9bb0816f0d2f
# ╠═71ee1b22-360b-44fa-a1d3-d03bdf2b13da
# ╠═83bf9b96-f048-418a-b712-b240002f7f0f
# ╠═fbed3295-c9d4-4f08-800e-8e795c2599e9
# ╠═86f6cf49-ac45-42dd-b974-e04ba90d4a8c
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
# ╠═314fc183-f07c-44b1-a93c-4bd761246ef1
# ╠═b4430725-a9aa-45fa-a806-506074619c9a
# ╠═aa26f84c-49d3-4f14-bc32-e3c3858f4b39
# ╠═9b77049b-9c3b-4ed8-838f-df958224763d
# ╠═cc947aef-3e96-4d66-a374-209330b25f77
# ╠═22f1ad7f-0e60-41c1-8d80-cb401f1002a0
# ╠═408298dc-da4a-431e-8c13-581786f6bccb
# ╠═cf4cf749-dcb4-4f23-80d3-c4e848ec8230
# ╠═232a378f-f300-49e4-ae3f-8f3fc7511ba3
# ╠═e90dc938-34cf-4b1c-9ac5-424c0ba123b9
# ╠═f73ff171-dedd-4c0c-9ec3-c02143c0f4cd
# ╠═9716ddea-4ddb-4893-85c7-f318751b1092
# ╠═9697aea0-0501-48ce-8a85-5e488395fa11
# ╠═3100cdac-66cc-4407-bb4b-698344d33986
# ╠═9e0f76ea-a3da-46f9-9bc2-7fa79b2e8b60
# ╠═9a9c5dee-ac20-4828-a6ac-586a5fec69cf
# ╠═258bac77-b65d-49c4-9c1a-bd397ad7ed6a
# ╠═1ddca540-72db-43d2-bd80-3c64699029f2
# ╠═cc268bdd-6c1b-4455-bcf9-7856e100a307
