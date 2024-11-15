### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ c7a4ca53-587c-4984-b712-b125e57a0aa8
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 30c525d5-1060-4a30-ab48-57944716e6fa
using DataFrames: DataFrame

# ╔═╡ b296c9fe-2a8b-462d-8227-0db4bf747610
using Statistics: mean

# ╔═╡ d2bb51fc-0e40-446f-bbe5-409c6fa0863f
using Statistics: std

# ╔═╡ c1ab6ce0-a442-4006-b24e-738d1e78a3b3
using Statistics: var

# ╔═╡ 5f323e93-8bc6-4f08-b2b8-f4b53060dfc6
using Distributions: Gamma

# ╔═╡ 1f13bab2-89ac-4371-8a4d-021e03a909db
using Distributions: Exponential

# ╔═╡ 8d468b7c-01b3-4dbd-8754-bcd35ee7ef81
using DelimitedFiles: writedlm

# ╔═╡ 33f41862-1247-44e1-9179-869d207db0fa
using Random: shuffle

# ╔═╡ 628177f1-0336-4b1b-9b33-0ef8381b7db4
md"# Imports"

# ╔═╡ d24d1a90-94cc-431c-8fce-c0f7e048b645
import ZebrafishHMM2023

# ╔═╡ 61d2a5cf-c855-4376-a464-b5ad721b5d0c
import HiddenMarkovModels

# ╔═╡ ffa80dd6-fc8c-451f-a477-038b0bf4257f
import Makie

# ╔═╡ a006c2c5-40fd-4561-b7a6-3a89a7fcd0d2
import CairoMakie

# ╔═╡ 726e2170-6926-44eb-8658-56169422ead3
import PlutoUI

# ╔═╡ 19809d1f-3527-41bf-a074-6e3af37af6fe
import CSV

# ╔═╡ f7aad8b6-83ce-4058-bbb6-680396ffe320
import HDF5

# ╔═╡ 95d062ff-5a77-45b9-99ce-a01ee3fe36cb
PlutoUI.TableOfContents()

# ╔═╡ 3e230ebc-73a3-4f50-bc78-59fa3b8552b5
md"# Train ARTR HMMs"

# ╔═╡ e8b3dc72-499d-4802-aa95-2c73ffbf4eb8
artr_data = Dict((; temperature, fish) =>
	ZebrafishHMM2023.load_artr_wolf_2023(; temperature, fish)
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ afac2c58-130b-4dd8-a1d0-379104f7ff7d
artr_hmms = Dict((; temperature, fish) => 
	first(ZebrafishHMM2023.easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ a23a808c-0fac-486e-b61f-377586af1899
md"# MSR functions"

# ╔═╡ 0bd4b8a4-dc0f-449b-b292-c833e4ccb4ec
function mean_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
	#std_trajs = [(traj .- mean(traj)) / std(traj)) for traj = trajs]
	#std_trajs = [traj .- mean(traj) for traj = trajs]
	std_trajs = trajs
    return mean(abs2(sum(traj[i:(i + q)])) for traj = std_trajs for i = 1:length(traj) - q)
end

# ╔═╡ 168c08a2-577f-4777-af07-186b6266bd4f
function std_MSR(trajs::AbstractVector{<:AbstractVector}, q::Int)
	#std_trajs = [(traj .- mean(traj)) / std(traj) for traj = trajs]
	#std_trajs = [traj .- mean(traj) for traj = trajs]
	std_trajs = trajs
    return std(abs2(sum(traj[i:(i + q)])) for traj = trajs for i = 1:length(traj) - q) / sqrt(sum(length(traj) - q for traj = std_trajs))
end

# ╔═╡ ae2b751b-81f0-4433-8b61-072a57875d81
md"# MSD functions -- mean-squared displacement"

# ╔═╡ a5c5ce96-5c61-4bf0-8956-059717bfef65
function mean_MSD(xs::AbstractVector{<:AbstractVector{Float64}}, ys::AbstractVector{<:AbstractVector{Float64}}, q::Int)
	@assert length(xs) == length(ys)
	for (x, y) = zip(xs, ys)
		@assert length(x) == length(y)
	end
	return mean(sum(x[i:(i + q)])^2 + sum(y[i:(i + q)])^2 for (x, y) = zip(xs, ys) for i = 1:length(x) - q)
end

# ╔═╡ b18ec775-3df6-4751-9b12-77f2fc56b165
md"# MSR Plots"

# ╔═╡ 135f350e-65c3-44f4-a681-553c6d785eef
_tmpdir = "/tmp/jl_fUcfzS"

# ╔═╡ 28c7dd4b-e51d-43bb-843d-f11f5cac1f06
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 10

	selected_temperatures = ZebrafishHMM2023.artr_wolf_2023_temperatures()[[1,3,5]]

	MSR_for_bar_plot_hmm = [Float64[] for temperature = selected_temperatures]
	MSR_for_bar_plot_dat = [0.0 for temperature = selected_temperatures]
	MSR_for_bar_plot_dat_shuffled = [0.0 for temperature = selected_temperatures]
	MSR_for_bar_plot_dat_long = [0.0 for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]
	
	for (col, temperature) = enumerate(selected_temperatures)
		ax = Makie.Axis(fig[1,col], width=150, height=200, title="T = $temperature C", xgridvisible=false, ygridvisible=false, xlabel="Streak length", ylabel="MSR")
		
		if col == 1
			Makie.hidespines!(ax, :t, :r)
		else
			Makie.hidespines!(ax, :l, :t, :r)
			Makie.hideydecorations!(ax)
		end

		if temperature == 26
			for (n, fish) = enumerate(ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())
				δθ_dat = [[obs.θ for obs = traj] for traj = fish]

				# normalize
				μ = mean(δθ for traj = δθ_dat for δθ = traj)
				δθ_dat = [[δθ - μ for δθ = traj] for traj = δθ_dat]
				
				msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]

				if n == 1
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="long")
				else
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=0.5)
				end

				MSR_for_bar_plot_dat_long[n] = msr[q_for_barplot]
			end
		end

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]

			# normalize
			μ = mean(θ for traj = δθ_hmm for θ = traj)
			δθ_hmm = [[θ - μ for θ = traj] for traj = δθ_hmm]
			
			msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
			
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1)
			end
			
			push!(MSR_for_bar_plot_hmm[col], msr[q_for_barplot])
		end

		δθ_dat = [[obs.θ for obs = traj] for traj = ZebrafishHMM2023.load_full_obs(temperature)]
		
		# normalize
		μ = mean(θ for traj = δθ_dat for θ = traj)
		δθ_dat = [[θ - μ for θ = traj] for traj = δθ_dat]
		
		shuffled_δθ_dat = [rand([θ for t = δθ_dat for θ = t], length(traj)) for traj = δθ_dat]
		
		msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
		msr_shuffled = [mean_MSR(shuffled_δθ_dat, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, msr_shuffled, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		Makie.ylims!(ax, 0, 4e4)
		MSR_for_bar_plot_dat[col] = msr[q_for_barplot]
		MSR_for_bar_plot_dat_shuffled[col] = msr_shuffled[q_for_barplot]
		
		if col == 5
			#Makie.axislegend(ax; position=:lt, framevisible=false)
		end
	end

	ax = Makie.Axis(fig[1,4], width=150, height=200, xgridvisible=false, ygridvisible=false, xticks=(1:length(selected_temperatures), ["T=$T" for T = selected_temperatures]), xlabel="Temperature (C)")
	Makie.barplot!(ax, 1:3, map(mean, MSR_for_bar_plot_hmm); color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:length(selected_temperatures), map(mean, MSR_for_bar_plot_hmm), map(std, MSR_for_bar_plot_hmm); #./ map(sqrt ∘ length, MSR_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5
	)
	for t = 1:length(selected_temperatures)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat[t:t]; color=:black, linewidth=4, label = t == 1 ? "data" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=(:dash, :dense), label = t == 1 ? "shuffled" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], map(mean, MSR_for_bar_plot_hmm)[t:t]; color=:blue, linewidth=2, label = t == 1 ? "N-HMM" : nothing)
	end
	#Makie.lines!(ax, [0.5, length(selected_temperatures) + 0.5], fill(mean(MSR_for_bar_plot_dat_long), 2); color=:gray, linewidth=1, linestyle=:dash)
	Makie.lines!(ax, [2 - 0.3, 2 + 0.3], fill(mean(MSR_for_bar_plot_dat_long), 2); color=:gray, linewidth=4, label="long traj.")
	Makie.errorbars!(
		ax, 2.1:2.1, [mean(MSR_for_bar_plot_dat_long)], [std(MSR_for_bar_plot_dat_long)];
		color=:gray, whiskerwidth=3
	)
	Makie.ylims!(ax, 0, 4e4)
	Makie.hidespines!(ax, :l, :t, :r)
	Makie.hideydecorations!(ax)

	fig[1,5] = Makie.Legend(fig, ax, "Legend"; framevisible=false)

	Makie.resize_to_layout!(fig)
	#Makie.save()
	fig
end

# ╔═╡ 45889da9-16b1-497b-8a8a-91060c04fa1b
_tmpfigdir = mktempdir()

# ╔═╡ a1ad569e-a7a9-4929-bef5-199e6a2d379d
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 15

	MSR_for_bar_plot_hmm = [Float64[] for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSR_for_bar_plot_dat = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSR_for_bar_plot_dat_shuffled = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]

	MSR_for_bar_plot_dat_long = [0.0 for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]
	
	for (col, temperature) = enumerate(ZebrafishHMM2023.artr_wolf_2023_temperatures())
		ax = Makie.Axis(fig[1,col], width=120, height=150, title="temperature = $temperature", xgridvisible=false, ygridvisible=false, xlabel="Streak length", ylabel="std. MSR")

		if col == 1
			Makie.hidespines!(ax, :t, :r)
		else
			Makie.hidespines!(ax, :l, :t, :r)
			Makie.hideydecorations!(ax)
		end

		if temperature == 26
			for (n, fish) = enumerate(ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())
				δθ_dat = [[obs.θ for obs = traj] for traj = fish]
		
				# normalize
				μ = mean(δθ for traj = δθ_dat for δθ = traj)
				σ = std(δθ for traj = δθ_dat for δθ = traj)
				δθ_dat = [[(δθ - μ) / σ for δθ = traj] for traj = δθ_dat]
				
				msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]

				if n == 1
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="long")
				else
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1)
				end

				MSR_for_bar_plot_dat_long[n] = msr[q_for_barplot]
			end
		end

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
			
			# normalize
			μ = mean(θ for traj = δθ_hmm for θ = traj)
			σ = std(θ for traj = δθ_hmm for θ = traj)
			δθ_hmm = [[(θ - μ) / σ for θ = traj] for traj = δθ_hmm]
			
			msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
			
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1)
			end
			
			push!(MSR_for_bar_plot_hmm[col], msr[q_for_barplot])
		end

		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]

		# normalize
		μ = mean(θ for traj = δθ_dat for θ = traj)
		σ = std(θ for traj = δθ_dat for θ = traj)
		δθ_dat = [[(θ - μ) / σ for θ = traj] for traj = δθ_dat]
		
		shuffled_δθ_dat = [rand(traj, length(traj)) for traj = δθ_dat]
		
		msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
		msr_shuffled = [mean_MSR(shuffled_δθ_dat, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, msr_shuffled, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.lines!(ax, 0:qmax, 0:qmax, color=:red, linewidth=1, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		Makie.ylims!(ax, 0, 50)
		MSR_for_bar_plot_dat[col] = msr[q_for_barplot]
		MSR_for_bar_plot_dat_shuffled[col] = msr_shuffled[q_for_barplot]
		
		if col == 1
			#Makie.axislegend(ax; position=:lt, framevisible=false)
		end
	end

	ax = Makie.Axis(fig[1,6], width=200, height=150, xgridvisible=false, ygridvisible=false, xticks=(1:5, ["T=$T" for T = [18, 22, 26, 30, 33]]), xlabel="Temperature (C)", ylabel="std. MSR")
	Makie.barplot!(ax, 1:5, map(mean, MSR_for_bar_plot_hmm); color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:5, map(mean, MSR_for_bar_plot_hmm), map(std, MSR_for_bar_plot_hmm);# ./ map(sqrt ∘ length, MSR_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5, label="HMM"
	)
	for t = 1:5
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat[t:t]; color=:black, linewidth=4, label = t == 1 ? "data" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=:dash, label = t == 1 ? "shuffled" : nothing)
	end
	
	Makie.lines!(ax, [3 - 0.3, 3 + 0.3], fill(mean(MSR_for_bar_plot_dat_long), 2); color=:gray, linewidth=4, label="long traj.")
	Makie.errorbars!(
		ax, 3:3, [mean(MSR_for_bar_plot_dat_long)], [std(MSR_for_bar_plot_dat_long)];
		color=:gray, whiskerwidth=3
	)

	Makie.ylims!(ax, 0, 50)
	Makie.hidespines!(ax, :l, :t, :r)
	Makie.hideydecorations!(ax)
	
	fig[1,7] = Makie.Legend(fig, ax, "Legend"; framevisible=false)

	Makie.resize_to_layout!(fig)
	Makie.save(joinpath(_tmpfigdir, "MSR_suppl_fig_normalized.pdf"), fig)
	fig
end

# ╔═╡ 89217b13-9045-41d2-b2ec-6dd831ba7bd8
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 10

	selected_temperatures = ZebrafishHMM2023.artr_wolf_2023_temperatures()

	MSR_for_bar_plot_hmm = [Float64[] for temperature = selected_temperatures]
	MSR_for_bar_plot_dat = [0.0 for temperature = selected_temperatures]
	MSR_for_bar_plot_dat_shuffled = [0.0 for temperature = selected_temperatures]
	MSR_for_bar_plot_dat_long = [0.0 for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]
	
	for (col, temperature) = enumerate(selected_temperatures)
		ax = Makie.Axis(fig[1,col], width=150, height=200, title="T = $temperature C", xgridvisible=false, ygridvisible=false, xlabel="Streak length", ylabel="MSR")
		
		if col == 1
			Makie.hidespines!(ax, :t, :r)
		else
			Makie.hidespines!(ax, :l, :t, :r)
			Makie.hideydecorations!(ax)
		end

		if temperature == 26
			for (n, fish) = enumerate(ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())
				δθ_dat = [[obs.θ for obs = traj] for traj = fish]

				# normalize
				μ = mean(δθ for traj = δθ_dat for δθ = traj)
				δθ_dat = [[δθ - μ for δθ = traj] for traj = δθ_dat]
				
				msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]

				if n == 1
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="long")
				else
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=0.5)
				end

				MSR_for_bar_plot_dat_long[n] = msr[q_for_barplot]
			end
		end

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]

			# normalize
			μ = mean(θ for traj = δθ_hmm for θ = traj)
			δθ_hmm = [[θ - μ for θ = traj] for traj = δθ_hmm]
			
			msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
			
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1)
			end
			
			push!(MSR_for_bar_plot_hmm[col], msr[q_for_barplot])
		end

		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		
		# normalize
		μ = mean(θ for traj = δθ_dat for θ = traj)
		δθ_dat = [[θ - μ for θ = traj] for traj = δθ_dat]
		
		#shuffled_δθ_dat = [rand(traj, length(traj)) for traj = δθ_dat]
		shuffled_δθ_dat = [rand([θ for t = δθ_dat for θ = t], length(traj)) for traj = δθ_dat]
		
		msr = [mean_MSR(δθ_dat, q) for q = 0:qmax]
		msr_shuffled = [mean_MSR(shuffled_δθ_dat, q) for q = 0:qmax]
		Makie.lines!(ax, 0:qmax, msr, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, msr_shuffled, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		Makie.ylims!(ax, 0, 4e4)
		MSR_for_bar_plot_dat[col] = msr[q_for_barplot]
		MSR_for_bar_plot_dat_shuffled[col] = msr_shuffled[q_for_barplot]
		
		if col == 5
			#Makie.axislegend(ax; position=:lt, framevisible=false)
		end
	end

	ax = Makie.Axis(fig[1,6], width=200, height=200, xgridvisible=false, ygridvisible=false, xticks=(1:length(selected_temperatures), ["T=$T" for T = selected_temperatures]), xlabel="Temperature (C)")
	Makie.barplot!(ax, 1:length(selected_temperatures), map(mean, MSR_for_bar_plot_hmm); color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:length(selected_temperatures), map(mean, MSR_for_bar_plot_hmm), map(std, MSR_for_bar_plot_hmm); #./ map(sqrt ∘ length, MSR_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5
	)
	for t = 1:length(selected_temperatures)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat[t:t]; color=:black, linewidth=4, label = t == 1 ? "data" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSR_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=(:dash, :dense), label = t == 1 ? "shuffled" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], map(mean, MSR_for_bar_plot_hmm)[t:t]; color=:blue, linewidth=2, label = t == 1 ? "N-HMM" : nothing)
	end
	#Makie.lines!(ax, [0.5, length(selected_temperatures) + 0.5], fill(mean(MSR_for_bar_plot_dat_long), 2); color=:gray, linewidth=1, linestyle=:dash)
	Makie.lines!(ax, [3 - 0.3, 3 + 0.3], fill(mean(MSR_for_bar_plot_dat_long), 2); color=:gray, linewidth=4, label="long traj.")
	Makie.errorbars!(
		ax, 3.1:3.1, [mean(MSR_for_bar_plot_dat_long)], [std(MSR_for_bar_plot_dat_long)];
		color=:gray, whiskerwidth=3
	)
	Makie.ylims!(ax, 0, 4e4)
	Makie.hidespines!(ax, :l, :t, :r)
	Makie.hideydecorations!(ax)

	fig[1,7] = Makie.Legend(fig, ax, "Legend"; framevisible=false)

	Makie.resize_to_layout!(fig)
	Makie.save(joinpath(_tmpfigdir, "MSR_suppl_fig.pdf"), fig)
	fig
end

# ╔═╡ 28f2ce11-5be3-4fce-868b-8ce955efdf0a
md"# MSD Plots"

# ╔═╡ 4b8eb55f-9091-4b9f-9537-583ab20cfee8
cumsum([1,2,3])

# ╔═╡ ee07aae7-bc94-40e7-bfbc-9ee6bd8823fa
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 10

	MSD_for_bar_plot_hmm = [Float64[] for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat_shuffled = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat_long = [0.0 for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]
	
	for (col, temperature) = enumerate(ZebrafishHMM2023.artr_wolf_2023_temperatures())
		ax = Makie.Axis(fig[1,col], width=150, height=150, title="T = $temperature C", xgridvisible=false, ygridvisible=false, xlabel="Streak length", ylabel="MSD")
		
		if col == 1
			Makie.hidespines!(ax, :t, :r)
		else
			Makie.hidespines!(ax, :l, :t, :r)
			Makie.hideydecorations!(ax)
		end

		if temperature == 26
			for (n, fish) = enumerate(ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())
				δθ_dat = [[obs.θ for obs = traj] for traj = fish]
				δd_dat = [[obs.d for obs = traj] for traj = fish]
				δx_dat = [[d * cos(θ) for (θ, d) = zip(cumsum(δθ_dat[t]), δd_dat[t])] for t = eachindex(fish)]
				δy_dat = [[d * sin(θ) for (θ, d) = zip(cumsum(δθ_dat[t]), δd_dat[t])] for t = eachindex(fish)]

				# normalize
				μx = mean(δx for traj = δx_dat for δx = traj)
				μy = mean(δy for traj = δy_dat for δy = traj)
				δx_dat = [[δx - μx for δx = traj] for traj = δx_dat]
				δy_dat = [[δy - μy for δy = traj] for traj = δy_dat]
				
				msr = [mean_MSD(δx_dat, δy_dat, q) for q = 0:qmax]

				if n == 1
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="long")
				else
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=0.5)
				end

				MSD_for_bar_plot_dat_long[n] = msr[q_for_barplot]
			end
		end

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
			δd_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_dist for rep = 1:100]
			δx_hmm = [[d * cos(θ) for (θ, d) = zip(cumsum(δθ_hmm[t]), δd_hmm[t])] for t = 1:100]
			δy_hmm = [[d * sin(θ) for (θ, d) = zip(cumsum(δθ_hmm[t]), δd_hmm[t])] for t = 1:100]

			# normalize
			μx = mean(δx for traj = δx_hmm for δx = traj)
			μy = mean(δy for traj = δy_hmm for δy = traj)
			δx_hmm = [[δx - μx for δx = traj] for traj = δx_hmm]
			δy_hmm = [[δy - μy for δy = traj] for traj = δy_hmm]
			
			msr = [mean_MSD(δx_hmm, δy_hmm, q) for q = 0:qmax]
			
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1)
			end
			
			push!(MSD_for_bar_plot_hmm[col], msr[q_for_barplot])
		end

		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		δd_dat = [[obs.d for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		δx_dat = [[d * cos(θ) for (θ, d) = zip(cumsum(δθ_dat[t]), δd_dat[t])] for t = eachindex(δθ_dat)]
		δy_dat = [[d * sin(θ) for (θ, d) = zip(cumsum(δθ_dat[t]), δd_dat[t])] for t = eachindex(δθ_dat)]

		# normalize
		μx = mean(δx for traj = δx_dat for δx = traj)
		μy = mean(δy for traj = δy_dat for δy = traj)
		δx_dat = [[δx - μx for δx = traj] for traj = δx_dat]
		δy_dat = [[δy - μy for δy = traj] for traj = δy_dat]
		
		shuffled_δx_dat = [rand([δx for traj = δx_dat for δx = traj], length(traj)) for traj = δx_dat]
		shuffled_δy_dat = [rand([δy for traj = δy_dat for δy = traj], length(traj)) for traj = δy_dat]
		
		msr = [mean_MSD(δx_dat, δy_dat, q) for q = 0:qmax]
		msr_shuffled = [mean_MSD(shuffled_δx_dat, shuffled_δy_dat, q) for q = 0:qmax]
		
		Makie.lines!(ax, 0:qmax, msr, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, msr_shuffled, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		#Makie.ylims!(ax, 0, 4e4)
		MSD_for_bar_plot_dat[col] = msr[q_for_barplot]
		MSD_for_bar_plot_dat_shuffled[col] = msr_shuffled[q_for_barplot]
		
		if col == 5
			#Makie.axislegend(ax; position=:lt, framevisible=false)
		end
	end

	ax = Makie.Axis(fig[1,6], width=2 * 150, height=150, xgridvisible=false, ygridvisible=false, xticks=(1:5, ["T=$T" for T = [18, 22, 26, 30, 33]]), xlabel="Temperature (C)")
	Makie.barplot!(ax, 1:5, map(mean, MSD_for_bar_plot_hmm); color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:5, map(mean, MSD_for_bar_plot_hmm), map(std, MSD_for_bar_plot_hmm) ./ map(sqrt ∘ length, MSD_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5
	)
	for t = 1:5
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSD_for_bar_plot_dat[t:t]; color=:black, linewidth=4, label = t == 1 ? "data" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSD_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=:dash, label = t == 1 ? "shuffled" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], map(mean, MSD_for_bar_plot_hmm)[t:t]; color=:blue, linewidth=2, label = t == 1 ? "HMM" : nothing)
	end
	Makie.lines!(ax, [0.5, 5.5], fill(mean(MSD_for_bar_plot_dat_long), 2); color=:gray, linewidth=1, linestyle=:dash)
	Makie.lines!(ax, [3 - 0.3, 3 + 0.3], fill(mean(MSD_for_bar_plot_dat_long), 2); color=:gray, linewidth=4, label="long traj.")
	Makie.errorbars!(
		ax, 3:3, [mean(MSD_for_bar_plot_dat_long)], [std(MSD_for_bar_plot_dat_long)];
		color=:gray, whiskerwidth=3
	)
	#Makie.ylims!(ax, 0, 4e4)
	Makie.hidespines!(ax, :l, :t, :r)
	Makie.hideydecorations!(ax)

	fig[1,7] = Makie.Legend(fig, ax, "Legend"; framevisible=false)

	Makie.resize_to_layout!(fig)
	#Makie.save()
	fig
end

# ╔═╡ ceb10287-1340-4834-b4c9-11d37f1eec01
let fig = Makie.Figure()
	qmax = 15

	q_for_barplot = 10

	MSD_for_bar_plot_hmm = [Float64[] for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat_shuffled = [0.0 for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()]
	MSD_for_bar_plot_dat_long = [0.0 for fish = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]
	
	for (col, temperature) = enumerate(ZebrafishHMM2023.artr_wolf_2023_temperatures())
		ax = Makie.Axis(fig[1,col], width=150, height=150, title="T = $temperature C", xgridvisible=false, ygridvisible=false, xlabel="Streak length", ylabel="MSD")
		
		if col == 1
			Makie.hidespines!(ax, :t, :r)
		else
			Makie.hidespines!(ax, :l, :t, :r)
			Makie.hideydecorations!(ax)
		end

		if temperature == 26
			for (n, fish) = enumerate(ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs())
				δθ_dat = [[obs.θ for obs = traj] for traj = fish]
				δd_dat = [[obs.d for obs = traj] for traj = fish]
				δx_dat = [[d * cos(θ) for (θ, d) = zip(δθ_dat[t], δd_dat[t])] for t = eachindex(fish)]
				δy_dat = [[d * sin(θ) for (θ, d) = zip(δθ_dat[t], δd_dat[t])] for t = eachindex(fish)]

				# normalize
				μx = mean(δx for traj = δx_dat for δx = traj)
				μy = mean(δy for traj = δy_dat for δy = traj)
				σx = std(δx for traj = δx_dat for δx = traj)
				σy = std(δy for traj = δy_dat for δy = traj)
				δx_dat = [[(δx - μx) / σx for δx = traj] for traj = δx_dat]
				δy_dat = [[(δy - μy) / σy for δy = traj] for traj = δy_dat]
				
				msr = [mean_MSD(δx_dat, δy_dat, q) for q = 0:qmax]

				if n == 1
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=1, label="long")
				else
					Makie.lines!(ax, 0:qmax, msr, color=:gray, linewidth=0.5)
				end

				MSD_for_bar_plot_dat_long[n] = msr[q_for_barplot]
			end
		end

		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]
			δd_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_dist for rep = 1:100]
			δx_hmm = [[d * cos(θ) for (θ, d) = zip(δθ_hmm[t], δd_hmm[t])] for t = 1:100]
			δy_hmm = [[d * sin(θ) for (θ, d) = zip(δθ_hmm[t], δd_hmm[t])] for t = 1:100]

			# normalize
			μx = mean(δx for traj = δx_hmm for δx = traj)
			μy = mean(δy for traj = δy_hmm for δy = traj)
			σx = std(δx for traj = δx_hmm for δx = traj)
			σy = std(δy for traj = δy_hmm for δy = traj)
			δx_hmm = [[(δx - μx) / σx for δx = traj] for traj = δx_hmm]
			δy_hmm = [[(δy - μy) / σy for δy = traj] for traj = δy_hmm]
			
			msr = [mean_MSD(δx_hmm, δy_hmm, q) for q = 0:qmax]
			
			if fish == first(ZebrafishHMM2023.artr_wolf_2023_fishes(temperature))
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1, label="HMM")
			else
				Makie.lines!(ax, 0:qmax, msr, color=:blue, linewidth=1)
			end
			
			push!(MSD_for_bar_plot_hmm[col], msr[q_for_barplot])
		end

		δθ_dat = [[obs.θ for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		δd_dat = [[obs.d for traj = ZebrafishHMM2023.load_full_obs(temperature) for obs = traj]]
		δx_dat = [[d * cos(θ) for (θ, d) = zip(δθ_dat[t], δd_dat[t])] for t = eachindex(δθ_dat)]
		δy_dat = [[d * sin(θ) for (θ, d) = zip(δθ_dat[t], δd_dat[t])] for t = eachindex(δθ_dat)]

		# normalize
		μx = mean(δx for traj = δx_dat for δx = traj)
		μy = mean(δy for traj = δy_dat for δy = traj)
		σx = std(δx for traj = δx_dat for δx = traj)
		σy = std(δy for traj = δy_dat for δy = traj)
		δx_dat = [[(δx - μx) / σx for δx = traj] for traj = δx_dat]
		δy_dat = [[(δy - μy) / σy for δy = traj] for traj = δy_dat]
		
		shuffled_δx_dat = [rand([δx for traj = δx_dat for δx = traj], length(traj)) for traj = δx_dat]
		shuffled_δy_dat = [rand([δy for traj = δy_dat for δy = traj], length(traj)) for traj = δy_dat]
		
		msr = [mean_MSD(δx_dat, δy_dat, q) for q = 0:qmax]
		msr_shuffled = [mean_MSD(shuffled_δx_dat, shuffled_δy_dat, q) for q = 0:qmax]
		
		Makie.lines!(ax, 0:qmax, msr, color=:black, label="Data", linewidth=3)
		Makie.lines!(ax, 0:qmax, msr_shuffled, color=:red, label="Data shuffled", linewidth=3, linestyle=:dash)
		Makie.xlims!(ax, 0, 15)
		#Makie.ylims!(ax, 0, 4e4)
		MSD_for_bar_plot_dat[col] = msr[q_for_barplot]
		MSD_for_bar_plot_dat_shuffled[col] = msr_shuffled[q_for_barplot]
		
		if col == 5
			#Makie.axislegend(ax; position=:lt, framevisible=false)
		end
	end

	ax = Makie.Axis(fig[1,6], width=2 * 150, height=150, xgridvisible=false, ygridvisible=false, xticks=(1:5, ["T=$T" for T = [18, 22, 26, 30, 33]]), xlabel="Temperature (C)")
	Makie.barplot!(ax, 1:5, map(mean, MSD_for_bar_plot_hmm); color=:lightblue, gap=0.4)
	Makie.errorbars!(
		ax, 1:5, map(mean, MSD_for_bar_plot_hmm), map(std, MSD_for_bar_plot_hmm) ./ map(sqrt ∘ length, MSD_for_bar_plot_hmm);
		color=:blue, whiskerwidth=5
	)
	for t = 1:5
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSD_for_bar_plot_dat[t:t]; color=:black, linewidth=4, label = t == 1 ? "data" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], MSD_for_bar_plot_dat_shuffled[t:t]; color=:red, linewidth=4, linestyle=:dash, label = t == 1 ? "shuffled" : nothing)
		Makie.lines!(ax, [t - 0.3, t + 0.3], map(mean, MSD_for_bar_plot_hmm)[t:t]; color=:blue, linewidth=2, label = t == 1 ? "HMM" : nothing)
	end
	Makie.lines!(ax, [0.5, 5.5], fill(mean(MSD_for_bar_plot_dat_long), 2); color=:gray, linewidth=1, linestyle=:dash)
	Makie.lines!(ax, [3 - 0.3, 3 + 0.3], fill(mean(MSD_for_bar_plot_dat_long), 2); color=:gray, linewidth=4, label="long traj.")
	Makie.errorbars!(
		ax, 3:3, [mean(MSD_for_bar_plot_dat_long)], [std(MSD_for_bar_plot_dat_long)];
		color=:gray, whiskerwidth=3
	)
	#Makie.ylims!(ax, 0, 4e4)
	Makie.hidespines!(ax, :l, :t, :r)
	Makie.hideydecorations!(ax)

	fig[1,7] = Makie.Legend(fig, ax, "Legend"; framevisible=false)

	Makie.resize_to_layout!(fig)
	#Makie.save()
	fig
end

# ╔═╡ b88952d7-9194-45a1-964a-4609797aa88c
md"# Save MSR for Matteo"

# ╔═╡ b65c8cd2-c4a7-4ec9-8465-44bd8c5f7b0d
_temp_dir_msr_for_matteo = mktempdir()

# ╔═╡ 205970d5-6f7e-49a7-8ab2-6fc0932c4b8c
let qmax = 15
	selected_temperatures = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for (col, temperature) = enumerate(selected_temperatures)
		for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
			δθ_hmm = [CSV.read(joinpath(_tmpdir, "temperature=$temperature-fish=$fish-rep=$rep.csv"), DataFrame).bout_angle for rep = 1:100]

			# normalize
			μ = mean(θ for traj = δθ_hmm for θ = traj)
			δθ_hmm = [[θ - μ for θ = traj] for traj = δθ_hmm]
			
			msr = [mean_MSR(δθ_hmm, q) for q = 0:qmax]
						
			writedlm(joinpath(_temp_dir_msr_for_matteo, "msr-N-HMM-gen-temperature=$temperature-fish=$fish.txt"), [0:qmax msr])
		end
	end
end

# ╔═╡ Cell order:
# ╠═628177f1-0336-4b1b-9b33-0ef8381b7db4
# ╠═c7a4ca53-587c-4984-b712-b125e57a0aa8
# ╠═d24d1a90-94cc-431c-8fce-c0f7e048b645
# ╠═61d2a5cf-c855-4376-a464-b5ad721b5d0c
# ╠═ffa80dd6-fc8c-451f-a477-038b0bf4257f
# ╠═a006c2c5-40fd-4561-b7a6-3a89a7fcd0d2
# ╠═726e2170-6926-44eb-8658-56169422ead3
# ╠═19809d1f-3527-41bf-a074-6e3af37af6fe
# ╠═f7aad8b6-83ce-4058-bbb6-680396ffe320
# ╠═30c525d5-1060-4a30-ab48-57944716e6fa
# ╠═b296c9fe-2a8b-462d-8227-0db4bf747610
# ╠═d2bb51fc-0e40-446f-bbe5-409c6fa0863f
# ╠═c1ab6ce0-a442-4006-b24e-738d1e78a3b3
# ╠═5f323e93-8bc6-4f08-b2b8-f4b53060dfc6
# ╠═1f13bab2-89ac-4371-8a4d-021e03a909db
# ╠═8d468b7c-01b3-4dbd-8754-bcd35ee7ef81
# ╠═33f41862-1247-44e1-9179-869d207db0fa
# ╠═95d062ff-5a77-45b9-99ce-a01ee3fe36cb
# ╠═3e230ebc-73a3-4f50-bc78-59fa3b8552b5
# ╠═e8b3dc72-499d-4802-aa95-2c73ffbf4eb8
# ╠═afac2c58-130b-4dd8-a1d0-379104f7ff7d
# ╠═a23a808c-0fac-486e-b61f-377586af1899
# ╠═0bd4b8a4-dc0f-449b-b292-c833e4ccb4ec
# ╠═168c08a2-577f-4777-af07-186b6266bd4f
# ╠═ae2b751b-81f0-4433-8b61-072a57875d81
# ╠═a5c5ce96-5c61-4bf0-8956-059717bfef65
# ╠═b18ec775-3df6-4751-9b12-77f2fc56b165
# ╠═135f350e-65c3-44f4-a681-553c6d785eef
# ╠═a1ad569e-a7a9-4929-bef5-199e6a2d379d
# ╠═28c7dd4b-e51d-43bb-843d-f11f5cac1f06
# ╠═45889da9-16b1-497b-8a8a-91060c04fa1b
# ╠═89217b13-9045-41d2-b2ec-6dd831ba7bd8
# ╠═28f2ce11-5be3-4fce-868b-8ce955efdf0a
# ╠═4b8eb55f-9091-4b9f-9537-583ab20cfee8
# ╠═ee07aae7-bc94-40e7-bfbc-9ee6bd8823fa
# ╠═ceb10287-1340-4834-b4c9-11d37f1eec01
# ╠═b88952d7-9194-45a1-964a-4609797aa88c
# ╠═b65c8cd2-c4a7-4ec9-8465-44bd8c5f7b0d
# ╠═205970d5-6f7e-49a7-8ab2-6fc0932c4b8c
