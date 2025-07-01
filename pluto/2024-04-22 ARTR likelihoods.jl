### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ c61f5cd6-023f-11ef-343e-6b895c2b8f54
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 4ec29bd5-35ca-4cf2-8305-148816cb2f5f
using HiddenMarkovModels: logdensityof, baum_welch, transition_matrix, initial_distribution, viterbi

# ╔═╡ 7a172b3e-3237-4986-a3a9-36351bbdc4eb
using Statistics: mean, std, cov, cor

# ╔═╡ fa2ce848-9051-4a8b-b0c3-f826f3fa8bba
using DataFrames: DataFrame

# ╔═╡ e6cc94f2-654d-49bb-b1d8-1503fdf3988f
using ZebrafishHMM2023: load_artr_wolf_2023, HMM_ARTR_Log, normalize_transition_matrix, ATol, viterbi_artr,
    artr_wolf_2023_temperatures, artr_wolf_2023_fishes, split_into_repeated_subsequences, load_hmm

# ╔═╡ 09548c3a-76b6-4292-9f6a-5fc6b683a46b
md"# Imports"

# ╔═╡ e3d756ca-ce8b-498c-84d7-7113f9d4bad0
import Makie

# ╔═╡ 4ace744f-4629-4982-a55e-be9d5cdae8a3
import CairoMakie

# ╔═╡ e256fea0-f477-48e6-8896-509a247c720b
import CSV

# ╔═╡ 748df523-4684-4dc9-a449-0c76c56983f2
df = let df = DataFrame()

	for temperature = artr_wolf_2023_temperatures(), fish = artr_wolf_2023_fishes(temperature)
	    data = load_artr_wolf_2023(; temperature, fish)
	    trajs = collect(eachcol(vcat(data.left, data.right)))

		empirical_cov = cov(vcat(data.left, data.right); dims=2)
	
	    time_unit = mean(diff(data.time)) # convert time bins to seconds
	
	    for nstates = 1:5
	        println("temperature=$temperature, fish=$fish, nstates=$nstates ...")
	        hmm = load_hmm(
				"/Users/jfdcd/projects/2023/Zebrafish_HMM/julia/Zebrafish_HMM_notebooks_2024/2024-02-20 ARTR/Saved HMMs/ARTR HMM $nstates states temperature=$temperature, fish=$fish.hd5",
				#"/data/cossio/projects/2023/zebrafish_hmm/Zebrafish_HMM/2024-02-20 ARTR/Saved HMMs/ARTR HMM $nstates states temperature=$(temperature), fish=$(fish).hd5",
				HMM_ARTR_Log
			)

			# log-likelihood
	        ll = logdensityof(hmm, trajs) / (data.time[end] - data.time[begin])

			# estimate covariances
	        generated_data = rand(hmm, 20000)
	        generated_cov = cov(stack(generated_data.obs_seq); dims=2)
	
	        empirical_cov_flat = [empirical_cov[i,j] for i = axes(empirical_cov, 1) for j = axes(empirical_cov, 2) if i ≤ j]
	        generated_cov_flat = [generated_cov[i,j] for i = axes(generated_cov, 1) for j = axes(generated_cov, 2) if i ≤ j]
	        ρ = cor(empirical_cov_flat, generated_cov_flat)

	        push!(df, (; temperature, fish, nstates, ll, ρ))
	    end
	end
	df
end

# ╔═╡ 51f78354-ad2c-46b2-80a5-b43a1a5ee8f8
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=100, height=100, xlabel="2 states", ylabel="3 states", title="Ll / sec", xticks=[-1000,0], yticks=[-1000,0])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 2), :].ll) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ll) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, -1500, 0)
	Makie.ylims!(ax, -1500, 0)
	Makie.Colorbar(fig[1,2], plt, label="temperature")
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 818cc7bb-6977-488e-9407-935a7b8bda85
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="3 states", ylabel="4 states", title="Ll / sec", xticks=[-1000,0], yticks=[-1000,0])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ll) 
		for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 4), :].ll) 
		for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, -1500, 0)
	Makie.ylims!(ax, -1500, 0)
	Makie.Colorbar(fig[1,2], plt, label="temperature")
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 38e7c0f8-a90c-497b-866d-66de0895e9b4
let fig = Makie.Figure()
	_sz = 100
	
	ax = Makie.Axis(fig[1,1], width=_sz, height=_sz, xlabel="Num. states", ylabel="avg. log-likelihood / sec", xticks=1:5, yticks=[-600, -500])
	plt = Makie.scatterlines!(ax, 1:5, [mean(df.ll[df.nstates .== nstates]) for nstates = 1:5])
	plt = Makie.scatterlines!(ax, [3], [mean(df.ll[df.nstates .== nstates]) for nstates = 3:3], color=:red, markersize=15)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ bafd3337-eaec-4c1e-b053-ceb9ed618f22
md"# Figure"

# ╔═╡ 64d72164-ab13-4bf6-9831-0240c420e5e7
_temp_fig_dir = mktempdir()

# ╔═╡ 24566b70-b026-47ef-b827-0a30d876f96f
let fig = Makie.Figure()
	_sz = 100

	ax = Makie.Axis(fig[1,1], width=_sz, height=_sz, xlabel="Num. states", ylabel="Two-point corr.", xticks=1:5)
	plt = Makie.scatterlines!(ax, 1:5, [mean(df.ρ[df.nstates .== nstates]) for nstates = 1:5])
	plt = Makie.scatterlines!(ax, [3], [mean(df.ρ[df.nstates .== nstates]) for nstates = 3:3], color=:red, markersize=10)
	plt = Makie.errorbars!(ax, 1:5, [mean(df.ρ[df.nstates .== nstates]) for nstates = 1:5], [std(df.ρ[df.nstates .== nstates]) for nstates = 1:5])
	plt = Makie.errorbars!(ax, [3], [mean(df.ρ[df.nstates .== nstates]) for nstates = 3:3], [std(df.ρ[df.nstates .== nstates]) for nstates = 3:3], color=:red)

	ax = Makie.Axis(fig[2,1], width=_sz, height=_sz, xlabel="Num. states", ylabel="avg. log-like. / sec", xticks=1:5, yticks=[-900, -700, -500, -300])
	plt = Makie.scatterlines!(ax, 1:5, [mean(df.ll[df.nstates .== nstates]) for nstates = 1:5])
	plt = Makie.scatterlines!(ax, [3], [mean(df.ll[df.nstates .== nstates]) for nstates = 3:3], color=:red, markersize=10)
	plt = Makie.errorbars!(ax, 1:5, [mean(df.ll[df.nstates .== nstates]) for nstates = 1:5], [std(df.ll[df.nstates .== nstates]) for nstates = 1:5])
	plt = Makie.errorbars!(ax, [3], [mean(df.ll[df.nstates .== nstates]) for nstates = 3:3], [std(df.ll[df.nstates .== nstates]) for nstates = 3:3], color=:red)
	
	ax = Makie.Axis(fig[1,2], width=_sz, height=_sz, xlabel="2 states", ylabel="3 states", title="2-point corr.", xticks=[0.5, 1], yticks=[0.5, 1])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 2), :].ρ) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ρ) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, 0.4, 1)
	Makie.ylims!(ax, 0.4, 1)
	Makie.Colorbar(fig[1,3], plt, label="temperature")

	ax = Makie.Axis(fig[1,4], width=_sz, height=_sz, xlabel="3 states", ylabel="4 states", title="2-point corr.", xticks=[0.5, 1], yticks=[0.5, 1])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ρ) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 4), :].ρ) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, 0.4, 1)
	Makie.ylims!(ax, 0.4, 1)
	Makie.Colorbar(fig[1,5], plt, label="temperature")

	ax = Makie.Axis(fig[2,2], width=_sz, height=_sz, xlabel="2 states", ylabel="3 states", title="Ll / sec", xticks=[-1000,0], yticks=[-1000,0])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 2), :].ll) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ll) for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, -1500, 0)
	Makie.ylims!(ax, -1500, 0)
	Makie.Colorbar(fig[2,3], plt, label="temperature")

	ax = Makie.Axis(fig[2,4], width=_sz, height=_sz, xlabel="3 states", ylabel="4 states", title="Ll / sec", xticks=[-1000,0], yticks=[-1000,0])
	plt = Makie.scatter!(ax, 
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 3), :].ll) 
		for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)],
	    [only(df[(df.temperature .== temperature) .& (df.fish .== fish) .& (df.nstates .== 4), :].ll) 
		for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)];
	    color = [temperature for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
	)
	Makie.ablines!(ax, 0, 1, color=:black, linestyle=:dash)
	Makie.xlims!(ax, -1500, 0)
	Makie.ylims!(ax, -1500, 0)
	Makie.Colorbar(fig[2,5], plt, label="temperature")

	Makie.Label(fig[1, 1, Makie.TopLeft()], "A)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)
	Makie.Label(fig[2, 1, Makie.TopLeft()], "B)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)
	Makie.Label(fig[1, 2, Makie.TopLeft()], "C)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)
	Makie.Label(fig[2, 2, Makie.TopLeft()], "D)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)
	Makie.Label(fig[1, 4, Makie.TopLeft()], "E)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)
	Makie.Label(fig[2, 4, Makie.TopLeft()], "F)", fontsize = 16, font = :bold, padding = (0, 5, 5, 0), halign = :right)

	Makie.resize_to_layout!(fig)

	fig_path = joinpath(_temp_fig_dir, "Sup5_new.pdf")
	Makie.save(fig_path, fig)
	@show fig_path

	fig
end

# ╔═╡ Cell order:
# ╠═09548c3a-76b6-4292-9f6a-5fc6b683a46b
# ╠═c61f5cd6-023f-11ef-343e-6b895c2b8f54
# ╠═e3d756ca-ce8b-498c-84d7-7113f9d4bad0
# ╠═4ace744f-4629-4982-a55e-be9d5cdae8a3
# ╠═e256fea0-f477-48e6-8896-509a247c720b
# ╠═4ec29bd5-35ca-4cf2-8305-148816cb2f5f
# ╠═7a172b3e-3237-4986-a3a9-36351bbdc4eb
# ╠═fa2ce848-9051-4a8b-b0c3-f826f3fa8bba
# ╠═e6cc94f2-654d-49bb-b1d8-1503fdf3988f
# ╠═748df523-4684-4dc9-a449-0c76c56983f2
# ╠═51f78354-ad2c-46b2-80a5-b43a1a5ee8f8
# ╠═818cc7bb-6977-488e-9407-935a7b8bda85
# ╠═38e7c0f8-a90c-497b-866d-66de0895e9b4
# ╠═bafd3337-eaec-4c1e-b053-ceb9ed618f22
# ╠═64d72164-ab13-4bf6-9831-0240c420e5e7
# ╠═24566b70-b026-47ef-b827-0a30d876f96f
