### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 1dc64219-66c6-44e8-8cba-bda286833f94
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 6ae0e2b3-16af-41c5-99d7-7bb4d5a5d7bb
using Statistics: mean, std, cov, cor

# ╔═╡ 5c8c9272-34dd-4aa0-a5e0-7df9eff2ba96
using DataFrames: DataFrame

# ╔═╡ d850c4ee-00c3-11ef-1a3f-63bb17e2bdab
md"# Imports"

# ╔═╡ 937668a0-5e1e-421b-8acc-cbd66549b1b6
import ZebrafishHMM2023

# ╔═╡ bcbc735a-0e24-44d6-89ed-4ddc4cd97e68
import CairoMakie

# ╔═╡ 0f287bd2-bd12-43bc-af95-deed437040f5
import Makie

# ╔═╡ 5c673973-3da8-4e43-b596-e900ad74e31d
import HiddenMarkovModels

# ╔═╡ 4bf4e443-19f5-4be8-84cb-2f06c6c7fdde
md"# Functions"

# ╔═╡ 95397cd2-cbd4-49d4-962d-523918d09cd4
md"# Analysis"

# ╔═╡ c651d7dd-40d8-43be-b470-7266fac5e8c7
raw_data = ZebrafishHMM2023.wolf_eyes_20240422_data()

# ╔═╡ 93272940-4259-44fa-9879-69a7e96db6ac
num_neurons_artr = size(raw_data.left, 1) + size(raw_data.right, 1)

# ╔═╡ 56901eb9-0383-4814-afb1-bd23528509f7
function train_hmm_with_num_states(num_states::Int)
	@info "Training with $num_states states"
	hmm_init = ZebrafishHMM2023.HMM_Eyes_ARTR_Only(
		ZebrafishHMM2023.normalize_transition_matrix(rand(num_states, num_states)),
		randn(num_neurons_artr, num_states), 1.0
	)
	hmm_trained, train_lls = HiddenMarkovModels.baum_welch(
		hmm_init, eachcol(vcat(raw_data.left, raw_data.right));
		max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7)
	)
	return (; hmm = hmm_trained, lls = train_lls, num_states)
end

# ╔═╡ 86b249b2-0cca-464b-ab7a-8533b5fd6266
empirical_cov = cov(vcat(raw_data.left, raw_data.right); dims=2)

# ╔═╡ 15492002-906f-4d87-9367-a21e1cec45a5
function estimate_covariance_correlations(hmm)
	generated_data = rand(hmm, 20000)
	generated_cov = cov(stack(generated_data.obs_seq); dims=2)

	empirical_cov_flat = [empirical_cov[i,j] for i = axes(empirical_cov, 1) for j = axes(empirical_cov, 2) if i ≤ j]
	generated_cov_flat = [generated_cov[i,j] for i = axes(generated_cov, 1) for j = axes(generated_cov, 2) if i ≤ j]

	return cor(empirical_cov_flat, generated_cov_flat)
end

# ╔═╡ ae9384a1-9788-4ce0-a622-30a18879a1dd
trained_hmms = [train_hmm_with_num_states(n) for n = 1:5]

# ╔═╡ b08f3f65-9b80-492f-8078-3b6403f1844a
cov_corrs = [estimate_covariance_correlations(m.hmm) for m = trained_hmms]

# ╔═╡ 651ef064-b2bc-4282-91b4-8521a22491d0
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=200, height=200, xlabel="Number of states", ylabel="log-likelihood")
	Makie.scatterlines!(ax, 1:5, [m.lls[end] for m = trained_hmms])
	Makie.scatterlines!(ax, [3], [m.lls[end] for m = [trained_hmms[3]]], color=:red, markersize=15)

	ax = Makie.Axis(fig[1,2], width=200, height=200, xlabel="Number of states", ylabel="Two-point correlation")
	Makie.scatterlines!(ax, 1:5, cov_corrs)
	Makie.scatterlines!(ax, [3], [cov_corrs[3]], color=:red, markersize=15)

    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ Cell order:
# ╠═d850c4ee-00c3-11ef-1a3f-63bb17e2bdab
# ╠═1dc64219-66c6-44e8-8cba-bda286833f94
# ╠═937668a0-5e1e-421b-8acc-cbd66549b1b6
# ╠═bcbc735a-0e24-44d6-89ed-4ddc4cd97e68
# ╠═0f287bd2-bd12-43bc-af95-deed437040f5
# ╠═5c673973-3da8-4e43-b596-e900ad74e31d
# ╠═6ae0e2b3-16af-41c5-99d7-7bb4d5a5d7bb
# ╠═5c8c9272-34dd-4aa0-a5e0-7df9eff2ba96
# ╠═4bf4e443-19f5-4be8-84cb-2f06c6c7fdde
# ╠═56901eb9-0383-4814-afb1-bd23528509f7
# ╠═15492002-906f-4d87-9367-a21e1cec45a5
# ╠═95397cd2-cbd4-49d4-962d-523918d09cd4
# ╠═c651d7dd-40d8-43be-b470-7266fac5e8c7
# ╠═93272940-4259-44fa-9879-69a7e96db6ac
# ╠═86b249b2-0cca-464b-ab7a-8533b5fd6266
# ╠═ae9384a1-9788-4ce0-a622-30a18879a1dd
# ╠═b08f3f65-9b80-492f-8078-3b6403f1844a
# ╠═651ef064-b2bc-4282-91b4-8521a22491d0
