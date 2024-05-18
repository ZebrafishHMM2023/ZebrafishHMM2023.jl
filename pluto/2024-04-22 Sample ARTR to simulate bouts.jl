### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ c31f2000-f2a7-43e0-9cc3-28cc64ab92cb
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ f6fcf6cc-83b0-4356-b670-743d32c4e95c
using Statistics: mean

# ╔═╡ 059837ce-df90-4e34-93cf-b5caa8896c26
using Distributions: Normal

# ╔═╡ 65a81eb6-1ec2-4f83-b251-42169d3b1122
md"# Imports"

# ╔═╡ 45fcf939-01c6-4101-bf13-e0df8bf5e28f
import CairoMakie

# ╔═╡ 11d3b9a4-8c4a-47fc-a853-679895119d72
import Makie

# ╔═╡ 3de8f8b6-8c11-44ef-8a13-28c290e22ee2
import ZebrafishHMM2023

# ╔═╡ 6c948504-6166-483b-8089-37c83c6546a2
import HiddenMarkovModels

# ╔═╡ 89cd4686-60bb-4fe2-98b0-faa369321029
md"# Functions"

# ╔═╡ a6d037a2-3b50-45dc-a4fa-ec748fa1d58c
md"# Analysis"

# ╔═╡ f27e0027-9500-456d-b689-844dd5815d94
num_hmm_states = 3

# ╔═╡ 02d1ca2e-1e9e-445e-8679-8688f5cdc393
# Train HMMs
for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures(), fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	@info "Training HMM for temperature = $temperature, fish = $fish ..."

	# load data
    data = ZebrafishHMM2023.load_artr_wolf_2023(; temperature, fish)
    trajs = collect(eachcol(vcat(data.left, data.right)))
    time_unit = mean(diff(data.time)) # convert time bins to seconds

    Nneurons = size(data.left, 1) + size(data.right, 1)

	# train HMM
	hmm = ZebrafishHMM2023.HMM_ARTR_Log(ZebrafishHMM2023.normalize_transition_matrix(rand(num_hmm_states, num_hmm_states)), randn(Nneurons, num_hmm_states), 5.0)
	(hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs; max_iterations = 200, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7))
end

# ╔═╡ dcd930a3-b799-497b-9935-daf7fb60d5b0


# ╔═╡ Cell order:
# ╠═65a81eb6-1ec2-4f83-b251-42169d3b1122
# ╠═c31f2000-f2a7-43e0-9cc3-28cc64ab92cb
# ╠═45fcf939-01c6-4101-bf13-e0df8bf5e28f
# ╠═11d3b9a4-8c4a-47fc-a853-679895119d72
# ╠═3de8f8b6-8c11-44ef-8a13-28c290e22ee2
# ╠═6c948504-6166-483b-8089-37c83c6546a2
# ╠═f6fcf6cc-83b0-4356-b670-743d32c4e95c
# ╠═059837ce-df90-4e34-93cf-b5caa8896c26
# ╠═89cd4686-60bb-4fe2-98b0-faa369321029
# ╠═a6d037a2-3b50-45dc-a4fa-ec748fa1d58c
# ╠═f27e0027-9500-456d-b689-844dd5815d94
# ╠═02d1ca2e-1e9e-445e-8679-8688f5cdc393
# ╠═dcd930a3-b799-497b-9935-daf7fb60d5b0
