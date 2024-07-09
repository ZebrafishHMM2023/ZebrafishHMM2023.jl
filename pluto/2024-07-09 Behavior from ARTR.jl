### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 7f95a967-426a-465f-bad4-671372ea1092
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 5b3604a6-3dcc-11ef-1ff1-a71b51c7d8e9
md"# Imports"

# ╔═╡ f9ed9420-cf6a-4c65-9e46-6e3e8fc54476
import ZebrafishHMM2023

# ╔═╡ 7571c197-a647-4c2e-8510-a05b7cbc447f
import HiddenMarkovModels

# ╔═╡ 44b6d472-e1e2-4e6b-970a-182bbe07f72f
import PlutoUI

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

# ╔═╡ fd8ef759-e4e2-4a8b-8fa7-bafad992ef7a
artr_viterbi_states[(; temperature=26, fish=6)]

# ╔═╡ 369d6c55-4239-4eeb-b23b-4c3910d791fd
md"# Load behavior data"

# ╔═╡ 60937be5-8526-4e3f-a17d-d4bb2f6e06a7
ZebrafishHMM2023.load_full_obs(26)[1][1]

# ╔═╡ Cell order:
# ╠═5b3604a6-3dcc-11ef-1ff1-a71b51c7d8e9
# ╠═7f95a967-426a-465f-bad4-671372ea1092
# ╠═f9ed9420-cf6a-4c65-9e46-6e3e8fc54476
# ╠═7571c197-a647-4c2e-8510-a05b7cbc447f
# ╠═44b6d472-e1e2-4e6b-970a-182bbe07f72f
# ╠═4f8b021f-7917-4689-a95e-02e20a766b3a
# ╠═a67c5bfa-02b5-41af-8bbe-5ca350bfb58f
# ╠═ee9ae8b1-e589-4c71-8450-0b9ea915df21
# ╠═c1fa7427-1167-446b-a8e7-50f0b321ae2d
# ╠═a6c708f3-7f96-416e-8aca-1a7c02a21e98
# ╠═ac410781-0712-45dd-b3eb-713b5629c329
# ╠═fd8ef759-e4e2-4a8b-8fa7-bafad992ef7a
# ╠═369d6c55-4239-4eeb-b23b-4c3910d791fd
# ╠═60937be5-8526-4e3f-a17d-d4bb2f6e06a7
