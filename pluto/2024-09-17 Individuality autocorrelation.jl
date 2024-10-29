### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 9611b800-e29b-4126-8857-03501df2de90
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ cf3f8b50-f15c-4085-bb6b-0fece67756cb
using Statistics: mean, std, cor

# ╔═╡ fae587e7-18fb-4e22-8079-5fec057e3465
md"# Imports"

# ╔═╡ 1ec35e4d-4907-4e5d-8ae0-f368457dea09
import ZebrafishHMM2023

# ╔═╡ 8feec761-fd4b-4e2e-8914-709402eed102
import HiddenMarkovModels

# ╔═╡ 69faad55-c375-41bd-bc2f-c03683d013d4
import Makie

# ╔═╡ 2cfa86f1-2f8c-4ca4-a5a3-c7930a4aaf40
import CairoMakie

# ╔═╡ 40786042-b8f0-4bcd-89a3-e7c64e68b49c
import PlutoUI

# ╔═╡ f1aa18ef-a6b3-4e1e-8cf1-d3db5106ef0d
import CSV

# ╔═╡ 58a34db7-fff2-46f0-a55d-f4c044b0adf6
import HDF5

# ╔═╡ 1d1838d9-125c-4c30-b0c5-cd41cc48c517
import Distributions

# ╔═╡ 2fe9c14e-b5f2-4070-9bd7-09767956a806
PlutoUI.TableOfContents()

# ╔═╡ 47abc679-bff0-468b-bdf0-1277ae709fe5
md"# Data"

# ╔═╡ 0cf2d69b-3bd0-4c19-840d-1d32e3cdb457
trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ 257d159c-b82f-432f-88e2-0ab5478f4525
md"# Autocorrelation"

# ╔═╡ 30d915a5-62ad-4daf-99cb-3149c5358fe9
all_autocors = [
	[cor([seq[t] for t = 1:length(seq) - Δ + 1], [seq[t] for t = Δ:length(seq)]) for Δ = 1:floor(Int, length(seq)) - 2]
	for (fish, fish_trajs) = enumerate(trajs) for (n, seq) = enumerate(fish_trajs)
]

# ╔═╡ c1892559-b5aa-4b0a-83a4-8c5f15994944
avg_autocors = [mean(a[Δ] for a = all_autocors if length(a) ≥ Δ) for Δ = 1:maximum(map(length, all_autocors))]

# ╔═╡ 7e0b20fc-474e-4628-9db8-ef2e0dde3daf
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=400, height=400)
	Makie.scatterlines!(avg_autocors)
	Makie.xlims!(0, 100)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ Cell order:
# ╠═fae587e7-18fb-4e22-8079-5fec057e3465
# ╠═9611b800-e29b-4126-8857-03501df2de90
# ╠═1ec35e4d-4907-4e5d-8ae0-f368457dea09
# ╠═8feec761-fd4b-4e2e-8914-709402eed102
# ╠═69faad55-c375-41bd-bc2f-c03683d013d4
# ╠═2cfa86f1-2f8c-4ca4-a5a3-c7930a4aaf40
# ╠═40786042-b8f0-4bcd-89a3-e7c64e68b49c
# ╠═f1aa18ef-a6b3-4e1e-8cf1-d3db5106ef0d
# ╠═58a34db7-fff2-46f0-a55d-f4c044b0adf6
# ╠═1d1838d9-125c-4c30-b0c5-cd41cc48c517
# ╠═cf3f8b50-f15c-4085-bb6b-0fece67756cb
# ╠═2fe9c14e-b5f2-4070-9bd7-09767956a806
# ╠═47abc679-bff0-468b-bdf0-1277ae709fe5
# ╠═0cf2d69b-3bd0-4c19-840d-1d32e3cdb457
# ╠═257d159c-b82f-432f-88e2-0ab5478f4525
# ╠═30d915a5-62ad-4daf-99cb-3149c5358fe9
# ╠═c1892559-b5aa-4b0a-83a4-8c5f15994944
# ╠═7e0b20fc-474e-4628-9db8-ef2e0dde3daf
