### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 71410271-e1ce-4cb9-ae84-1649120f2a3d
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 825689ff-4b29-4343-a9a9-13dc5b226384
using Statistics: mean

# ╔═╡ 1bcbcf6f-533b-4b26-b349-f30e2d437527
using Distributions: Gamma

# ╔═╡ 1c552b40-8a75-4255-aeea-6968f4142405
using Distributions: Exponential

# ╔═╡ e0a2bd7b-af61-404a-b384-67e25bb2fa5e
md"# Imports"

# ╔═╡ d6b77835-0736-435a-9534-962e5e6cc635
import ZebrafishHMM2023

# ╔═╡ 340e17bd-1f99-4b95-bbe3-157f21f7818d
import HiddenMarkovModels

# ╔═╡ acb1e02a-43d0-491a-8dde-de143b6a53cc
import Makie

# ╔═╡ ddff579c-e46b-43aa-a5ad-b3c514f5214b
import CairoMakie

# ╔═╡ 6e21301c-383d-46ed-9192-706f7b930837
import PlutoUI

# ╔═╡ 1e03bdac-a251-4e2c-8ce0-4bea146c4a96
import CSV

# ╔═╡ 9c809e20-800d-4697-9f4f-b55a68c6a403
import HDF5

# ╔═╡ bec9b6ac-1efc-4241-ab99-77f727132a61
import Distributions

# ╔═╡ 65542451-1f57-4fd7-8c1d-01c37f6616b0
PlutoUI.TableOfContents()

# ╔═╡ f38e6b71-7fdf-4c5d-b9db-dc46c7a03495
md"# Train HMM models for each temperature"

# ╔═╡ bfb29b9d-813b-4d80-8000-693eeb2025a0
function train_hmm_at_temperature(T)
	trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)
    trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes

	println("Training HMM at temperature $T ...")
	
    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Distributions.Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

	return hmm, lL
end

# ╔═╡ 1b3e46b3-ffb2-4ddd-af93-2d4b0d56763b
trained_hmms = Dict(T => train_hmm_at_temperature(T) for T = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ b23a2be9-1542-45a5-8d77-9e565851758c


# ╔═╡ Cell order:
# ╠═e0a2bd7b-af61-404a-b384-67e25bb2fa5e
# ╠═71410271-e1ce-4cb9-ae84-1649120f2a3d
# ╠═d6b77835-0736-435a-9534-962e5e6cc635
# ╠═340e17bd-1f99-4b95-bbe3-157f21f7818d
# ╠═acb1e02a-43d0-491a-8dde-de143b6a53cc
# ╠═ddff579c-e46b-43aa-a5ad-b3c514f5214b
# ╠═6e21301c-383d-46ed-9192-706f7b930837
# ╠═1e03bdac-a251-4e2c-8ce0-4bea146c4a96
# ╠═9c809e20-800d-4697-9f4f-b55a68c6a403
# ╠═bec9b6ac-1efc-4241-ab99-77f727132a61
# ╠═825689ff-4b29-4343-a9a9-13dc5b226384
# ╠═1bcbcf6f-533b-4b26-b349-f30e2d437527
# ╠═1c552b40-8a75-4255-aeea-6968f4142405
# ╠═65542451-1f57-4fd7-8c1d-01c37f6616b0
# ╠═f38e6b71-7fdf-4c5d-b9db-dc46c7a03495
# ╠═bfb29b9d-813b-4d80-8000-693eeb2025a0
# ╠═1b3e46b3-ffb2-4ddd-af93-2d4b0d56763b
# ╠═b23a2be9-1542-45a5-8d77-9e565851758c
