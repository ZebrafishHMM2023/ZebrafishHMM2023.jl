### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 7f95a967-426a-465f-bad4-671372ea1092
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 7d355591-dddd-4463-97b1-d53e223bd6b3
using Statistics: mean

# ╔═╡ 8398c332-235b-4f6c-ab1e-8b5fd62d2d5a
using Distributions: Gamma

# ╔═╡ 0aae9cb6-85c1-4792-8c81-2d772b2f3b19
using Distributions: Exponential

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

# ╔═╡ 37ab290b-e7b6-42b2-bccf-7d73c1916557
md"# Train behavior models"

# ╔═╡ 88d1dfc7-e0c6-47c0-8170-787e290fc2c8
function train_bouts_hmm_sym(T::Int)
	@info "Training models for T = $T"

    trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)

    trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes
    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ a3add9b1-7a02-4231-981d-05646c73a8fd
function train_bouts_hmm_nosym(T::Int)
	@info "Training models for T = $T"

    trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)

	p_turn = rand()

    trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes
    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3([1 - p_turn, p_turn/2, p_turn/2], rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ 0601dac5-cef9-4b2b-83ce-d37064e4d9e1
bouts_hmms_sym = Dict(temperature => train_bouts_hmm_sym(temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ d1a0cda4-9970-47a8-aad8-03b8ef08409d
bouts_hmms_nosym = Dict(temperature => train_bouts_hmm_nosym(temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures())

# ╔═╡ e25e545e-d41f-4780-8dc4-4b2886b4d15c
md"# Check transition matrices"

# ╔═╡ f385e68d-cce4-40d2-85a4-0e470761d7ac
let fig = Makie.Figure()
	for (col, temperature) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		ax = Makie.Axis(fig[1,col]; width=120, height=120, xlabel="Symmetric", ylabel="Unconstrained", title="Temp. = $temperature")
		Makie.ablines!(ax, 0, 1, color=:black)
		Makie.scatter!(ax, vec(bouts_hmms_sym[temperature].transition_matrix), vec(bouts_hmms_nosym[temperature].transition_matrix))
		Makie.xlims!(ax, 0, 1)
		Makie.ylims!(ax, 0, 1)
	end
	
	Makie.resize_to_layout!(fig)

	fig_path = joinpath(mktempdir(), "Sym_vs_Asym.pdf")
	Makie.save(fig_path, fig)
	@show fig_path

	fig
end

# ╔═╡ 93ff04f0-efa2-4ea7-9a11-96315eb88c02
md"# Long trajectories"

# ╔═╡ ed9d3737-4558-4752-afd3-3c2329d7ed55
long_trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ e9dd05e4-1e36-4b5a-8f21-e75e8d55f666
begin
	hmms_individual_sym = ZebrafishHMM2023.ZebrafishHMM_G3_Sym[]
	hmms_individual_asym = ZebrafishHMM2023.ZebrafishHMM_G3[]
	for (n, fish) = enumerate(long_trajs)
	    println("Fitting fish $n of 18.")

		# fit symmetric
	    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
	    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish, length(fish); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	    push!(hmms_individual_sym, hmm)

		# fit asymmetric
		p_turn = rand()
		hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3([1 - p_turn, p_turn/2, p_turn/2], rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
		(hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish, length(fish); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))
		push!(hmms_individual_asym, hmm)
	end
end

# ╔═╡ b1d2fec3-384a-4b49-9bf4-12b37623b442
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1]; width=400, height=400, xlabel="Symmetric", ylabel="Unconstrained", title="Individual fish trajectories")
	Makie.ablines!(ax, 0, 1, color=:black)
	for (hmm_sym, hmm_asym) = zip(hmms_individual_sym, hmms_individual_asym)
		Makie.scatter!(ax, vec(hmm_sym.transition_matrix), vec(hmm_asym.transition_matrix))
	end
	Makie.xlims!(ax, 0, 1)
	Makie.ylims!(ax, 0, 1)
	Makie.resize_to_layout!(fig)
	fig_path = joinpath(mktempdir(), "Sym_vs_Asym_long.pdf")
	Makie.save(fig_path, fig)
	@show fig_path
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
# ╠═7d355591-dddd-4463-97b1-d53e223bd6b3
# ╠═8398c332-235b-4f6c-ab1e-8b5fd62d2d5a
# ╠═0aae9cb6-85c1-4792-8c81-2d772b2f3b19
# ╠═4f8b021f-7917-4689-a95e-02e20a766b3a
# ╠═37ab290b-e7b6-42b2-bccf-7d73c1916557
# ╠═88d1dfc7-e0c6-47c0-8170-787e290fc2c8
# ╠═a3add9b1-7a02-4231-981d-05646c73a8fd
# ╠═0601dac5-cef9-4b2b-83ce-d37064e4d9e1
# ╠═d1a0cda4-9970-47a8-aad8-03b8ef08409d
# ╠═e25e545e-d41f-4780-8dc4-4b2886b4d15c
# ╠═f385e68d-cce4-40d2-85a4-0e470761d7ac
# ╠═93ff04f0-efa2-4ea7-9a11-96315eb88c02
# ╠═ed9d3737-4558-4752-afd3-3c2329d7ed55
# ╠═e9dd05e4-1e36-4b5a-8f21-e75e8d55f666
# ╠═b1d2fec3-384a-4b49-9bf4-12b37623b442
