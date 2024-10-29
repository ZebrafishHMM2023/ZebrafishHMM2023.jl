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

# ╔═╡ 99879535-3957-45f6-a150-8baa82ce780f
md"# Label ARTR states in the data"

# ╔═╡ 85bee758-0273-4244-bc32-03ce527a267a
artr_data = Dict((; temperature, fish) =>
	ZebrafishHMM2023.load_artr_wolf_2023(; temperature, fish)
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 3186c6a6-4517-4f57-b801-6080c77fc877
artr_hmms = Dict((; temperature, fish) => 
	first(ZebrafishHMM2023.easy_train_artr_hmm(; temperature, fish, matteo_states_sort=true))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 825f3959-b20d-4885-a985-9000bb1a5305
artr_trajs = Dict((; temperature, fish) =>
	collect(eachcol(vcat(artr_data[(; temperature, fish)].left, artr_data[(; temperature, fish)].right)))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 1d4072ba-351b-4b3e-9b78-ec537d8e4257
artr_viterbi_states = Dict((; temperature, fish) =>
	HiddenMarkovModels.viterbi(artr_hmms[(; temperature, fish)], artr_trajs[(; temperature, fish)])
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ d6b7b2f2-d886-4776-a75b-2cdcd06a81c0
for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures(), fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
	@assert all(diff(artr_data[(; temperature, fish)].time) .≈ mean(diff(artr_data[(; temperature, fish)].time)))
end

# ╔═╡ 67a29f4b-da79-4961-9cc3-adc652f4972b
artr_time_unit = Dict(
	(; temperature, fish) => mean(diff(artr_data[(; temperature=26, fish=6)].time))
	for temperature = ZebrafishHMM2023.artr_wolf_2023_temperatures()
	for fish = ZebrafishHMM2023.artr_wolf_2023_fishes(temperature)
)

# ╔═╡ 315eb13a-ee2c-4973-8a31-46d12cd856d8
md"# HMMs trained on long trajectories of individual fish"

# ╔═╡ a6f7f26a-1b16-4960-9946-990dc1a93eb7
function my_train_full_behavior_hmm_for_long_trajs(full_fish_trajs)
	@info "Training bouts only HMM ..."
	
	trajs_bouts_only = [obs.θ for traj = full_fish_trajs for obs = traj]

	hmm_bouts = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0)
	hmm_bouts = ZebrafishHMM2023.normalize_all!(hmm_bouts)
	(hmm_bouts, lL) = HiddenMarkovModels.baum_welch(hmm_bouts, fish_trajs, length(fish_trajs); max_iterations=5000, check_loglikelihood_increasing=false, atol = ZebrafishHMM2023.ATol(1e-5))

	@info "Training full HMM ..."
	
	hmm = ZebrafishHMM2023.ZebrafishHMM_G3_Sym_Full_Exp(;
        pinit_turn=rand(), 
		transition_matrix=ZebrafishHMM2023.normalize_transition_matrix(rand(3,3)),
        σforw=0.1, turn=Gamma(1.5, 20.0),
        forward_displacement=Gamma(1.5, 1.2),
		turn_displacement=Gamma(1.5, 1.2),
        forward_interboutinterval=Exponential(1.8),
		turn_interboutinterval=Exponential(1.8),
        min_alpha = bouts_hmm.min_turn_alpha,
        only_train_spacetime=true
    )

    hmm.pinit_turn = bouts_hmm.pinit_turn
    hmm.transition_matrix .= bouts_hmm.transition_matrix
    hmm.σforw = bouts_hmm.σforw
    hmm.turn = bouts_hmm.turn

    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, full_fish_trajs, length(full_fish_trajs); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))

    return hmm
end

# ╔═╡ e525baca-de09-4fe9-9af1-42018cf08ab8
full_hmms = [my_train_full_behavior_hmm_for_long_trajs(fish_trajs) for fish_trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_full_obs()]

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
# ╠═0ae0436c-6db4-4013-8db0-2b4f90c568e4
# ╠═99879535-3957-45f6-a150-8baa82ce780f
# ╠═85bee758-0273-4244-bc32-03ce527a267a
# ╠═3186c6a6-4517-4f57-b801-6080c77fc877
# ╠═825f3959-b20d-4885-a985-9000bb1a5305
# ╠═1d4072ba-351b-4b3e-9b78-ec537d8e4257
# ╠═d6b7b2f2-d886-4776-a75b-2cdcd06a81c0
# ╠═67a29f4b-da79-4961-9cc3-adc652f4972b
# ╠═315eb13a-ee2c-4973-8a31-46d12cd856d8
# ╠═a6f7f26a-1b16-4960-9946-990dc1a93eb7
# ╠═e525baca-de09-4fe9-9af1-42018cf08ab8
