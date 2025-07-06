### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 34bee668-5efa-472b-9093-e082daf4a796
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 353a003c-acc2-4632-b8c5-3765ace40490
using Statistics: mean

# ╔═╡ 6d701605-96b1-4dfe-a806-4841fd1c6b48
using Distributions: Gamma, Exponential, Multinomial

# ╔═╡ 3eaf5a73-3865-4cfb-b72f-b5829ae5c1e5
using StatsBase: countmap, proportionmap, countmap

# ╔═╡ 7e97f5ea-6a7f-4b35-92c5-8896205d94cb
using Makie: @L_str

# ╔═╡ 8c4edcda-6200-4982-9286-e51b13fe0ba0
md"# Imports"

# ╔═╡ 77baf223-b717-4d30-a6b6-c9fed06e6cd4
import ZebrafishHMM2023, HiddenMarkovModels, Makie, CairoMakie, PlutoUI, CSV, HDF5

# ╔═╡ f1541577-4727-4293-a9a7-4a3c6839596c
PlutoUI.TableOfContents()

# ╔═╡ 18e2db6e-fa7c-4d7b-a6f6-689e6a013969
md"# Functions"

# ╔═╡ 034cfb05-8f49-455e-9ea9-7fb7cbda8bcd
threshold_angle = 10

# ╔═╡ 4a219437-ccb9-4b2b-957b-07d9b52ef0d0
function threshold_classify_angle(θ::Real)
	if -threshold_angle ≤ θ ≤ threshold_angle
		return 0
	elseif θ > threshold_angle
		return 1
	elseif θ < threshold_angle
		return -1
	end
end

# ╔═╡ 27c404e5-f134-42b9-aeed-279f11a532a0
ZebrafishHMM2023.behaviour_free_swimming_temperatures()

# ╔═╡ 508767cd-69b2-4c83-b294-1bf106aa54a9
function load_thresholded_mc_states(T::Real)
	trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)
	return [map(threshold_classify_angle, seq) for seq = trajs]
end

# ╔═╡ c447cc13-94ad-4611-afab-0b0e14c38635
function word_counts(trajs, n::Int)
	full_counts = Dict{NTuple{n,Int},Int}()
	for seq = trajs
		words = [Tuple(p) for p = Base.Iterators.partition(seq, n) if length(p) == n]
		dict = countmap(words)
		for (k,v) = dict
			if haskey(full_counts, k)
				full_counts[k] += v
			else
				full_counts[k] = v
			end
		end
	end
	return full_counts
end

# ╔═╡ f9b074f0-75e9-47f1-b1ef-fa4d6173eb6a
function word_frequencies(trajs, n::Int)
	full_counts = word_counts(trajs, n)
	Z = sum(values(full_counts))
	return Dict(Tuple(k) => full_counts[k]/Z for (k,v) = full_counts)
end

# ╔═╡ 6dc99bf3-ed4b-443f-b19d-c4f4c3d71c8f
function simulate_markovian_triples(doubles, triples)
	sim_triples = Dict{NTuple{3,Int}, Int}()
	for (k,v) = triples
		p_unorm = [doubles[(x3,k[2])] for x3=-1:1]
		p = p_unorm / sum(p_unorm)
		simulated_counts = rand(Multinomial(v, p))
		for (i,x3) = enumerate(-1:1)
			if haskey(sim_triples, (k[1],k[2],x3))
				sim_triples[(k[1],k[2],x3)] += simulated_counts[i]
			else
				sim_triples[(k[1],k[2],x3)] = simulated_counts[i]
			end
		end
	end
	return sim_triples
end

# ╔═╡ f26d1864-cf3e-4a96-97e7-2abe383395ca
md"# Markovianity"

# ╔═╡ 82c1aa23-6ba5-4e87-849f-a2d7279ab25d
let fig = Makie.Figure()
	for (i,T) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		triples = word_frequencies(load_thresholded_mc_states(T), 3)
		doubles = word_frequencies(load_thresholded_mc_states(T), 2)
		singles = word_frequencies(load_thresholded_mc_states(T), 1)
		
		ax = Makie.Axis(fig[1,i]; width=150, height=150, title="T=$T", xlabel=L"P(x_{t}|x_{t-1},x_{t-2})", ylabel=L"P(x_{t}|x_{t-1})")
		Makie.ablines!(ax, 0, 1; color=:black)
		Makie.scatter!(ax, [triples[k] / doubles[(k[1],k[2])] for k = keys(triples)], [doubles[(k[2],k[3])] / singles[(k[2],)] for k = keys(triples)])
		Makie.xlims!(ax, 0, 1)
		Makie.ylims!(ax, 0, 1)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ e59cf68f-0009-41b9-b54a-bfb6ceda2a86
let fig = Makie.Figure()
	for (i,T) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		triples = word_frequencies(load_thresholded_mc_states(T), 3)
		doubles = word_frequencies(load_thresholded_mc_states(T), 2)
		singles = word_frequencies(load_thresholded_mc_states(T), 1)

		ax = Makie.Axis(fig[1,i]; width=150, height=150, title="T=$T", xlabel=L"P(x_{t}|x_{t-1},x_{t-2}) - P(x_{t}|x_{t-1})", ylabel="Frequency")
		Makie.hist!(ax, [triples[(x1,x2,x3)] / doubles[(x1,x2)] - doubles[(x2,x3)] / singles[(x2,)] for x1 = -1:1 for x2=-1:1 for x3=-1:1]; normalization=:pdf, bins=-0.08:0.02:0.08)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ b23e920b-6cfb-49a8-bef3-af9dfcd378d8
let fig = Makie.Figure()
	for (i,T) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		triples = word_counts(load_thresholded_mc_states(T), 3)
		doubles = word_counts(load_thresholded_mc_states(T), 2)
		singles = word_counts(load_thresholded_mc_states(T), 1)

		simulated_triples = simulate_markovian_triples(doubles, triples)

		triples_normalized = Dict(k => v / sum(values(triples)) for (k,v) = triples)
		doubles_normalized = Dict(k => v / sum(values(doubles)) for (k,v) = doubles)
		singles_normalized = Dict(k => v / sum(values(singles)) for (k,v) = singles)

		simulated_triples_normalized = Dict(k => v / sum(values(simulated_triples)) for (k,v) = simulated_triples)

		ax = Makie.Axis(fig[1,i]; width=250, height=250, title="T=$T", xlabel=L"\log_{10} P(x_{t}|x_{t-1},x_{t-2}) / P(x_{t}|x_{t-1})", ylabel="Frequency")
		Makie.hist!(ax, log10.([triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)] / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) for x1 = -1:1 for x2=-1:1 for x3=-1:1]); 
					normalization=:pdf, bins=-0.2:0.02:0.2, label="data")
		Makie.hist!(ax, log10.([simulated_triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)] / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) for x1 = -1:1 for x2=-1:1 for x3=-1:1]); 
					normalization=:pdf, bins=-0.2:0.02:0.2, label="null model")

		Makie.axislegend(ax, position=:rt)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 5dd84cdf-410a-4ff2-b01e-5eb2be4bc97c
let fig = Makie.Figure()
	for (i,T) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		triples = word_counts(load_thresholded_mc_states(T), 3)
		doubles = word_counts(load_thresholded_mc_states(T), 2)
		singles = word_counts(load_thresholded_mc_states(T), 1)

		simulated_triples = simulate_markovian_triples(doubles, triples)

		triples_normalized = Dict(k => v / sum(values(triples)) for (k,v) = triples)
		doubles_normalized = Dict(k => v / sum(values(doubles)) for (k,v) = doubles)
		singles_normalized = Dict(k => v / sum(values(singles)) for (k,v) = singles)

		simulated_triples_normalized = Dict(k => v / sum(values(simulated_triples)) for (k,v) = simulated_triples)

		ax = Makie.Axis(fig[1,i]; width=250, height=250, title="T=$T", xlabel=L"(P(x_{t}|x_{t-1},x_{t-2}) - P(x_{t}|x_{t-1})) / P(x_{t}|x_{t-1})", ylabel="Frequency")
		Makie.hist!(ax, 
					[(triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)]) / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) - 1 for x1 = -1:1 for x2=-1:1 for x3=-1:1]; 
					normalization=:pdf, bins=-0.5:0.05:0.5, label="data"
					)
		Makie.hist!(ax,
					[(simulated_triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)]) / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) - 1 for x1 = -1:1 for x2=-1:1 for x3=-1:1]; 
					normalization=:pdf, bins=-0.5:0.05:0.5, label="null model")

		Makie.axislegend(ax, position=:rt)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ cb92016e-c378-4038-976a-ec50243af26d
md"# Markovianity with HMM labeled states (Viterbi)"

# ╔═╡ 8ea43eca-c3bf-4d54-a68a-3c6aff386ff7
function train_bouts_hmm_sym(T::Int)
	@info "Training models for T = $T"

    trajs = ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T)

    trajs = filter(traj -> all(!iszero, traj), trajs) # zeros give trouble sometimes
    hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Gamma(1.5, 20.0), 1.0))
    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, trajs, length(trajs); max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-6))

    return hmm
end

# ╔═╡ b3edcf47-8dd6-47f7-b54c-ac002bf913b7
trajs_free = Dict(T => ZebrafishHMM2023.load_behaviour_free_swimming_trajs(T) for T = ZebrafishHMM2023.behaviour_free_swimming_temperatures());

# ╔═╡ 543f98c0-778f-4625-80b9-c69451c886cc
bouts_hmms = Dict(temperature => train_bouts_hmm_sym(temperature) for temperature = ZebrafishHMM2023.behaviour_free_swimming_temperatures());

# ╔═╡ 3c9f00be-5c24-4908-90d3-c174334b4258
viterbi_labeled_trajs = Dict(T => HiddenMarkovModels.viterbi(bouts_hmms[T], trajs_free[T], length(trajs_free[T])) for T = ZebrafishHMM2023.behaviour_free_swimming_temperatures());

# ╔═╡ 31aae8bd-a16b-4ad6-a612-3f79eca56dd3
let fig = Makie.Figure()
	for (i, T) = enumerate(ZebrafishHMM2023.behaviour_free_swimming_temperatures())
		labeled_seqs = [t .- 2 for t = viterbi_labeled_trajs[T]]
		
		triples = word_counts(labeled_seqs, 3)
		doubles = word_counts(labeled_seqs, 2)
		singles = word_counts(labeled_seqs, 1)

		simulated_triples = simulate_markovian_triples(doubles, triples)

		triples_normalized = Dict(k => v / sum(values(triples)) for (k,v) = triples)
		doubles_normalized = Dict(k => v / sum(values(doubles)) for (k,v) = doubles)
		singles_normalized = Dict(k => v / sum(values(singles)) for (k,v) = singles)

		simulated_triples_normalized = Dict(k => v / sum(values(simulated_triples)) for (k,v) = simulated_triples)

		ax = Makie.Axis(fig[1,i]; width=250, height=250, title="T=$T", xlabel=L"(P(x_{t}|x_{t-1},x_{t-2}) - P(x_{t}|x_{t-1})) / P(x_{t}|x_{t-1})", ylabel="Frequency")
		Makie.hist!(ax, 
			[(triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)]) / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) - 1 for x1 = -1:1 for x2=-1:1 for x3=-1:1]; 
			normalization=:pdf, bins=-0.5:0.05:0.5, label="HMM labels"
		)
		Makie.hist!(ax,
			[(simulated_triples_normalized[(x1,x2,x3)] / doubles_normalized[(x1,x2)]) / (doubles_normalized[(x2,x3)] / singles_normalized[(x2,)]) - 1 for x1 = -1:1 for x2=-1:1 for x3=-1:1]; 
			normalization=:pdf, bins=-0.5:0.05:0.5, label="null model"
		)

		Makie.axislegend(ax, position=:rt)
	end
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ c2f11ebb-5aee-4341-84f9-75eb838d4620
viterbi(hmm, trajs, length(trajs))

# ╔═╡ Cell order:
# ╠═8c4edcda-6200-4982-9286-e51b13fe0ba0
# ╠═34bee668-5efa-472b-9093-e082daf4a796
# ╠═77baf223-b717-4d30-a6b6-c9fed06e6cd4
# ╠═353a003c-acc2-4632-b8c5-3765ace40490
# ╠═6d701605-96b1-4dfe-a806-4841fd1c6b48
# ╠═3eaf5a73-3865-4cfb-b72f-b5829ae5c1e5
# ╠═7e97f5ea-6a7f-4b35-92c5-8896205d94cb
# ╠═f1541577-4727-4293-a9a7-4a3c6839596c
# ╠═18e2db6e-fa7c-4d7b-a6f6-689e6a013969
# ╠═034cfb05-8f49-455e-9ea9-7fb7cbda8bcd
# ╠═4a219437-ccb9-4b2b-957b-07d9b52ef0d0
# ╠═27c404e5-f134-42b9-aeed-279f11a532a0
# ╠═508767cd-69b2-4c83-b294-1bf106aa54a9
# ╠═c447cc13-94ad-4611-afab-0b0e14c38635
# ╠═f9b074f0-75e9-47f1-b1ef-fa4d6173eb6a
# ╠═6dc99bf3-ed4b-443f-b19d-c4f4c3d71c8f
# ╠═f26d1864-cf3e-4a96-97e7-2abe383395ca
# ╠═82c1aa23-6ba5-4e87-849f-a2d7279ab25d
# ╠═e59cf68f-0009-41b9-b54a-bfb6ceda2a86
# ╠═b23e920b-6cfb-49a8-bef3-af9dfcd378d8
# ╠═5dd84cdf-410a-4ff2-b01e-5eb2be4bc97c
# ╠═cb92016e-c378-4038-976a-ec50243af26d
# ╠═8ea43eca-c3bf-4d54-a68a-3c6aff386ff7
# ╠═b3edcf47-8dd6-47f7-b54c-ac002bf913b7
# ╠═543f98c0-778f-4625-80b9-c69451c886cc
# ╠═3c9f00be-5c24-4908-90d3-c174334b4258
# ╠═31aae8bd-a16b-4ad6-a612-3f79eca56dd3
# ╠═c2f11ebb-5aee-4341-84f9-75eb838d4620
