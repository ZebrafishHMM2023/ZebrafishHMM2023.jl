### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ b5731209-833e-400a-bd91-0ca87807f3df
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 5c003b8f-2e2b-498f-af41-033c6b7c691c
using Statistics: mean

# ╔═╡ cf43f299-64fe-4aeb-9a3c-4f963fa72525
using Distributions: Normal

# ╔═╡ 5b7dce2e-1f72-4885-92eb-4c2cbc6d60b9
md"# Imports"

# ╔═╡ b634b8dd-491f-4ad6-a177-7ef89744a41c
import CairoMakie

# ╔═╡ 3c9b3f3f-d145-4a52-85de-9052b5803de2
import Makie

# ╔═╡ c4a03051-84af-48e7-a3eb-e815517fff77
import ZebrafishHMM2023

# ╔═╡ e2fde7f0-fc5e-4ad7-b35c-1a6c91242a53
import HiddenMarkovModels

# ╔═╡ 3bc16dc1-a7ec-4e01-9bb2-df7027a51ebb
md"# Functions"

# ╔═╡ e8bb2a96-4542-4253-8a5b-4e3c40498086
function equal_partition(n::Int64, parts::Int64)
    if n < parts
        return [ x:x for x in 1:n ]
    end
    starts = push!(Int64.(round.(1:n/parts:n)), n+1)
    return [ starts[i]:starts[i+1]-1 for i in 1:length(starts)-1 ]
end

# ╔═╡ 7a8f34d7-a193-4c61-b3e6-8d33c0e29d10
function equal_partition(V::AbstractVector, parts::Int64)
    ranges = equal_partition(length(V), parts)
    return [ view(V,range) for range in ranges ]
end

# ╔═╡ 405cd60c-5d49-4981-9254-210eb472fdb0
md"# Analysis"

# ╔═╡ dc200e3a-35f1-4078-b926-dca55611e6f3
raw_data = ZebrafishHMM2023.wolf_eyes_20240501_run_data(3)

# ╔═╡ f2921685-b0c0-4915-90cd-9df3676ce0f0
gaze_data_subsampled = map(mean, equal_partition(raw_data.gaze, size(raw_data.left, 2)))

# ╔═╡ e0ec0817-e823-47d2-a6d3-c9b6852dd438
aggregated_data = collect(zip(gaze_data_subsampled, eachcol(vcat(raw_data.left, raw_data.right))))

# ╔═╡ fcd00cde-3ab4-4004-bb02-9484d707a5dc
num_neurons_artr = size(raw_data.left, 1) + size(raw_data.right, 1)

# ╔═╡ cf06a342-e143-4947-91f6-e140616a3282
hmm_num_states = 3

# ╔═╡ ce35a0bc-a905-447b-ba0a-ace0e1607b5d
hmm_init = ZebrafishHMM2023.HMM_Gaze_ARTR(
	ZebrafishHMM2023.normalize_transition_matrix(rand(hmm_num_states, hmm_num_states)),
	randn(hmm_num_states), ones(hmm_num_states), randn(num_neurons_artr, hmm_num_states), 1.0
)

# ╔═╡ b24bcd54-bf05-4e33-946e-f5e18313d5b8
hmm_trained, train_lls = HiddenMarkovModels.baum_welch(
	hmm_init, aggregated_data; max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7)
)

# ╔═╡ 3f0f82be-19d3-473e-b1ee-2a492f9f3044
viterbi_states = HiddenMarkovModels.viterbi(hmm_trained, aggregated_data)

# ╔═╡ a80514e2-7ac3-46fd-855b-b10366b1759f
raw_data.left[:, viterbi_states .== 1]

# ╔═╡ a12bfd34-695c-453f-981d-8b3ac87c42c2
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :blue]
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="mL", ylabel="mR")
    for s = reverse(1:length(hmm_trained))
        Makie.scatter!(ax,
            vec(mean(raw_data.left[:, viterbi_states .== s]; dims=1)),
            vec(mean(raw_data.right[:, viterbi_states .== s]; dims=1));
            markersize=5, color=(_colors[s], 0.75), label="$s"
        )
    end
    Makie.xlims!(ax, -0.01, 0.5)
    Makie.ylims!(ax, -0.01, 0.5)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 33352bd9-ed35-454b-87da-a4e88cdc72e4
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :blue]
    ax = Makie.Axis(fig[1,1], width=1500, height=300, xlabel="time", ylabel="gaze")
	Makie.lines!(ax, gaze_data_subsampled, color=:gray)
    for s = 1:length(hmm_trained)
		Makie.scatter!(ax, findall(viterbi_states .== s), gaze_data_subsampled[viterbi_states .== s], markersize=7, color=_colors[s], label="$s")
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ c0c6176f-fe4e-4773-bf8d-5b2f67254db1
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :blue]
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="gaze", ylabel="frequency")
    for s = 1:length(hmm_trained)
		Makie.stephist!(ax, gaze_data_subsampled[viterbi_states .== s], color=_colors[s], label="$s", normalization=:pdf)
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 5765329b-0d4e-46cf-9dcd-34defb942214
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="gaze", ylabel="frequency")
	Makie.stephist!(ax, raw_data.gaze, color=:gray, normalization=:pdf, bins=100)
	Makie.stephist!(ax, gaze_data_subsampled, color=:blue, normalization=:pdf)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 3e5ae845-1b8c-474a-a15b-cbfb47bc0c75
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=500, xticks=1:hmm_num_states, yticks=1:hmm_num_states, xlabel="initial state", ylabel="final state")
	plt = Makie.heatmap!(ax, float(hmm_trained.transition_matrix))
	Makie.Colorbar(fig[1,2], plt)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ b2ba5d73-5c91-46e8-a621-4da1b1cad5af
float(sum(hmm_trained.transition_matrix; dims=2))

# ╔═╡ 47b253ef-7642-4897-b1be-399b7ce9aef4
float(hmm_trained.transition_matrix)

# ╔═╡ 70a5231b-33d2-42e8-b49e-fa8b516f5dfb
[state.gaze for state = hmm_trained.emit]

# ╔═╡ Cell order:
# ╠═5b7dce2e-1f72-4885-92eb-4c2cbc6d60b9
# ╠═b5731209-833e-400a-bd91-0ca87807f3df
# ╠═b634b8dd-491f-4ad6-a177-7ef89744a41c
# ╠═3c9b3f3f-d145-4a52-85de-9052b5803de2
# ╠═c4a03051-84af-48e7-a3eb-e815517fff77
# ╠═e2fde7f0-fc5e-4ad7-b35c-1a6c91242a53
# ╠═5c003b8f-2e2b-498f-af41-033c6b7c691c
# ╠═cf43f299-64fe-4aeb-9a3c-4f963fa72525
# ╠═3bc16dc1-a7ec-4e01-9bb2-df7027a51ebb
# ╠═e8bb2a96-4542-4253-8a5b-4e3c40498086
# ╠═7a8f34d7-a193-4c61-b3e6-8d33c0e29d10
# ╠═405cd60c-5d49-4981-9254-210eb472fdb0
# ╠═dc200e3a-35f1-4078-b926-dca55611e6f3
# ╠═f2921685-b0c0-4915-90cd-9df3676ce0f0
# ╠═e0ec0817-e823-47d2-a6d3-c9b6852dd438
# ╠═fcd00cde-3ab4-4004-bb02-9484d707a5dc
# ╠═cf06a342-e143-4947-91f6-e140616a3282
# ╠═ce35a0bc-a905-447b-ba0a-ace0e1607b5d
# ╠═b24bcd54-bf05-4e33-946e-f5e18313d5b8
# ╠═3f0f82be-19d3-473e-b1ee-2a492f9f3044
# ╠═a80514e2-7ac3-46fd-855b-b10366b1759f
# ╠═a12bfd34-695c-453f-981d-8b3ac87c42c2
# ╠═33352bd9-ed35-454b-87da-a4e88cdc72e4
# ╠═c0c6176f-fe4e-4773-bf8d-5b2f67254db1
# ╠═5765329b-0d4e-46cf-9dcd-34defb942214
# ╠═3e5ae845-1b8c-474a-a15b-cbfb47bc0c75
# ╠═b2ba5d73-5c91-46e8-a621-4da1b1cad5af
# ╠═47b253ef-7642-4897-b1be-399b7ce9aef4
# ╠═70a5231b-33d2-42e8-b49e-fa8b516f5dfb
