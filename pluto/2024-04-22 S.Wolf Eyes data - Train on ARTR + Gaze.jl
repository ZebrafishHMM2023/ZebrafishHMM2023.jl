### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 34fc02ca-be22-4200-8dc4-6a0c9d7806bb
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ ec9792cc-1651-46f2-9eb4-216321dde7a7
using Statistics: mean

# ╔═╡ 4f5f45fe-d114-44ca-acad-455afba4c3e3
using Distributions: Normal

# ╔═╡ 7b22ff78-00b2-11ef-1f6c-f5efb95c8842
md"# Imports"

# ╔═╡ 0e3287c8-fef7-440e-ac3f-13af4b8d1b7d
import CairoMakie

# ╔═╡ 0edcb025-cb43-4a65-b661-1cc243c4dca6
import Makie

# ╔═╡ 4ce66144-2c40-4f3a-bf0d-8d598d06abf3
import ZebrafishHMM2023

# ╔═╡ 425f6888-2605-458b-8443-940b904521e3
import HiddenMarkovModels

# ╔═╡ 12045797-b2a1-48ce-a3cf-bee1f58c169e
md"# Functions"

# ╔═╡ 7b961547-7942-46ff-81ff-44d8b537911d
md"# Analysis"

# ╔═╡ f233f5c5-cfca-4dd9-bff5-e70ed2f2bd3f
raw_data = ZebrafishHMM2023.wolf_eyes_20240422_data()

# ╔═╡ 49adf835-2d1e-400e-bc31-db21c43c2489
gaze_data_subsampled = map(mean, ZebrafishHMM2023.chunks(raw_data.gaze, size(raw_data.left, 2)))

# ╔═╡ 00c78793-79aa-4d70-9c4f-1e245e4e0fb9
aggregated_data = collect(zip(gaze_data_subsampled, eachcol(vcat(raw_data.left, raw_data.right))))

# ╔═╡ 110e289a-be91-440f-a28b-d1f761859286
num_neurons_artr = size(raw_data.left, 1) + size(raw_data.right, 1)

# ╔═╡ fe3eaeb8-a03f-4173-aa85-c143065ddbce
hmm_num_states = 3

# ╔═╡ e18d08cd-f786-46a8-877b-28c89a122b5c
hmm_init = ZebrafishHMM2023.HMM_Gaze_ARTR(
	ZebrafishHMM2023.normalize_transition_matrix(rand(hmm_num_states, hmm_num_states)),
	randn(hmm_num_states), ones(hmm_num_states), randn(num_neurons_artr, hmm_num_states), 1.0
)

# ╔═╡ cb5b06de-82a1-41e9-b6ae-247a56f275a8
hmm_trained, train_lls = HiddenMarkovModels.baum_welch(
	hmm_init, aggregated_data; max_iterations = 500, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-7)
)

# ╔═╡ 80a3fa8e-7766-4ca8-b00d-ec0fb5e9b344
viterbi_states = HiddenMarkovModels.viterbi(hmm_trained, aggregated_data)

# ╔═╡ d1c34427-3ffd-4275-9444-cd30ca7a8675
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="mL", ylabel="mR")
    for s = 1:length(hmm_trained)
        Makie.scatter!(ax,
            vec(mean(raw_data.left[:, viterbi_states .== s]; dims=1)),
            vec(mean(raw_data.right[:, viterbi_states .== s]; dims=1));
            markersize=7, color=(_colors[s], 0.25), label="$s"
        )
    end
    Makie.xlims!(ax, -0.01, 0.5)
    Makie.ylims!(ax, -0.01, 0.5)
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ c61c3528-9db4-48cb-aed2-cb69c994bcb5
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=1500, height=300, xlabel="time", ylabel="gaze")
	Makie.lines!(ax, gaze_data_subsampled, color=:gray)
    for s = 1:length(hmm_trained)
		Makie.scatter!(ax, findall(viterbi_states .== s), gaze_data_subsampled[viterbi_states .== s], markersize=7, color=_colors[s], label="$s")
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 220e73ea-e2ea-4238-bb13-2885d1780ed3
let fig = Makie.Figure()
    _colors = [:teal, :orange, :purple, :pink]
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="gaze", ylabel="frequency")
    for s = 1:length(hmm_trained)
		Makie.stephist!(ax, gaze_data_subsampled[viterbi_states .== s], color=_colors[s], label="$s", normalization=:pdf)
    end
    Makie.axislegend(ax)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 127d61fb-149c-40fd-a92d-67ef26f5fb0f
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=500, xlabel="gaze", ylabel="frequency")
	Makie.stephist!(ax, raw_data.gaze, color=:gray, normalization=:pdf, bins=100)
	Makie.stephist!(ax, gaze_data_subsampled, color=:blue, normalization=:pdf)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 0363799b-49d7-436f-9b8c-a30a516765b6
let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=500, height=500, xticks=(1:3, ["L", "F", "R"]), yticks=(1:3, ["L", "F", "R"]))
	plt = Makie.heatmap!(ax, float(hmm_trained.transition_matrix))
	Makie.Colorbar(fig[1,2], plt)
    Makie.resize_to_layout!(fig)
    fig
end

# ╔═╡ 60337847-ed31-4301-aaf9-fbdb066cf644
float(sum(hmm_trained.transition_matrix; dims=2))

# ╔═╡ 61775836-44e1-432a-a320-77a140421ad0
float(hmm_trained.transition_matrix)

# ╔═╡ 9e385a52-d848-4044-914d-fbd7e4afd7b7
[state.gaze for state = hmm_trained.emit]

# ╔═╡ Cell order:
# ╠═7b22ff78-00b2-11ef-1f6c-f5efb95c8842
# ╠═34fc02ca-be22-4200-8dc4-6a0c9d7806bb
# ╠═0e3287c8-fef7-440e-ac3f-13af4b8d1b7d
# ╠═0edcb025-cb43-4a65-b661-1cc243c4dca6
# ╠═4ce66144-2c40-4f3a-bf0d-8d598d06abf3
# ╠═425f6888-2605-458b-8443-940b904521e3
# ╠═ec9792cc-1651-46f2-9eb4-216321dde7a7
# ╠═4f5f45fe-d114-44ca-acad-455afba4c3e3
# ╠═12045797-b2a1-48ce-a3cf-bee1f58c169e
# ╠═7b961547-7942-46ff-81ff-44d8b537911d
# ╠═f233f5c5-cfca-4dd9-bff5-e70ed2f2bd3f
# ╠═49adf835-2d1e-400e-bc31-db21c43c2489
# ╠═00c78793-79aa-4d70-9c4f-1e245e4e0fb9
# ╠═110e289a-be91-440f-a28b-d1f761859286
# ╠═fe3eaeb8-a03f-4173-aa85-c143065ddbce
# ╠═e18d08cd-f786-46a8-877b-28c89a122b5c
# ╠═cb5b06de-82a1-41e9-b6ae-247a56f275a8
# ╠═80a3fa8e-7766-4ca8-b00d-ec0fb5e9b344
# ╠═d1c34427-3ffd-4275-9444-cd30ca7a8675
# ╠═c61c3528-9db4-48cb-aed2-cb69c994bcb5
# ╠═220e73ea-e2ea-4238-bb13-2885d1780ed3
# ╠═127d61fb-149c-40fd-a92d-67ef26f5fb0f
# ╠═0363799b-49d7-436f-9b8c-a30a516765b6
# ╠═60337847-ed31-4301-aaf9-fbdb066cf644
# ╠═61775836-44e1-432a-a320-77a140421ad0
# ╠═9e385a52-d848-4044-914d-fbd7e4afd7b7
