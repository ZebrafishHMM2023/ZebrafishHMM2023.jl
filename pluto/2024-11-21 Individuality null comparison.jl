### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 82fb3325-81b1-4516-a57a-721f991282f7
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ a3deaeb2-12df-4f1e-97a8-e14dc8aab28f
using Statistics: median, std, mean

# ╔═╡ b1ddac8e-2fab-4d13-852c-6b19adb1594e
using Makie: @L_str

# ╔═╡ 2b137a46-b4c3-4f92-afea-6e3a6455015c
md"# Imports"

# ╔═╡ 9661ba40-3fe2-48a3-86b7-cb813a08a3b6
import ZebrafishHMM2023

# ╔═╡ e203e5fe-7f69-42f8-a531-4c44a649f628
import HiddenMarkovModels

# ╔═╡ 849d147c-80ff-458a-88a7-e07f6fa4c10f
import Makie

# ╔═╡ 98c4a3a9-8b3b-46e1-b0b2-aa5e6d8c7c94
import CairoMakie

# ╔═╡ 635f788f-4420-434f-90a5-7965b2824434
import PlutoUI

# ╔═╡ 67a90ef5-abde-4070-8df2-0322bd4229d9
import CSV

# ╔═╡ fb0ca60e-70f2-4a07-8f86-21617b8b76fc
import HDF5

# ╔═╡ d975eb75-d606-4071-b71e-5b495fb0c905
import Distributions

# ╔═╡ 36f82b92-1746-4265-9f83-0b0ee59cec8d
PlutoUI.TableOfContents()

# ╔═╡ 3c4ff1c7-8c9a-458b-94cb-6b4c3905a097
md"# Data"

# ╔═╡ b9614c9d-5f59-4faa-89c7-536c25c54ab5
trajs = ZebrafishHMM2023.legoc2021_single_fish_T26_trajs();

# ╔═╡ e9e2fd45-8652-45c1-a6b3-e0039f8eceb9
glob_all_trajs = [t for fish = trajs for t = fish]

# ╔═╡ 1b797630-87cd-4d9c-877d-55b1ad6d4b6f
md"""
# Train global HMM

HMM trained on ALL trajectories (of all fish) together.
"""

# ╔═╡ 6d46f555-a49b-4b38-96d4-06f401480c78
glob_hmm_init = ZebrafishHMM2023.ZebrafishHMM_G3_Sym(
    rand(),
    rand(3,3),
    1.0,
    Distributions.Gamma(1.1, 15.0),
    1.0
)

# ╔═╡ 127793d3-811f-4563-81f0-ae9bf2700410
# HMM trained on ALL trajectories together
begin
	(glob_hmm, glob_lLs) = HiddenMarkovModels.baum_welch(glob_hmm_init, glob_all_trajs, length(glob_all_trajs); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	ZebrafishHMM2023.normalize_all!(glob_hmm)
end

# ╔═╡ 3d2ebf0a-ed1a-40fd-a207-b3ddc7f76a54
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=900, height=300)
	Makie.lines!(ax, glob_lLs)
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ f379c0a7-8f76-435d-bdea-a6767fed3336
md"# Train HMM on all trajectories of each fish"

# ╔═╡ 58a69963-41a3-4d25-8d69-558bc5a413df
begin
	global_hmms = ZebrafishHMM2023.ZebrafishHMM_G4_Sym[]
	global_lLs = Vector{Float64}[]
	for (n, fish) = enumerate(trajs)
	    println("Fitting fish $n of 18.")
	    hmm = ZebrafishHMM2023.ZebrafishHMM_G4_Sym(
	        rand(),
	        rand(4,4),
	        1.0,
	        Distributions.Gamma(0.5, 15.0)
	    )
	    ZebrafishHMM2023.normalize_all!(hmm)
	    (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, fish, length(fish); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	    ZebrafishHMM2023.FL_FR_canon!(hmm)
	    push!(global_hmms, hmm)
	    push!(global_lLs, lL)
	end
end

# ╔═╡ bb8254ff-5c88-4478-926b-83f4acba0ad0
md"# Chunks"

# ╔═╡ f5b6fff6-e936-4cd5-bfb7-877914848fc8
n_chunks = 10

# ╔═╡ 91bd83a3-3ae6-411c-9675-e1b301f40e9d
function train_chunk_hmms(n_chunks::Int; fictitious::Bool = false)
	chunk_hmms = Vector{Vector{ZebrafishHMM2023.ZebrafishHMM_G3_Sym}}()
	chunk_lLs = Vector{Vector{Vector{Float64}}}()
	for (n, fish) = enumerate(trajs)
	    push!(chunk_hmms, ZebrafishHMM2023.ZebrafishHMM_G3_Sym[])
	    push!(chunk_lLs, Vector{Float64}[])
	    for (t, chunk) = enumerate(ZebrafishHMM2023.chunks(fish, n_chunks))
	        println("Fitting fish $n of 18, chunk $t (of length $(length(chunk))).")
			hmm = ZebrafishHMM2023.normalize_all!(ZebrafishHMM2023.ZebrafishHMM_G3_Sym(rand(), rand(3,3), 1.0, Distributions.Gamma(1.5, 20.0), 1.0))

			if fictitious
				# use same global HMM for all fish ... null model for lack of individuality
				my_data = [rand(glob_hmm, length(seq)).obs_seq for seq = chunk]
			else
				my_data = chunk
			end

			# my_data = chunk
			# @show length(my_data)
			
	        (hmm, lL) = HiddenMarkovModels.baum_welch(hmm, my_data, length(my_data); max_iterations = 5000, check_loglikelihood_increasing = false, atol = ZebrafishHMM2023.ATol(1e-5))
	        push!(chunk_hmms[n], hmm)
	        push!(chunk_lLs[n], lL)
	    end
	end
	return chunk_hmms, chunk_lLs
end

# ╔═╡ f1b3ed3c-1666-4cec-b221-b1f7d5be7f8f
chunk_hmms_null, chunk_lLs_null = train_chunk_hmms(n_chunks; fictitious=true)

# ╔═╡ 18f3f547-8f96-484d-8997-1072d3fd1877
chunk_hmms, chunk_lLs = train_chunk_hmms(n_chunks; fictitious=false)

# ╔═╡ b3d11d9b-eaac-413e-8aec-fa2ba6305ea4
_temp_fig_dir = mktempdir()

# ╔═╡ 777e1781-53fa-46fa-a950-a460e82f5aea
let fig = Makie.Figure()
	_opts = (; width=150, height=150, xlabel="global HMM", ylabel="<chunk HMMs>")

	ax = Makie.Axis(fig[1,1]; _opts..., title="std(forw)")
	Makie.errorbars!(ax, [hmm.σforw for hmm = global_hmms],
	    [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms_null],
	    [std(hmm.σforw for hmm = fish) for fish = chunk_hmms_null]/2; color=:red
	)
	Makie.errorbars!(ax, [hmm.σforw for hmm = global_hmms],
	    [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.σforw for hmm = fish) for fish = chunk_hmms] / 2; color=:blue
	)
	scatter_null = Makie.scatter!(ax, [hmm.σforw for hmm = global_hmms], [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms_null]; label="null", color=:red)
	scatter_real = Makie.scatter!(ax, [hmm.σforw for hmm = global_hmms], [mean(hmm.σforw for hmm = fish) for fish = chunk_hmms]; label="real", color=:blue)
	Makie.ylims!(ax, 1, 7)
	
	ax = Makie.Axis(fig[1,2]; _opts..., title="mean(turn)")
	Makie.errorbars!(ax, [mean(hmm.turn) for hmm = global_hmms], 
	    [mean(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms_null],
	    [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [mean(hmm.turn) for hmm = global_hmms], 
	    [mean(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms],
	    [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [mean(hmm.turn) for hmm = global_hmms], [mean(mean(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [mean(hmm.turn) for hmm = global_hmms], [mean(mean(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 20, 45)
	
	ax = Makie.Axis(fig[1,3]; _opts..., title="std(turn)")
	Makie.errorbars!(ax, [std(hmm.turn) for hmm = global_hmms], 
	    [mean(std(hmm.turn) for hmm = fish) for fish = chunk_hmms_null],
	    [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [std(hmm.turn) for hmm = global_hmms], 
	    [mean(std(hmm.turn) for hmm = fish) for fish = chunk_hmms],
	    [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [std(hmm.turn) for hmm = global_hmms], [mean(std(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [std(hmm.turn) for hmm = global_hmms], [mean(std(hmm.turn) for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 13, 28)

	Makie.Legend(fig[1,4], [scatter_null, scatter_real], ["Null model", "Real fish"])

	
	ax = Makie.Axis(fig[2,1]; _opts..., title="F->F")
	Makie.errorbars!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms_null],
	    [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], [mean(hmm.transition_matrix[1,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [hmm.transition_matrix[1,1] for hmm = global_hmms], [mean(hmm.transition_matrix[1,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 0, 0.7)
	
	ax = Makie.Axis(fig[2,2]; _opts..., title="F->L, F->R")
	Makie.errorbars!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms_null],
	    [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms],
	    [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], [mean(hmm.transition_matrix[1,2] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [hmm.transition_matrix[1,2] for hmm = global_hmms], [mean(hmm.transition_matrix[1,2] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 0.2, 0.5)
	
	ax = Makie.Axis(fig[2,3]; _opts..., title="L->L, R->R")
	Makie.errorbars!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null],
	    [std(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms],
	    [std(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [hmm.transition_matrix[3,3] for hmm = global_hmms], [mean(hmm.transition_matrix[3,3] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 0.1, 0.6)
	
	ax = Makie.Axis(fig[2,4]; _opts..., title="L->F, R->F")
	Makie.errorbars!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null],
	    [std(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null] / 2; color=:red
	)
	Makie.errorbars!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], 
	    [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms],
	    [std(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms] / 2; color=:blue
	)
	Makie.scatter!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms_null]; color=:red)
	Makie.scatter!(ax, [hmm.transition_matrix[3,1] for hmm = global_hmms], [mean(hmm.transition_matrix[3,1] for hmm = chunk_hmm) for chunk_hmm = chunk_hmms]; color=:blue)
	Makie.ylims!(ax, 0.2, 0.7)
	
	Makie.resize_to_layout!(fig)
	Makie.save(joinpath(_temp_fig_dir, "Sup5_new.pdf"), fig)
	fig
end

# ╔═╡ 0acd20a5-3015-4fa9-8f90-8ae2f09c5f32
joinpath(_temp_fig_dir, "Sup5_new.pdf")

# ╔═╡ 271fbcca-b96b-467c-815f-074ae031b76b
let fig = Makie.Figure()
	_sz = 200

	ax = Makie.Axis(fig[1,1], title=L"\sigma_F", width=_sz, height=_sz)
	hist_real = Makie.hist!(ax, [hmm.σforw - mean(hmm.σforw for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue)
	hist_null = Makie.stephist!(ax, [hmm.σforw - mean(hmm.σforw for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2)

	ax = Makie.Axis(fig[1,2], title=L"\mu_T", width=_sz, height=_sz)
	Makie.hist!(ax, [mean(hmm.turn) - mean(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue)
	Makie.stephist!(ax, [mean(hmm.turn) - mean(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2)

	ax = Makie.Axis(fig[1,3], title=L"\sigma_T", width=_sz, height=_sz)
	Makie.hist!(ax, [std(hmm.turn) - mean(std(hmm.turn) for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue)
	Makie.stephist!(ax, [std(hmm.turn) - mean(std(hmm.turn) for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2)

	ax = Makie.Axis(fig[2,1], title=L"P(F\rightarrow F)", width=_sz, height=_sz)
	Makie.hist!(ax, [hmm.transition_matrix[1,1] - mean(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue)
	Makie.stephist!(ax, [hmm.transition_matrix[1,1] - mean(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2)
	Makie.resize_to_layout!(fig)

	ax = Makie.Axis(fig[2,2], title=L"P(F\rightarrow L) = P(F\rightarrow R)", width=_sz, height=_sz)
	Makie.hist!(ax, [hmm.transition_matrix[1,2] - mean(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue)
	Makie.stephist!(ax, [hmm.transition_matrix[1,2] - mean(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2)
	Makie.resize_to_layout!(fig)


	ax = Makie.Axis(fig[2,3], title=L"P(L\rightarrow L) = P(R\rightarrow R)", width=_sz, height=_sz)
	Makie.hist!(ax, [hmm.transition_matrix[3,3] - mean(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue, bins=-0.3:0.02:0.3)
	Makie.stephist!(ax, [hmm.transition_matrix[3,3] - mean(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2, bins=-0.3:0.02:0.3)

	ax = Makie.Axis(fig[2,4], title=L"P(L\rightarrow F) = P(R\rightarrow F)", width=_sz, height=_sz)
	Makie.hist!(ax, [hmm.transition_matrix[3,1] - mean(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms for hmm = fish]; normalization=:pdf, color=:blue, bins=-0.3:0.02:0.3)
	Makie.stephist!(ax, [hmm.transition_matrix[3,1] - mean(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms_null for hmm = fish]; normalization=:pdf, color=:red, linewidth=2, bins=-0.3:0.02:0.3)

	Makie.Legend(fig[1,4], [hist_null, hist_real], ["Null model", "Real fish"])

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 74d8ec02-8784-4886-ac91-975b4787126c
median([std(hmm.σforw for hmm = fish) for fish = chunk_hmms]), median([std(hmm.σforw for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 3fd8f38f-5f3a-4caa-b818-a052f63ab636
median([std(hmm.σforw for hmm = fish) for fish = chunk_hmms]), median([std(hmm.σforw for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 18385ce6-1aa9-420c-9f0c-9c7e4b35c613
let fig = Makie.Figure()
	_opts = (; width=150, height=150, xlabel="Real fish", ylabel="Null model")

	ax = Makie.Axis(fig[1,1]; _opts..., title="std(forw)")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(hmm.σforw for hmm = fish) for fish = chunk_hmms], [std(hmm.σforw for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 2)
	Makie.ylims!(ax, 0, 2)
	
	ax = Makie.Axis(fig[1,2]; _opts..., title="mean(turn)")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms], [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 8)
	Makie.ylims!(ax, 0, 8)
	
	ax = Makie.Axis(fig[1,3]; _opts..., title="std(turn)")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms], [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 5)
	Makie.ylims!(ax, 0, 5)
	
	ax = Makie.Axis(fig[2,1]; _opts..., title="F->F")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms], [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 0.25)
	Makie.ylims!(ax, 0, 0.25)

	ax = Makie.Axis(fig[2,2]; _opts..., title="F->L, F->R")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms], [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 0.12)
	Makie.ylims!(ax, 0, 0.12)
	
	ax = Makie.Axis(fig[2,3]; _opts..., title="L->L, R->R")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms], [std(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 0.15)
	Makie.ylims!(ax, 0, 0.15)

	ax = Makie.Axis(fig[2,4]; _opts..., title="L->F, R->F")
	Makie.ablines!(ax, 0, 1; color=:red, linestyle=:dash)
	Makie.scatter!(ax, [std(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms], [std(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms_null])
	Makie.xlims!(ax, 0, 0.23)
	Makie.ylims!(ax, 0, 0.23)
	
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 38b4dcab-4b0c-47e3-9aeb-21182fa75ed9
mean([std(hmm.σforw for hmm = fish) for fish = chunk_hmms] ./ [std(hmm.σforw for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 3b9291d9-83b1-489e-81fd-4e40a40d12c9
mean([std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms] ./ [std(mean(hmm.turn) for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 86688985-f6b6-438b-934e-66a0ca65b046
mean([std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms] ./ [std(std(hmm.turn) for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 7dfa8477-b62b-4597-abe2-86ff3b73cae7
mean([std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms] ./ [std(hmm.transition_matrix[1,1] for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 3af53999-c343-440b-80bc-e7e61ca28344
mean([std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms] ./ [std(hmm.transition_matrix[1,2] for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ bb82cbe0-0d5b-4a64-9763-b8b21bc2c75e
mean([std(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms] ./ [std(hmm.transition_matrix[3,3] for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ 2cf548b9-a703-4293-bff3-197f98b782f8
mean([std(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms] ./ [std(hmm.transition_matrix[3,1] for hmm = fish) for fish = chunk_hmms_null])

# ╔═╡ Cell order:
# ╠═2b137a46-b4c3-4f92-afea-6e3a6455015c
# ╠═82fb3325-81b1-4516-a57a-721f991282f7
# ╠═9661ba40-3fe2-48a3-86b7-cb813a08a3b6
# ╠═e203e5fe-7f69-42f8-a531-4c44a649f628
# ╠═849d147c-80ff-458a-88a7-e07f6fa4c10f
# ╠═98c4a3a9-8b3b-46e1-b0b2-aa5e6d8c7c94
# ╠═635f788f-4420-434f-90a5-7965b2824434
# ╠═67a90ef5-abde-4070-8df2-0322bd4229d9
# ╠═fb0ca60e-70f2-4a07-8f86-21617b8b76fc
# ╠═d975eb75-d606-4071-b71e-5b495fb0c905
# ╠═a3deaeb2-12df-4f1e-97a8-e14dc8aab28f
# ╠═b1ddac8e-2fab-4d13-852c-6b19adb1594e
# ╠═36f82b92-1746-4265-9f83-0b0ee59cec8d
# ╠═3c4ff1c7-8c9a-458b-94cb-6b4c3905a097
# ╠═b9614c9d-5f59-4faa-89c7-536c25c54ab5
# ╠═e9e2fd45-8652-45c1-a6b3-e0039f8eceb9
# ╠═1b797630-87cd-4d9c-877d-55b1ad6d4b6f
# ╠═6d46f555-a49b-4b38-96d4-06f401480c78
# ╠═127793d3-811f-4563-81f0-ae9bf2700410
# ╠═3d2ebf0a-ed1a-40fd-a207-b3ddc7f76a54
# ╠═f379c0a7-8f76-435d-bdea-a6767fed3336
# ╠═58a69963-41a3-4d25-8d69-558bc5a413df
# ╠═bb8254ff-5c88-4478-926b-83f4acba0ad0
# ╠═f5b6fff6-e936-4cd5-bfb7-877914848fc8
# ╠═91bd83a3-3ae6-411c-9675-e1b301f40e9d
# ╠═f1b3ed3c-1666-4cec-b221-b1f7d5be7f8f
# ╠═18f3f547-8f96-484d-8997-1072d3fd1877
# ╠═b3d11d9b-eaac-413e-8aec-fa2ba6305ea4
# ╠═777e1781-53fa-46fa-a950-a460e82f5aea
# ╠═0acd20a5-3015-4fa9-8f90-8ae2f09c5f32
# ╠═271fbcca-b96b-467c-815f-074ae031b76b
# ╠═74d8ec02-8784-4886-ac91-975b4787126c
# ╠═3fd8f38f-5f3a-4caa-b818-a052f63ab636
# ╠═18385ce6-1aa9-420c-9f0c-9c7e4b35c613
# ╠═38b4dcab-4b0c-47e3-9aeb-21182fa75ed9
# ╠═3b9291d9-83b1-489e-81fd-4e40a40d12c9
# ╠═86688985-f6b6-438b-934e-66a0ca65b046
# ╠═7dfa8477-b62b-4597-abe2-86ff3b73cae7
# ╠═3af53999-c343-440b-80bc-e7e61ca28344
# ╠═bb82cbe0-0d5b-4a64-9763-b8b21bc2c75e
# ╠═2cf548b9-a703-4293-bff3-197f98b782f8
