struct Gaze_ARTR_Emit
    gaze::Normal{Float64} # Normal gaze distribution
    logbias::Vector{Float64} # Fields for the ARTR neural activity
    pseudocount::Float64 # regularize inference of fields
    function Gaze_ARTR_Emit(gaze::Normal, logbias::AbstractVector{<:Real}, pseudocount::Real = 0.0)
        @assert all(isfinite, logbias)
        @assert pseudocount ≥ 0
        return new(gaze, logbias, pseudocount)
    end
end

struct HMM_Gaze_ARTR <: HiddenMarkovModels.AbstractHMM
    pinit::Vector{LogFloat64} # initial state probabilities
    transition_matrix::Matrix{LogFloat64} # T[y,z] = P(y -> z)
    emit::Vector{Gaze_ARTR_Emit}

    function HMM_Gaze_ARTR(
        pinit::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        emit::AbstractVector{<:Gaze_ARTR_Emit}
    )
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == length(emit)
        @assert all(≥(0), pinit)
        @assert isapprox(sum(pinit), 1; rtol=1e-8)
        @assert all(≥(0), transition_matrix)
        @assert sum(transition_matrix; dims=2) ≈ ones(size(transition_matrix, 1))
        return new(pinit, transition_matrix, emit)
    end
end

function DensityInterface.logdensityof(d::Gaze_ARTR_Emit, (gaze, artr)::Tuple{Real,AbstractVector})
    return _Gaze_ARTR_Emit_logpdf(d, gaze, artr)
end

function _Gaze_ARTR_Emit_logpdf(d::Gaze_ARTR_Emit, gaze::Real, artr::AbstractVector{<:Real})
    return logpdf(d.gaze, gaze) + dot(d.logbias, artr) - sum(log1pexp, d.logbias)
end

function Base.rand(rng::AbstractRNG, d::Gaze_ARTR_Emit)
    gaze = rand(rng, d.gaze)
    artr = rand.(rng) .< logistic.(d.h)
    return (gaze, artr)
end

function HMM_Gaze_ARTR(
    transition_matrix::AbstractMatrix{<:Real},
    μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, h::AbstractMatrix{<:Real},
    pseudocount::Real = 0.0
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == length(μ) == length(σ) == size(h,2) # number of states
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    emit = [Gaze_ARTR_Emit(Normal(μ[i], σ[i]), h[:,i], pseudocount) for i = eachindex(μ)]
    return HMM_Gaze_ARTR(pinit, transition_matrix, emit)
end

Base.length(hmm::HMM_Gaze_ARTR) = size(hmm.transition_matrix, 1) # number of hidden states
HiddenMarkovModels.transition_matrix(hmm::HMM_Gaze_ARTR) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_Gaze_ARTR) = hmm.pinit
HiddenMarkovModels.obs_distribution(hmm::HMM_Gaze_ARTR, i::Int) = hmm.emit[i]

function StatsAPI.fit!(
    hmm::HMM_Gaze_ARTR,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector{<:Tuple{Real, AbstractVector{<:Real}}},
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert dropdims(sum(state_marginals; dims=1); dims=1) ≈ ones(length(obs_seq))

    hmm.pinit .= init_count / sum(init_count)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    obs_p = stack(last.(obs_seq)) * (state_marginals ./ sum(state_marginals; dims=2))'
    @assert size(obs_p) == (length(obs_seq[1][2]), length(hmm))

    for i = 1:length(hmm)
        gaze = fit_mle(typeof(hmm.emit[i].gaze), first.(obs_seq), float(state_marginals[i,:]))

        λ = hmm.emit[i].pseudocount / length(obs_seq)
        q = (1 - λ) * obs_p[:,i] .+ λ * 0.5
        @assert all(0 .≤ q .≤ 1)
        logbias = log.(q ./ (1 .- q))

        hmm.emit[i] = Gaze_ARTR_Emit(gaze, logbias, hmm.emit[i].pseudocount)
    end

    return hmm
end

# load gaze-artr data in a convenient manner
function gaze_artr_data()
    gaze_data = vec(wolf_eyes_data()["gaze"]["orient"])
    gaze_data_subsampled = map(mean, Iterators.partition(gaze_data, 20))

    artr_data = wolf_eyes_artr_data()

    return collect(zip(gaze_data_subsampled, eachcol(vcat(artr_data.left, artr_data.right))))
end
