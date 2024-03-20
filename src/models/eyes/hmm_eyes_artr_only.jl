struct HMM_Eyes_ARTR_Only <: HiddenMarkovModels.AbstractHMM
    pinit::Vector{LogFloat64} # initial state probabilities
    transition_matrix::Matrix{LogFloat64} # T[y,z] = P(y -> z)
    emit::Vector{NeuronsBinaryDistribution}
    pseudocount::Float64

    function HMM_Eyes_ARTR_Only(
        pinit::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        emit::AbstractVector{<:NeuronsBinaryDistribution},
        pseudocount::Real = 0.0
    )
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == length(emit)
        @assert all(≥(0), pinit)
        @assert isapprox(sum(pinit), 1; rtol=1e-8)
        @assert all(≥(0), transition_matrix)
        @assert sum(transition_matrix; dims=2) ≈ ones(size(transition_matrix, 1))
        @assert pseudocount ≥ 0
        return new(pinit, transition_matrix, emit, pseudocount)
    end
end

function HMM_Eyes_ARTR_Only(
    transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, pseudocount::Real = 0.0
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == size(h,2) # number of states
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    emit = [NeuronsBinaryDistribution(h[:,i]) for i = axes(h, 2)]
    return HMM_Eyes_ARTR_Only(pinit, transition_matrix, emit, pseudocount)
end

Base.length(hmm::HMM_Eyes_ARTR_Only) = size(hmm.transition_matrix, 1) # number of hidden states
HiddenMarkovModels.transition_matrix(hmm::HMM_Eyes_ARTR_Only) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_Eyes_ARTR_Only) = hmm.pinit
HiddenMarkovModels.obs_distribution(hmm::HMM_Eyes_ARTR_Only, i::Int) = hmm.emit[i]

function StatsAPI.fit!(
    hmm::HMM_Eyes_ARTR_Only,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector{<:AbstractVector{<:Real}},
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert dropdims(sum(state_marginals; dims=1); dims=1) ≈ ones(length(obs_seq))

    hmm.pinit .= init_count / sum(init_count)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    obs_p = stack(obs_seq) * (state_marginals ./ sum(state_marginals; dims=2))'
    @assert size(obs_p) == (length(first(obs_seq)), length(hmm))

    for i = 1:length(hmm)
        λ = hmm.pseudocount / length(obs_seq)
        q = (1 - λ) * obs_p[:,i] .+ λ * 0.5
        @assert all(0 .≤ q .≤ 1)
        logbias = log.(q ./ (1 .- q))

        hmm.emit[i] = NeuronsBinaryDistribution(logbias)
    end

    return hmm
end
