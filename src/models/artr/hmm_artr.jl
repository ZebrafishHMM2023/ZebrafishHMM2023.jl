struct HMM_ARTR <: HiddenMarkovModels.AbstractHMM
    transition_matrix::Matrix{Float64} # T[i,j] = P(i -> j)
    h::Matrix{Float64} # h[:,i] = fields in hidden state 'i'
    pinit::AbstractVector{Float64} # initial state probabilities
    pseudocount::Float64 # pseudocount for the inference of 'h'
    function HMM_ARTR(transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, pinit::AbstractVector{<:Real}, pseudocount::Float64)
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
        @assert all(≥(0), transition_matrix)
        @assert all(≈(1), sum(transition_matrix; dims=2))
        @assert all(≥(0), pinit)
        @assert sum(pinit) ≈ 1
        @assert pseudocount ≥ 0
        return new(transition_matrix, h, pinit, pseudocount)
    end
end

struct NeuronsBinaryDistribution{V <: AbstractVector}
    h::V
end

function HMM_ARTR(transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, pseudocount::Real=0.0)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    return HMM_ARTR(transition_matrix, h, pinit, pseudocount)
end

function Base.rand(rng::AbstractRNG, d::NeuronsBinaryDistribution)
    return rand.(rng) .< logistic.(d.h)
end

function DensityInterface.logdensityof(d::NeuronsBinaryDistribution, x::AbstractVector{<:Real})
    return dot(d.h, x) - sum(log1pexp, d.h)
end

Base.length(hmm::HMM_ARTR) = size(hmm.transition_matrix, 1) # number of hidden states
HiddenMarkovModels.transition_matrix(hmm::HMM_ARTR) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_ARTR) = hmm.pinit

function HiddenMarkovModels.obs_distribution(hmm::HMM_ARTR, i::Int)
    return NeuronsBinaryDistribution(view(hmm.h, :, i))
end

function normalize_transition_matrix!(hmm::HMM_ARTR)
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix; dims=2)
    return hmm.transition_matrix
end

function StatsAPI.fit!(hmm::HMM_ARTR, init_count::AbstractVector, trans_count::AbstractMatrix, obs_seq::AbstractVector, state_marginals::AbstractMatrix)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert all(≈(1), sum(state_marginals; dims=1))

    hmm.pinit .= normalize(init_count, 1)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    obs_mat = stack(obs_seq)
    @assert size(obs_mat) == (size(hmm.h, 1), length(obs_seq))

    obs_p = stack(obs_seq) * (state_marginals ./ sum(state_marginals; dims=2))'
    @assert size(obs_p) == size(hmm.h)

    λ = hmm.pseudocount / length(obs_seq)
    q = (1 - λ) * obs_p .+ λ * 0.5
    @assert all(0 .≤ q .≤ 1)

    hmm.h .= log.(q)
    return hmm
end

function save_hmm(path::AbstractString, hmm::HMM_ARTR)
    h5open(path, "w") do h5
        write(h5, "type", "HMM_ARTR")
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "h", hmm.h)
        write(h5, "pinit", hmm.pinit)
        write(h5, "h_abs_max", [h_abs_max])
    end
end

function load_hmm(path::AbstractString, ::Type{HMM_ARTR})
    h5open(path, "r") do h5
        read(h5, "type") == "HMM_ARTR" || throw(ArgumentError("HMM type missmatch"))
        transition_matrix = read(h5, "transition_matrix")
        h = read(h5, "h")
        pinit = read(h5, "pinit")
        h_abs_max = only(read(h5, "h_abs_max"))
        return HMM_ARTR(transition_matrix, h, pinit, h_abs_max)
    end
end
