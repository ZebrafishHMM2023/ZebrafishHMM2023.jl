#= Freeze some parameters of the model =#
struct HMM_ARTR_Log_Freeze <: HiddenMarkovModels.AbstractHMM
    transition_matrix::Matrix{LogFloat64} # T[y,z] = P(y -> z)
    h::Matrix{Float64} # h[:,z] = fields in hidden state 'z'
    pinit::Vector{LogFloat64} # initial state probabilities
    pseudocount::Float64 # pseudocount for the inference of 'h'

    transition_matrix_frozen::Bool
    h_frozen::Bool
    pinit_frozen::Bool

    function HMM_ARTR_Log_Freeze(
        transition_matrix::AbstractMatrix{<:Real},
        h::AbstractMatrix{<:Real},
        pinit::AbstractVector{<:Real},
        pseudocount::Float64,
        transition_matrix_frozen::Bool,
        h_frozen::Bool,
        pinit_frozen::Bool
    )
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
        @assert all(≥(0), transition_matrix)
        @assert sum(transition_matrix; dims=2) ≈ ones(size(transition_matrix, 1))
        #@assert all(≈(1), sum(transition_matrix; dims=2))
        @assert all(≥(0), pinit)
        @assert isapprox(sum(pinit), 1; rtol=1e-8)
        @assert pseudocount ≥ 0
        return new(transition_matrix, h, pinit, pseudocount, transition_matrix_frozen, h_frozen, pinit_frozen)
    end
end

function HMM_ARTR_Log_Freeze(
    ;
    transition_matrix::AbstractMatrix{<:Real}, pinit::AbstractVector{<:Real},
    h::AbstractMatrix{<:Real}, pseudocount::Real = 0.0,
    transition_matrix_frozen::Bool = false, h_frozen::Bool = false, pinit_frozen::Bool = false
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2) == length(pinit)
    return HMM_ARTR_Log_Freeze(transition_matrix, h, pinit, pseudocount, transition_matrix_frozen, h_frozen, pinit_frozen)
end

function HMM_ARTR_Log_Freeze(
    model::HMM_ARTR_Log;
    transition_matrix_frozen::Bool = false, h_frozen::Bool = false, pinit_frozen::Bool = false
)
    return HMM_ARTR_Log_Freeze(
        ; model.transition_matrix, model.h, model.pinit, model.pseudocount,
        transition_matrix_frozen, h_frozen, pinit_frozen
    )
end

function HMM_ARTR_Log(model::HMM_ARTR_Log_Freeze)
    return HMM_ARTR_Log(model.transition_matrix, model.h, model.pinit, model.pseudocount)
end

# number of hidden states
Base.length(hmm::HMM_ARTR_Log_Freeze) = size(hmm.transition_matrix, 1)

HiddenMarkovModels.transition_matrix(hmm::HMM_ARTR_Log_Freeze) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_ARTR_Log_Freeze) = hmm.pinit

function HiddenMarkovModels.obs_distribution(hmm::HMM_ARTR_Log_Freeze, i::Int)
    return NeuronsBinaryDistribution(view(hmm.h, :, i))
end

function StatsAPI.fit!(
    hmm::HMM_ARTR_Log_Freeze,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector,
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert dropdims(sum(state_marginals; dims=1); dims=1) ≈ ones(length(obs_seq))

    if !hmm.pinit_frozen
        hmm.pinit .= normalize(init_count, 1)
    end

    if !hmm.transition_matrix_frozen
        hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)
    end

    if !hmm.h_frozen
        obs_mat = stack(obs_seq)
        @assert size(obs_mat) == (size(hmm.h, 1), length(obs_seq))

        obs_p = stack(obs_seq) * (state_marginals ./ sum(state_marginals; dims=2))'
        @assert size(obs_p) == size(hmm.h)

        λ = hmm.pseudocount / length(obs_seq)
        q = (1 - λ) * obs_p .+ λ * 0.5
        @assert all(0 .≤ q .≤ 1)

        hmm.h .= log.(q ./ (1 .- q))
    end

    return hmm
end

function save_hmm(path::AbstractString, hmm::HMM_ARTR_Log_Freeze)
    h5open(path, "w") do h5
        write(h5, "type", "HMM_ARTR_Log")
        write(h5, "transition_matrix", float(hmm.transition_matrix))
        write(h5, "h", float(hmm.h))
        write(h5, "pinit", float(hmm.pinit))
        write(h5, "pseudocount", [hmm.pseudocount])
    end
end

function load_hmm(path::AbstractString, ::Type{HMM_ARTR_Log_Freeze})
    h5open(path, "r") do h5
        read(h5, "type") == "HMM_ARTR_Log" || throw(ArgumentError("HMM type missmatch"))
        transition_matrix = read(h5, "transition_matrix")
        h = read(h5, "h")
        pinit = read(h5, "pinit")
        pseudocount = only(read(h5, "pseudocount"))
        return HMM_ARTR_Log(transition_matrix, h, pinit, pseudocount)
    end
end
