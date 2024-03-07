struct HMM_ARTR_Log <: HiddenMarkovModels.AbstractHMM
    transition_matrix::Matrix{LogFloat64} # T[y,z] = P(y -> z)
    h::Matrix{Float64} # h[:,z] = fields in hidden state 'z'
    pinit::Vector{LogFloat64} # initial state probabilities
    pseudocount::Float64 # pseudocount for the inference of 'h'

    function HMM_ARTR_Log(
        transition_matrix::AbstractMatrix{<:Real},
        h::AbstractMatrix{<:Real},
        pinit::AbstractVector{<:Real},
        pseudocount::Float64
    )
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
        @assert all(≥(0), transition_matrix)
        #@assert all(≈(1), sum(transition_matrix; dims=2))
        @assert sum(transition_matrix; dims=2) ≈ ones(size(transition_matrix, 1))
        @assert all(≥(0), pinit)
        @assert isapprox(sum(pinit), 1; rtol=1e-8)
        @assert pseudocount ≥ 0
        return new(transition_matrix, h, pinit, pseudocount)
    end
end

function HMM_ARTR_Log(
    transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, pseudocount::Real = 0.0
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    return HMM_ARTR_Log(transition_matrix, h, pinit, pseudocount)
end

# number of hidden states
Base.length(hmm::HMM_ARTR_Log) = size(hmm.transition_matrix, 1)

HiddenMarkovModels.transition_matrix(hmm::HMM_ARTR_Log) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_ARTR_Log) = hmm.pinit

function HiddenMarkovModels.obs_distribution(hmm::HMM_ARTR_Log, i::Int)
    return NeuronsBinaryDistribution(view(hmm.h, :, i))
end

function StatsAPI.fit!(
    hmm::HMM_ARTR_Log,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector,
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert dropdims(sum(state_marginals; dims=1); dims=1) ≈ ones(length(obs_seq))

    hmm.pinit .= normalize(init_count, 1)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    obs_mat = stack(obs_seq)
    @assert size(obs_mat) == (size(hmm.h, 1), length(obs_seq))

    obs_p = stack(obs_seq) * (state_marginals ./ sum(state_marginals; dims=2))'
    @assert size(obs_p) == size(hmm.h)

    λ = hmm.pseudocount / length(obs_seq)
    q = (1 - λ) * obs_p .+ λ * 0.5
    @assert all(0 .≤ q .≤ 1)

    hmm.h .= log.(q ./ (1 .- q))
    return hmm
end

function save_hmm(path::AbstractString, hmm::HMM_ARTR_Log)
    h5open(path, "w") do h5
        write(h5, "type", "HMM_ARTR_Log")
        write(h5, "transition_matrix", float(hmm.transition_matrix))
        write(h5, "h", float(hmm.h))
        write(h5, "pinit", float(hmm.pinit))
        write(h5, "pseudocount", [hmm.pseudocount])
    end
end

function load_hmm(path::AbstractString, ::Type{HMM_ARTR_Log})
    h5open(path, "r") do h5
        read(h5, "type") == "HMM_ARTR_Log" || throw(ArgumentError("HMM type missmatch"))
        transition_matrix = read(h5, "transition_matrix")
        h = read(h5, "h")
        pinit = read(h5, "pinit")
        pseudocount = only(read(h5, "pseudocount"))
        return HMM_ARTR_Log(transition_matrix, h, pinit, pseudocount)
    end
end

function easy_train_artr_hmm(
    ; temperature::Int, fish::Int, nstates::Int = 3, verbose::Bool=false,
    max_iterations = 200, atol = 1e-7, pseudocount=5.0
)
    verbose && println("Training on temperature = $temperature, fish = $fish ...")

    data = load_artr_wolf_2023(; temperature, fish)
    trajs = collect(eachcol(vcat(data.left, data.right)))

    Nleft = size(data.left, 1)
    Nright = size(data.right, 1)
    Nneurons = Nleft + Nright

    hmm = HMM_ARTR_Log(normalize_transition_matrix(rand(nstates, nstates)), randn(Nneurons, nstates), pseudocount)
    (hmm, lL) = HiddenMarkovModels.baum_welch(
        hmm, trajs; max_iterations, check_loglikelihood_increasing = false, atol = ATol(atol)
    )

    # identify states corresponding to L, R, F
    Rstate, Fstate, Lstate = sortperm([mean(hmm.h[1:Nleft, z]) - mean(hmm.h[Nleft + 1:end, z]) for z = 1:3])

    # sort states (F = 1, L = 2, R = 3)
    hmm.transition_matrix .= hmm.transition_matrix[[Fstate, Lstate, Rstate], [Fstate, Lstate, Rstate]]
    hmm.h .= hmm.h[:, [Fstate, Lstate, Rstate]]
    hmm.pinit .= hmm.pinit[[Fstate, Lstate, Rstate]]

    return (; hmm, lL)
end
