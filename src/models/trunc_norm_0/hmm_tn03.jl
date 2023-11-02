mutable struct ZebrafishHMM_TN03 <: HiddenMarkovModels.AbstractHMM
    #=
    Truncated normal (TN) turning angle emissions (zero mean).
    States are stored in this order:
        1. Forward (forward bouts)
        2. Left (left turning bouts)
        3. Right (right turning bouts)
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    forw::Normal{Float64}
    turn::Normal{Float64}

    function ZebrafishHMM_TN03(
        initial_probs::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        forw::Normal{<:Real},
        turn::Normal{<:Real}
    )
        length(initial_probs) == 3 || throw(ArgumentError("initial_probs should have 3 elements"))
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        return new(initial_probs, transition_matrix, forw, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_TN03) = length(hmm.initial_probs)

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_TN03)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_TN03)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_TN03, i::Int)
    if i == 1 # forward
        dist = hmm.forw
    elseif i == 2 # left
        dist = truncated(hmm.turn, nothing, 0)
    elseif i == 3 # right
        dist = truncated(hmm.turn, 0, nothing)
    else
        throw(ArgumentError("State index must be 1, 2, or 3; got $i"))
    end

    return dist
end

function normalize_initial_probs!(hmm::ZebrafishHMM_TN03)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_TN03)
    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_TN03)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_TN03, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities =#
    hmm.forw = fit_mle(typeof(hmm.forw), obs_seq, state_marginals[1,:]; mu = 0.0)

    #= Update left-right turn emission probabilities =#
    turn_marginals = ifelse.(obs_seq .< 0, state_marginals[2,:], state_marginals[3,:])
    hmm.turn = fit_mle(typeof(hmm.turn), abs.(obs_seq), turn_marginals; mu = 0.0)
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_TN03)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_TN03")
        write(h5, "initial_probs", hmm.initial_probs)
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "forw", collect(params(hmm.forw)))
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_TN03})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_TN03" || throw(ArgumentError("HMM type missmatch"))
        initial_probs = read(h5, "initial_probs")
        transition_matrix = read(h5, "transition_matrix")
        forw_params = read(h5, "forw")
        turn_params = read(h5, "turn")
        return ZebrafishHMM_TN03(initial_probs, transition_matrix, Normal(forw_params...), Normal(turn_params...))
    end
end
