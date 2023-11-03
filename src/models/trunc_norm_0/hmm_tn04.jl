mutable struct ZebrafishHMM_TN04 <: HiddenMarkovModels.AbstractHMM
    #=
    Truncated normal (TN) turning angle emissions (zero mean). States are stored in this order:
        1. Forward-Left (forward bouts, but last turn was left)
        2. Forward-Right (forward bouts, but last turn was right)
        3. Left (left turning bouts)
        4. Right (right turning bouts)
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64}
    forw::Normal{Float64}
    turn::Normal{Float64}

    function ZebrafishHMM_TN04(
        initial_probs::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        forw::Normal{<:Real}, turn::Normal{<:Real}
    )
        length(initial_probs) == 4 || throw(ArgumentError("initial_probs should have 4 elements"))
        size(transition_matrix) == (4, 4) || throw(ArgumentError("transition_matrix should be 4x4"))
        iszero(mean(turn)) || throw(ArgumentError("turn should have zero mean"))
        return new(initial_probs, transition_matrix, forw, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_TN04) = length(hmm.initial_probs)

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_TN04)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_TN04)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_TN04, i::Int)
    if i ∈ (1, 2) # forward-left, forward-right
        dist = hmm.forw
    elseif i == 3 # left
        dist = truncated(hmm.turn, nothing, 0) # (-Inf, 0]
    elseif i == 4 # right
        dist = truncated(hmm.turn, 0, nothing) # [0, Inf)
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end

    return dist
end

function normalize_initial_probs!(hmm::ZebrafishHMM_TN04)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_TN04)
    # forbidden transitions
    hmm.transition_matrix[3,2] = 0 # forbid left -> forward-right
    hmm.transition_matrix[4,1] = 0 # forbid right -> forward-left

    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_TN04)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_TN04, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities. Forward angles are always centered at μ = 0. =#
    hmm.forw = fit_mle(typeof(hmm.forw), obs_seq, state_marginals[1,:] + state_marginals[2,:]; mu = 0.0)

    #= Update left-right turn emission probabilities =#
    turn_marginals = ifelse.(obs_seq .< 0, state_marginals[3,:], state_marginals[4,:])
    hmm.turn = fit_mle(typeof(hmm.turn), abs.(obs_seq), turn_marginals; mu = 0.0)
end


function save_hmm(path::AbstractString, hmm::ZebrafishHMM_TN04)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_TN04")
        write(h5, "initial_probs", hmm.initial_probs)
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "forw", collect(params(hmm.forw)))
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_TN04})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_TN04" || throw(ArgumentError("HMM type missmatch"))
        initial_probs = read(h5, "initial_probs")
        transition_matrix = read(h5, "transition_matrix")
        forw_params = read(h5, "forw")
        turn_params = read(h5, "turn")
        return ZebrafishHMM_TN04(initial_probs, transition_matrix, Normal(forw_params...), Normal(turn_params...))
    end
end
