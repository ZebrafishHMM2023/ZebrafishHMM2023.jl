mutable struct ZebrafishHMM <: HiddenMarkovModels.AbstractHMM
    #=
    states are stored in this order:
        1. Forward-Left (forward bouts, but last turn was left)
        2. Forward-Right (forward bouts, but last turn was right)
        3. Left (left turning bouts)
        4. Right (right turning bouts)
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    forw::Normal{Float64}
    turn::Gamma{Float64}
end

# number of hidden states
Base.length(hmm::ZebrafishHMM) = 4

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM, i::Int)
    if i ∈ (1, 2) # forward-left, forward-right
        dist = hmm.forw
    elseif i == 3 # left
        dist = AffineDistribution(0, -1, hmm.turn)
    elseif i == 4 # right
        dist = hmm.turn
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end

    return DistributionMissingWrapper(dist)
end

function normalize_initial_probs!(hmm::ZebrafishHMM)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM)
    # forbidden transitions
    hmm.transition_matrix[3,2] = 0 # left -> forward-right is forbidden
    hmm.transition_matrix[4,1] = 0 # right -> forward-left is forbidden
    hmm.transition_matrix[1,2] = 0 # forward-left -> forward-right is forbidden
    hmm.transition_matrix[2,1] = 0 # forward-right -> forward-left is forbidden

    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities =#
    forw_obs, forw_marginals = filter_obs(
        !ismissing, obs_seq, state_marginals[1,:] + state_marginals[2,:]
    )
    forw = fit(typeof(hmm.forw), forw_obs, forw_marginals)

    # forward angles are always centered at 0, so we only use the fitted σ, discarding μ
    hmm.forw = Normal(0, forw.σ)

    #= Update left-right turn emission probabilities =#
    # discard negative and missing entries, since they could not have been emitted by turning states
    turn_obs, turn_marginals = filter_obs(
        x -> !ismissing(x) && x > 0,
        [-obs_seq; obs_seq],
        [state_marginals[3,:]; state_marginals[4,:]]
    )
    hmm.turn = fit(typeof(hmm.turn), turn_obs, turn_marginals)
end

function filter_obs(cond, obs_seq::AbstractVector, state_marginals::AbstractVector)
    _idx = findall(cond, obs_seq)
    return map(identity, obs_seq[_idx]), map(identity, state_marginals[_idx])
end
