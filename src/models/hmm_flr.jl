mutable struct ZebrafishHMM_FLR <: HiddenMarkovModels.AbstractHMM
    #=
    States are stored in this order:
        1. Forward (forward bouts)
        3. Left (left turning bouts)
        4. Right (right turning bouts)
    In contrast to ZebrafishHMM, Forward bouts have no memory of the last turning direction.
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    forw::Normal{Float64}
    turn::Gamma{Float64}
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_FLR) = 3

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_FLR)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_FLR)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_FLR, i::Int)
    if i == 1 # forward
        dist = hmm.forw
    elseif i == 2 # left
        dist = AffineDistribution(0, -1, hmm.turn)
    elseif i == 3 # right
        dist = hmm.turn
    else
        throw(ArgumentError("State index must be 1, 2, or 3; got $i"))
    end

    return DistributionMissingWrapper(dist)
end

function normalize_initial_probs!(hmm::ZebrafishHMM_FLR)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_FLR)
    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_FLR)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_FLR, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities =#
    forw_obs, forw_marginals = filter_obs(!ismissing, obs_seq, state_marginals[1,:])
    forw = fit(typeof(hmm.forw), forw_obs, forw_marginals)

    # forward angles are always centered at 0, so we only use the fitted σ, discarding μ
    hmm.forw = Normal(0, forw.σ)

    #= Update left-right turn emission probabilities =#
    # discard negative and missing entries, since they could not have been emitted by turning states
    turn_obs, turn_marginals = filter_obs(
        x -> !ismissing(x) && x > 0,
        [-obs_seq; obs_seq],
        [state_marginals[2,:]; state_marginals[3,:]]
    )
    hmm.turn = fit(typeof(hmm.turn), turn_obs, turn_marginals)
end
