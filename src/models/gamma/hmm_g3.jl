mutable struct ZebrafishHMM_G3 <: HiddenMarkovModels.AbstractHMM
    #=
    States are stored in this order:
        1. Forward (forward bouts)
        2. Left (left turning bouts)
        3. Right (right turning bouts)
    In contrast to ZebrafishHMM_G4, Forward bouts have no memory of the last turning direction.
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    forw::Normal{Float64}
    turn::Gamma{Float64}

    function ZebrafishHMM_G3(
        initial_probs::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        forw::Normal{<:Real}, turn::Gamma{<:Real}
    )
        length(initial_probs) == 3 || throw(ArgumentError("initial_probs should have 3 elements"))
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        return new(initial_probs, transition_matrix, forw, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_G3) = length(hmm.initial_probs)

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_G3)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_G3)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_G3, i::Int)
    if i == 1 # forward
        dist = hmm.forw
    elseif i == 2 # left
        dist = AffineDistribution(0, -1, hmm.turn)
    elseif i == 3 # right
        dist = hmm.turn
    else
        throw(ArgumentError("State index must be 1, 2, or 3; got $i"))
    end

    return dist
end

function normalize_initial_probs!(hmm::ZebrafishHMM_G3)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_G3)
    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_G3)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_G3, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities. Forward angles are always centered at μ = 0 =#
    hmm.forw = fit_mle(typeof(hmm.forw), obs_seq, state_marginals[1,:]; mu = 0.0)

    #= Update left-right turn emission probabilities. =#
    turn_marginals = ifelse.(obs_seq .< 0, state_marginals[2,:], state_marginals[3,:])
    hmm.turn = fit_mle(typeof(hmm.turn), abs.(obs_seq), turn_marginals)
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_G3)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_G3")
        write(h5, "initial_probs", hmm.initial_probs)
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "forw", collect(params(hmm.forw)))
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_G3})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_G3" || throw(ArgumentError("HMM type missmatch"))
        initial_probs = read(h5, "initial_probs")
        transition_matrix = read(h5, "transition_matrix")
        forw_params = read(h5, "forw")
        turn_params = read(h5, "turn")
        return ZebrafishHMM_G3(initial_probs, transition_matrix, Normal(forw_params...), Gamma(turn_params...))
    end
end
