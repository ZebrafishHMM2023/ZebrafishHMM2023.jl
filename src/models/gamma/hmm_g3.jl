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
    σforw::Float64
    turn::Gamma{Float64}
    min_turn_alpha::Float64

    function ZebrafishHMM_G3(
        initial_probs::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        σforw::Real, turn::Gamma{<:Real},
        min_turn_alpha::Float64 = 0.0
    )
        length(initial_probs) == 3 || throw(ArgumentError("initial_probs should have 3 elements"))
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        turn.α ≥ min_turn_alpha || throw(ArgumentError("turn.α should be greater than min_turn_alpha"))
        return new(initial_probs, transition_matrix, σforw, turn, min_turn_alpha)
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
        dist = Normal(0, hmm.σforw)
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
    hmm.σforw = fit_mle(Normal, obs_seq, state_marginals[1,:]; mu = 0.0).σ

    #= Update left-right turn emission probabilities. =#
    @assert iszero(state_marginals[2, findall(obs_seq .> 0)])
    @assert iszero(state_marginals[3, findall(obs_seq .< 0)])

    turn_obs = [
        -obs_seq[(obs_seq .< 0) .& (state_marginals[2,:] .> 0)];
        +obs_seq[(obs_seq .> 0) .& (state_marginals[3,:] .> 0)]
    ]
    turn_marginals = [
        state_marginals[2, (obs_seq .< 0) .& (state_marginals[2,:] .> 0)];
        state_marginals[3, (obs_seq .> 0) .& (state_marginals[3,:] .> 0)]
    ]
    @assert all(>(0), turn_obs)
    @assert all(>(0), turn_marginals)

    hmm.turn = fit_mle(typeof(hmm.turn), turn_obs, turn_marginals)

    # enforce minimum alpha
    if hmm.turn.α < hmm.min_turn_alpha
        hmm.turn = Gamma(hmm.min_turn_alpha, hmm.turn.θ)
    end

    return hmm
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

function stubborness_factor(hmm::ZebrafishHMM_G3, q::Int) # does not depend on q
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_3_state(T)
end
