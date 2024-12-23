mutable struct ZebrafishHMM_TN3 <: HiddenMarkovModels.AbstractHMM
    #=
    Truncated normal (TN) turning angle emissions, with fitted mean.
    States are stored in this order:
        1. Forward (forward bouts)
        2. Left (left turning bouts)
        3. Right (right turning bouts)
    =#
    const initial_probs::Vector{Float64}
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    forw::Normal{Float64}
    turn::Normal{Float64}
    regularization::Float64

    function ZebrafishHMM_TN3(
        initial_probs::AbstractVector{<:Real},
        transition_matrix::AbstractMatrix{<:Real},
        forw::Normal{<:Real},
        turn::Normal{<:Real},
        regularization::Float64 = 1e-10
    )
        length(initial_probs) == 3 || throw(ArgumentError("initial_probs should have 3 elements"))
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        regularization > 0 || throw(ArgumentError("regularization must be positive; got $regularization"))
        return new(initial_probs, transition_matrix, forw, turn, regularization)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_TN3) = length(hmm.initial_probs)

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_TN3)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_TN3)
    return hmm.initial_probs
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_TN3, i::Int)
    if i == 1 # forward
        dist = hmm.forw
    elseif i == 2 # left
        dist = truncated(Normal(-mean(hmm.turn), std(hmm.turn)), nothing, 0)
    elseif i == 3 # right
        dist = truncated(Normal(+mean(hmm.turn), std(hmm.turn)), 0, nothing)
    else
        throw(ArgumentError("State index must be 1, 2, or 3; got $i"))
    end

    return dist
end

function normalize_initial_probs!(hmm::ZebrafishHMM_TN3)
    hmm.initial_probs .= hmm.initial_probs ./ sum(hmm.initial_probs)
    return hmm.initial_probs
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_TN3)
    # TODO: impose left / right symmetry ?

    # normalize transition matrix
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_TN3)
    normalize_initial_probs!(hmm)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_TN3, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.initial_probs .= init_count
    normalize_initial_probs!(hmm)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities =#
    hmm.forw = fit_mle(typeof(hmm.forw), obs_seq, state_marginals[1,:]; mu = 0.0)

    #= Update left-right turn emission probabilities =#
    @assert iszero(state_marginals[2, findall(obs_seq .> 0)])
    @assert iszero(state_marginals[3, findall(obs_seq .< 0)])
    @assert isempty(findall(>(0), state_marginals[2,:]) ∩ findall(>(0), state_marginals[3,:]))

    turn_obs = [-obs_seq[state_marginals[2,:] .> 0]; obs_seq[state_marginals[3,:] .> 0]]
    turn_marginals = [state_marginals[2, state_marginals[2,:] .> 0]; state_marginals[3, state_marginals[3,:] .> 0]]
    @assert all(>(0), turn_obs)
    @assert all(>(0), turn_marginals)

    S1 = mean(turn_marginals .* turn_obs) / mean(turn_marginals)
    ν = mean(turn_marginals .* abs2.(turn_obs .- S1)) / mean(turn_marginals)
    S2 = ν + S1^2

    μ, σ = half_normal_fit(S1, S2; γ = hmm.regularization)
    hmm.turn = Normal(μ, σ)

    # turn_marginals = ifelse.(obs_seq .< 0, state_marginals[2,:], state_marginals[3,:])
    # S1 = mean(turn_marginals .* abs.(obs_seq)) / mean(turn_marginals)
    # S2 = mean(turn_marginals .* abs2.(obs_seq)) / mean(turn_marginals)
    # μ, σ = half_normal_fit(S1, S2)
    # hmm.turn = Normal(μ, σ)
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_TN3)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_TN3")
        write(h5, "initial_probs", hmm.initial_probs)
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "forw", collect(params(hmm.forw)))
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_TN3})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_TN3" || throw(ArgumentError("HMM type missmatch"))
        initial_probs = read(h5, "initial_probs")
        transition_matrix = read(h5, "transition_matrix")
        forw_params = read(h5, "forw")
        turn_params = read(h5, "turn")
        return ZebrafishHMM_TN3(initial_probs, transition_matrix, Normal(forw_params...), Normal(turn_params...))
    end
end

function stubborness_factor(hmm::ZebrafishHMM_TN3, q::Int) # does not depend on q
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_3_state(T)
end
