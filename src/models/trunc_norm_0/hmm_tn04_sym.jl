mutable struct ZebrafishHMM_TN04_Sym <: HiddenMarkovModels.AbstractHMM
    pinit_turn::Float64
    const transition_matrix::Matrix{Float64}
    σforw::Float64
    σturn::Float64

    function ZebrafishHMM_TN04_Sym(
        pinit_turn::Real,
        transition_matrix::AbstractMatrix{<:Real},
        σforw::Real, σturn::Real
    )
        size(transition_matrix) == (4, 4) || throw(ArgumentError("transition_matrix should be 4x4"))
        return new(pinit_turn, transition_matrix, σforw, σturn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_TN04_Sym) = 4

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_TN04_Sym)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_TN04_Sym)
    p0 = hmm.pinit_turn
    return @SVector [(1 - p0) / 2, (1 - p0) / 2, p0 / 2, p0 / 2]
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_TN04_Sym, i::Int)
    if i ∈ (1, 2) # forward-left, forward-right
        dist = Normal(0, hmm.σforw)
    elseif i == 3 # left
        dist = truncated(Normal(0, hmm.σturn), nothing, 0) # (-Inf, 0]
    elseif i == 4 # right
        dist = truncated(Normal(0, hmm.σturn), 0, nothing) # [0, Inf)
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end

    return dist
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_TN04_Sym)
    # normalize
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    # left/right symmetry
    hmm.transition_matrix[1,1] = hmm.transition_matrix[2,2] = middle(hmm.transition_matrix[1,1], hmm.transition_matrix[2,2]) # FL -> FL, FR -> FR
    hmm.transition_matrix[1,2] = hmm.transition_matrix[2,1] = middle(hmm.transition_matrix[1,2], hmm.transition_matrix[2,1]) # FL -> FR, FR -> FL
    hmm.transition_matrix[3,3] = hmm.transition_matrix[4,4] = middle(hmm.transition_matrix[3,3], hmm.transition_matrix[4,4]) # TL -> TL, TR -> TR
    hmm.transition_matrix[3,4] = hmm.transition_matrix[4,3] = middle(hmm.transition_matrix[3,4], hmm.transition_matrix[4,3]) # TL -> TR, TR -> TL

    hmm.transition_matrix[1,3] = hmm.transition_matrix[2,4] = middle(hmm.transition_matrix[1,3], hmm.transition_matrix[2,4]) # FL -> TL, FR -> TR
    hmm.transition_matrix[1,4] = hmm.transition_matrix[2,3] = middle(hmm.transition_matrix[1,4], hmm.transition_matrix[2,3]) # FL -> TR, FR -> TL
    hmm.transition_matrix[3,1] = hmm.transition_matrix[4,2] = middle(hmm.transition_matrix[3,1], hmm.transition_matrix[4,2]) # TL -> FL, TR -> FR
    hmm.transition_matrix[3,2] = hmm.transition_matrix[4,1] = middle(hmm.transition_matrix[3,2], hmm.transition_matrix[4,1]) # TL -> FR, TR -> FL

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_TN04_Sym)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_TN04_Sym, init_count, trans_count, obs_seq, state_marginals)
    @assert length(init_count) == 4
    @assert size(trans_count) == (4, 4)

    #= Update initial state probabilities =#
    #= Update initial state probabilities =#
    hmm.pinit_turn = (init_count[3] + init_count[4]) / sum(init_count)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities. Forward angles are always centered at μ = 0. =#
    hmm.σforw = fit_mle(Normal{Float64}, obs_seq, state_marginals[1,:] + state_marginals[2,:]; mu = 0.0).σ

    #= Update left-right turn emission probabilities =#
    @assert iszero(state_marginals[3, findall(obs_seq .> 0)])
    @assert iszero(state_marginals[4, findall(obs_seq .< 0)])

    turn_obs = [
        -obs_seq[(obs_seq .< 0) .& (state_marginals[3,:] .> 0)];
        +obs_seq[(obs_seq .> 0) .& (state_marginals[4,:] .> 0)]
    ]
    turn_marginals = [
        state_marginals[3, (obs_seq .< 0) .& (state_marginals[3,:] .> 0)];
        state_marginals[4, (obs_seq .> 0) .& (state_marginals[4,:] .> 0)]
    ]
    @assert all(>(0), turn_obs)
    @assert all(>(0), turn_marginals)

    hmm.σturn = fit_mle(Normal{Float64}, obs_seq, turn_marginals; mu = 0.0).σ

    return hmm
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_TN04_Sym)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_TN04_Sym")
        write(h5, "initial_probs", [hmm.pinit_turn])
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "σforw", [hmm.σforw])
        write(h5, "σturn", [hmm.σturn])
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_TN04_Sym})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_TN04_Sym" || throw(ArgumentError("HMM type missmatch"))
        pinit_turn = only(read(h5, "initial_probs"))
        transition_matrix = read(h5, "transition_matrix")
        σforw = only(read(h5, "σforw"))
        σturn = only(read(h5, "σturn"))
        return ZebrafishHMM_TN04_Sym(pinit_turn, transition_matrix, σforw, σturn)
    end
end

function stubborness_factor(hmm::ZebrafishHMM_TN04_Sym, q::Int)
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_4_state(T, q)
end
