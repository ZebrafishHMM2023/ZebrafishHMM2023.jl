mutable struct ZebrafishHMM_G3_Sym <: HiddenMarkovModels.AbstractHMM
    pinit_turn::Float64
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    σforw::Float64
    turn::Gamma{Float64}

    function ZebrafishHMM_G3_Sym(
        pinit_turn::Real,
        transition_matrix::AbstractMatrix{<:Real},
        σforw::Real, turn::Gamma{<:Real}
    )
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        return new(pinit_turn, transition_matrix, σforw, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_G3_Sym) = size(hmm.transition_matrix, 1)

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_G3_Sym)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_G3_Sym)
    p0 = hmm.pinit_turn
    return @SVector [1 - p0, p0 / 2, p0 / 2]
end


function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_G3_Sym, i::Int)
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

function normalize_transition_matrix!(hmm::ZebrafishHMM_G3_Sym)
    # normalize
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    # left/right symmetry
    hmm.transition_matrix[2,2] = hmm.transition_matrix[3,3] = middle(hmm.transition_matrix[2,2], hmm.transition_matrix[2,2]) # L -> L, R -> R
    hmm.transition_matrix[2,3] = hmm.transition_matrix[3,2] = middle(hmm.transition_matrix[2,3], hmm.transition_matrix[3,2]) # L -> R, R -> L
    hmm.transition_matrix[1,2] = hmm.transition_matrix[1,3] = middle(hmm.transition_matrix[1,2], hmm.transition_matrix[1,3]) # F -> L, F -> R
    hmm.transition_matrix[2,1] = hmm.transition_matrix[3,1] = middle(hmm.transition_matrix[2,1], hmm.transition_matrix[3,1]) # L -> F, R -> F

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_G3_Sym)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(hmm::ZebrafishHMM_G3_Sym, init_count, trans_count, obs_seq, state_marginals)
    @assert length(init_count) == 3
    @assert size(trans_count) == (3, 3)

    #= Update initial state probabilities =#
    hmm.pinit_turn = (init_count[2] + init_count[3]) / sum(init_count)

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

    return hmm
end

function stubborness_factor(hmm::ZebrafishHMM_G3_Sym, q::Int) # does not depend on q
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_3_state(T)
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_G3_Sym)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_G3_Sym")
        write(h5, "pinit_turn", [hmm.pinit_turn])
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "σforw", [hmm.σforw])
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_G3_Sym})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_G3_Sym" || throw(ArgumentError("HMM type missmatch"))
        pinit_turn = only(read(h5, "pinit_turn"))
        transition_matrix = read(h5, "transition_matrix")
        σforw = only(read(h5, "σforw"))
        turn_params = read(h5, "turn")
        return ZebrafishHMM_G3_Sym(pinit_turn, transition_matrix, σforw, Gamma(turn_params...))
    end
end
