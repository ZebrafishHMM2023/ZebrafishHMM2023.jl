#=
Two-state model, L and R.
=#
mutable struct ZebrafishHMM_G2_Sym <: HiddenMarkovModels.AbstractHMM
    pinit_L::Float64
    pturn::Float64
    turn::Gamma{Float64}
    function ZebrafishHMM_G2_Sym(pinit_L::Real, pturn::Real, turn::Gamma)
        return new(pinit_L, pturn, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_G2_Sym) = 2

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_G2_Sym)
    pturn = hmm.pturn
    return @SMatrix [
        # Left          Right
        (1 - pturn)     pturn       # Left
        pturn           (1 - pturn) # Right
        # ↔ DESTINATION STATE       # ↕ INITIAL STATE
    ]
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_G2_Sym)
    return @SVector [hmm.pinit_L, 1 - hmm.pinit_L]
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_G2_Sym, i::Int)
    if i == 1 # left
        return AffineDistribution(0, -1, hmm.turn)
    elseif i == 2 # right
        return AffineDistribution(0, +1, hmm.turn)
    else
        throw(ArgumentError("State index must be 1, or 2; got $i"))
    end
end

function StatsAPI.fit!(
    hmm::ZebrafishHMM_G2_Sym,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector,
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == 2
    @assert size(trans_count) == (2, 2)
    hmm.pinit_L = init_count[1] / sum(init_count)
    hmm.pturn = middle(trans_count[1,2], trans_count[2,1])

    #= Update left-right turn emission probabilities. =#
    @assert size(state_marginals) == (2, length(obs_seq))
    @assert iszero(state_marginals[1, findall(obs_seq .> 0)])
    @assert iszero(state_marginals[2, findall(obs_seq .< 0)])

    turn_obs = [
        -obs_seq[(obs_seq .< 0) .& (state_marginals[1,:] .> 0)];
        +obs_seq[(obs_seq .> 0) .& (state_marginals[2,:] .> 0)]
    ]
    turn_marginals = [
        state_marginals[1, (obs_seq .< 0) .& (state_marginals[1,:] .> 0)];
        state_marginals[2, (obs_seq .> 0) .& (state_marginals[2,:] .> 0)]
    ]
    @assert all(>(0), turn_obs)
    @assert all(>(0), turn_marginals)

    hmm.turn = fit_mle(typeof(hmm.turn), turn_obs, turn_marginals)

    return nothing
end
