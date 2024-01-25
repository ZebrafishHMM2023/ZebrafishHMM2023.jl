mutable struct ZebrafishHMM_G4_Sym <: HiddenMarkovModels.AbstractHMM
    #=
    Imposes Left/Right symmetry
    =#
    pinit_turn::Float64
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    σforw::Float64
    turn::Gamma{Float64}
    min_turn_alpha::Float64

    function ZebrafishHMM_G4_Sym(
        pinit_turn::Real,
        transition_matrix::AbstractMatrix{<:Real},
        σforw::Float64, turn::Gamma{<:Real},
        min_turn_alpha::Float64 = 0.0
    )
        size(transition_matrix) == (4, 4) || throw(ArgumentError("transition_matrix should be 4x4"))
        σforw ≥ 0 || throw(ArgumentError("σforw should be non-negative"))
        turn.α ≥ min_turn_alpha || throw(ArgumentError("turn.α should be greater than min_turn_alpha"))
        return new(pinit_turn, transition_matrix, σforw, turn, min_turn_alpha)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_G4_Sym) = 4

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_G4_Sym)
    return hmm.transition_matrix
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_G4_Sym)
    p0 = hmm.pinit_turn
    return @SVector [(1 - p0) / 2, (1 - p0) / 2, p0 / 2, p0 / 2]
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_G4_Sym, i::Int)
    if i ∈ (1, 2) # forward-left, forward-right
        dist = Normal(0, hmm.σforw)
    elseif i == 3 # left
        dist = SignedGamma(hmm.turn; positive = false)
    elseif i == 4 # right
        dist = SignedGamma(hmm.turn; positive = true)
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end

    return dist
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_G4_Sym)
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

#=
There is a degeneracy between the first two hidden states (which should correspond to FL, FR).
This function fixes that degeneracy by imposing that FL -> L be more likely than FL -> R.
If this condition is not satisfied, states are permuted appropriately.
=#
function FL_FR_canon!(hmm::ZebrafishHMM_G4_Sym)
    if hmm.transition_matrix[1,3] < hmm.transition_matrix[1,4]
        hmm.transition_matrix .= hmm.transition_matrix[[2,1,3,4], [2,1,3,4]]
    end
    return hmm
end

function normalize_all!(hmm::ZebrafishHMM_G4_Sym)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(
    hmm::ZebrafishHMM_G4_Sym,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector,
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == 4
    @assert size(trans_count) == (4, 4)

    #= Update initial state probabilities =#
    hmm.pinit_turn = (init_count[3] + init_count[4]) / sum(init_count)

    #= Update transition matrix =#
    hmm.transition_matrix .= trans_count
    normalize_transition_matrix!(hmm)

    #= Update forward emission probabilities. =#
    hmm.σforw = fit_mle(Normal, obs_seq, state_marginals[1,:] + state_marginals[2,:]; mu = 0.0).σ

    #= Update left-right turn emission probabilities. =#
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

    hmm.turn = fit_mle(typeof(hmm.turn), turn_obs, turn_marginals)

    # enforce minimum alpha
    if hmm.turn.α < hmm.min_turn_alpha
        hmm.turn = Gamma(hmm.min_turn_alpha, hmm.turn.θ)
    end

    return hmm
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_G4_Sym)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_G4_Sym")
        write(h5, "initial_probs", [hmm.pinit_turn])
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "forw", [hmm.σforw])
        write(h5, "turn", collect(params(hmm.turn)))
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_G4_Sym})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_G4_Sym" || throw(ArgumentError("HMM type missmatch"))
        pinit_turn = only(read(h5, "initial_probs"))
        transition_matrix = read(h5, "transition_matrix")
        σforw = only(read(h5, "forw"))
        turn_params = read(h5, "turn")
        return ZebrafishHMM_G4_Sym(pinit_turn, transition_matrix, σforw, Gamma(turn_params...))
    end
end

function stubborness_factor(hmm::ZebrafishHMM_G4_Sym, q::Int)
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_4_state(T, q)
end
