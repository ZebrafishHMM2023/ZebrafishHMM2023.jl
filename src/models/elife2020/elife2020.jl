#= This model should be the Hidden Markov model equivalent of the Markov chain model of
Karpenko et al Elife 2020

The hidden states represent two independent Markov chains, one for Forward/Turn states, and
one for Left/Right states. Left/right orientations are symmetric. Turning angle emissions
are truncated normal with μ=0.
=#
mutable struct ZebrafishHMM_Elife2020 <: HiddenMarkovModels.AbstractHMM
    pinit_turn::Float64

    pturn::Float64
    pflip::Float64
    σturn::Float64
    σforw::Float64

    function ZebrafishHMM_Elife2020(
        ; pinit_turn::Real, pturn::Real, pflip::Real, σturn::Real, σforw::Real
    )
        0 ≤ pinit_turn ≤ 1 || throw(ArgumentError("pinit_turn must be between 0 and 1; got $pinit_turn"))
        0 ≤ pturn ≤ 1 || throw(ArgumentError("pturn must be between 0 and 1; got $pturn"))
        0 ≤ pflip ≤ 1 || throw(ArgumentError("pflip must be between 0 and 1; got $pflip"))
        0 ≤ σturn || throw(ArgumentError("σturn must be non-negative; got $σturn"))
        0 ≤ σforw || throw(ArgumentError("σforw must be non-negative; got $σforw"))
        return new(pinit_turn, pturn, pflip, σturn, σforw)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_Elife2020) = 4

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_Elife2020)
    #=
    States are stored in this order:
        1. Forward-Left
        2. Forward-Right
        3. Turn-Left
        4. Turn-Right
    =#
    pturn, pflip = hmm.pturn, hmm.pflip

    # t[i → j] = Prob. of going from 'i' to 'j'.
    # Rows give the initial state, column the destination state.
    t = @SMatrix [
        # Forward/Left      Forward/Right       Turn/Left       Turn/Right
        (1-pturn)*(1-pflip) (1-pturn)*pflip     pturn*(1-pflip) pturn*pflip     # Forward/Left
        (1-pturn)*pflip     (1-pturn)*(1-pflip) pturn*pflip     pturn*(1-pflip) # Forward/Right
        (1-pturn)*(1-pflip) (1-pturn)*pflip     pturn*(1-pflip) pturn*pflip     # Turn/Left
        (1-pturn)*pflip     (1-pturn)*(1-pflip) pturn*pflip     pturn*(1-pflip) # Turn/Right
        # ↔ DESTINATION STATE                                                   # ↕ INITIAL STATE
    ]

    return t
end

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_Elife2020)
    pinit_turn = hmm.pinit_turn
    p0 = @SVector [(1 - pinit_turn) / 2, (1 - pinit_turn) / 2, pinit_turn / 2, pinit_turn / 2]
    return p0
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_Elife2020, i::Int)
    if i == 1 || i == 2 # forward / left, forward / right
        dist = Normal(0, hmm.σforw)
    elseif i == 3 # turn / left
        dist = truncated(Normal(0, hmm.σturn), nothing, 0)
    elseif i == 4 # turn / right
        dist = truncated(Normal(0, hmm.σturn), 0, nothing)
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end
    return dist
end

function StatsAPI.fit!(hmm::ZebrafishHMM_Elife2020, init_count, trans_count, obs_seq, state_marginals)
    #= Update initial state probabilities =#
    hmm.pinit_turn = (init_count[3] + init_count[4]) / sum(init_count)

    #= Update transition matrix =#
    trans_tot = sum(trans_count)
    turn_count = sum(@view trans_count[:,3:4])
    flip_L_to_R_count = sum(@view trans_count[[1,3], [2,4]])
    flip_R_to_L_count = sum(@view trans_count[[2,4], [1,3]])
    hmm.pturn = turn_count / trans_tot
    hmm.pflip = (flip_L_to_R_count + flip_R_to_L_count) / trans_tot

    #= Update forward emission probabilities. Forward angles are always centered at μ = 0. =#
    hmm.σforw = std(fit_mle(Normal, obs_seq, state_marginals[1,:] + state_marginals[2,:]; mu = 0.0))

    #= Update emission probabilities =#
    @assert iszero(state_marginals[3, findall(obs_seq .> 0)])
    @assert iszero(state_marginals[4, findall(obs_seq .< 0)])
    @assert isempty(findall(>(0), state_marginals[3,:]) ∩ findall(>(0), state_marginals[4,:]))

    turn_obs = [-obs_seq[state_marginals[3,:] .> 0]; obs_seq[state_marginals[4,:] .> 0]]
    turn_marginals = [state_marginals[3, state_marginals[3,:] .> 0]; state_marginals[4, state_marginals[4,:] .> 0]]
    @assert all(>(0), turn_obs)
    @assert all(>(0), turn_marginals)

    #= Update forward emission probabilities. Forward angles are always centered at μ = 0. =#
    forw = fit_mle(Normal{Float64}, obs_seq, state_marginals[1,:] + state_marginals[2,:]; mu = 0.0)
    hmm.σforw = std(forw)

    #= Update left-right turn emission probabilities =#
    turn_marginals = ifelse.(obs_seq .< 0, state_marginals[3,:], state_marginals[4,:])
    turn = fit_mle(Normal, abs.(obs_seq), turn_marginals; mu = 0.0)
    hmm.σturn = std(turn)

    return hmm
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_Elife2020)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_Elife2020")
        write(h5, "params", [hmm.pinit_turn, hmm.pflip, hmm.pturn, hmm.σforw, hmm.σturn])
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_Elife2020})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_Elife2020" || throw(ArgumentError("HMM type missmatch"))
        params = read(h5, "params")
        return ZebrafishHMM_Elife2020(;
            pinit_turn = params[1],
            pflip = params[2],
            pturn = params[3],
            σforw = params[4],
            σturn = params[5]
        )
    end
end

function stubborness_factor(hmm::ZebrafishHMM_Elife2020, q::Int)
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_4_state(T, q)
end
