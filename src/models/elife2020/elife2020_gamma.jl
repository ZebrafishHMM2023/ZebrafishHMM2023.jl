#= Like the Elife2020 model, but with Gamma turn emissions. =#
mutable struct ZebrafishHMM_Elife2020_Gamma <: HiddenMarkovModels.AbstractHMM
    pinit_turn::Float64

    pturn::Float64
    pflip::Float64

    σforw::Float64
    turn::Gamma{Float64}

    function ZebrafishHMM_Elife2020_Gamma(
        ; pinit_turn::Real, pturn::Real, pflip::Real, σforw::Real, turn::Gamma
    )
        0 ≤ pinit_turn ≤ 1 || throw(ArgumentError("pinit_turn must be between 0 and 1; got $pinit_turn"))
        0 ≤ pturn ≤ 1 || throw(ArgumentError("pturn must be between 0 and 1; got $pturn"))
        0 ≤ pflip ≤ 1 || throw(ArgumentError("pflip must be between 0 and 1; got $pflip"))
        0 ≤ σforw || throw(ArgumentError("σforw must be non-negative; got $σforw"))
        return new(pinit_turn, pturn, pflip, σforw, turn)
    end
end

# number of hidden states
Base.length(hmm::ZebrafishHMM_Elife2020_Gamma) = 4

function HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_Elife2020_Gamma)
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

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_Elife2020_Gamma)
    pinit_turn = hmm.pinit_turn
    p0 = @SVector [(1 - pinit_turn) / 2, (1 - pinit_turn) / 2, pinit_turn / 2, pinit_turn / 2]
    return p0
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_Elife2020_Gamma, i::Int)
    if i == 1 || i == 2 # forward / left, forward / right
        dist = Normal(0, hmm.σforw)
    elseif i == 3 # turn left
        dist = SignedGamma(hmm.turn; positive = false)
    elseif i == 4 # turn right
        dist = SignedGamma(hmm.turn; positive = true)
    else
        throw(ArgumentError("State index must be 1, 2, 3, or 4; got $i"))
    end
    return dist
end

function StatsAPI.fit!(hmm::ZebrafishHMM_Elife2020_Gamma, init_count, trans_count, obs_seq, state_marginals)
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

    #= Update turn emission probabilities =#
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

    return hmm
end

function save_hmm(path::AbstractString, hmm::ZebrafishHMM_Elife2020_Gamma)
    h5open(path, "w") do h5
        write(h5, "type", "ZebrafishHMM_Elife2020_Gamma")
        write(h5, "params", [hmm.pinit_turn, hmm.pflip, hmm.pturn, hmm.σforw, hmm.turn.α, hmm.turn.θ])
    end
end

function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_Elife2020_Gamma})
    h5open(path, "r") do h5
        read(h5, "type") == "ZebrafishHMM_Elife2020_Gamma" || throw(ArgumentError("HMM type missmatch"))
        params = read(h5, "params")
        return ZebrafishHMM_Elife2020_Gamma(;
            pinit_turn = params[1],
            pflip = params[2],
            pturn = params[3],
            σforw = params[4],
            turn = Gamma(params[5], params[6])
        )
    end
end

function stubborness_factor(hmm::ZebrafishHMM_Elife2020_Gamma, q::Int)
    T = HiddenMarkovModels.transition_matrix(hmm)
    return _stubborness_factor_4_state(T, q)
end
