mutable struct ZebrafishHMM_G3_Sym_Full <: HiddenMarkovModels.AbstractHMM
    pinit_turn::Float64
    const transition_matrix::Matrix{Float64} # T[i,j] = probability of transitions i -> j
    σforw::Float64
    turn::Gamma{Float64}
    forward_displacement::Gamma{Float64}
    turn_displacement::Gamma{Float64}
    forward_interboutinterval::Gamma{Float64}
    turn_interboutinterval::Gamma{Float64}
    min_alpha::Float64

    only_train_spacetime::Bool # freezes parameters ineferrable from bout angles:s transition_matrix, pinit_turn, ...

    function ZebrafishHMM_G3_Sym_Full(
        pinit_turn::Real,
        transition_matrix::AbstractMatrix{<:Real},
        σforw::Real, turn::Gamma{<:Real},
        forward_displacement::Gamma{<:Real}, turn_displacement::Gamma{<:Real},
        forward_interboutinterval::Gamma{<:Real}, turn_interboutinterval::Gamma{<:Real},
        min_alpha::Float64 = 0.0; only_train_spacetime::Bool = false
    )
        size(transition_matrix) == (3, 3) || throw(ArgumentError("transition_matrix should be 3x3"))
        turn.α ≥ min_alpha || throw(ArgumentError("turn.α should be greater than min_turn_alpha"))
        forward_displacement.α ≥ min_alpha || throw(ArgumentError("forward_displacement.α should be greater than min_turn_alpha"))
        turn_displacement.α ≥ min_alpha || throw(ArgumentError("turn_displacement.α should be greater than min_turn_alpha"))
        forward_interboutinterval.α ≥ min_alpha || throw(ArgumentError("forward_interboutinterval.α should be greater than min_turn_alpha"))
        turn_interboutinterval.α ≥ min_alpha || throw(ArgumentError("turn_interboutinterval.α should be greater than min_turn_alpha"))
        return new(
            pinit_turn, transition_matrix, σforw, turn,
            forward_displacement, turn_displacement,
            forward_interboutinterval, turn_interboutinterval, min_alpha, only_train_spacetime
        )
    end
end

struct ZebrafishHMM_G3_Sym_Full_Obs
    θ::Float64
    d::Float64
    t::Float64
end

struct ZebrafishHMM_G3_Sym_Full_Emit_Distribution
    σforw::Float64
    turn::Gamma{Float64}
    forward_displacement::Gamma{Float64}
    turn_displacement::Gamma{Float64}
    forward_interboutinterval::Gamma{Float64}
    turn_interboutinterval::Gamma{Float64}
    state_index::Int
end

function ZebrafishHMM_G3_Sym_Full(
    ; pinit_turn::Real,
    transition_matrix::AbstractMatrix{<:Real},
    σforw::Real, turn::Gamma{<:Real},
    forward_displacement::Gamma{<:Real}, turn_displacement::Gamma{<:Real},
    forward_interboutinterval::Gamma{<:Real}, turn_interboutinterval::Gamma{<:Real},
    min_alpha::Real = 0.0, only_train_spacetime::Bool = false
)
    return ZebrafishHMM_G3_Sym_Full(
        pinit_turn, transition_matrix, σforw, turn,
        forward_displacement, turn_displacement,
        forward_interboutinterval, turn_interboutinterval, min_alpha;
        only_train_spacetime
    )
end

function HiddenMarkovModels.obs_distribution(hmm::ZebrafishHMM_G3_Sym_Full, state_index::Int)
    return ZebrafishHMM_G3_Sym_Full_Emit_Distribution(
        hmm.σforw, hmm.turn,
        hmm.forward_displacement, hmm.turn_displacement,
        hmm.forward_interboutinterval, hmm.turn_interboutinterval,
        state_index
    )
end

function Base.rand(rng::AbstractRNG, dist::ZebrafishHMM_G3_Sym_Full_Emit_Distribution)
    if dist.state_index == 1 # forward
        θ = rand(rng, Normal(0, dist.σforw))
        d = rand(rng, dist.forward_displacement)
        t = rand(rng, dist.forward_interboutinterval)
    else # turn
        if dist.state_index == 2 # left
            θ = -rand(rng, dist.turn)
        elseif dist.state_index == 3 # right
            θ = +rand(rng, dist.turn)
        end
        d = rand(rng, dist.turn_displacement)
        t = rand(rng, dist.turn_interboutinterval)
    end
    return ZebrafishHMM_G3_Sym_Full_Obs(θ, d, t)
end

function DensityInterface.logdensityof(d::ZebrafishHMM_G3_Sym_Full_Emit_Distribution, obs::ZebrafishHMM_G3_Sym_Full_Obs)
    if d.state_index == 1
        lpdf_θ = logdensityof(Normal(0, d.σforw), obs.θ)
        lpdf_d = logdensityof(d.forward_displacement, obs.d)
        lpdf_t = logdensityof(d.forward_interboutinterval, obs.t)
    else
        if d.state_index == 2
            lpdf_θ = logdensityof(d.turn, -obs.θ) # left
        elseif d.state_index == 3
            lpdf_θ = logdensityof(d.turn, +obs.θ) # right
        end
        lpdf_d = logdensityof(d.turn_displacement, obs.d)
        lpdf_t = logdensityof(d.turn_interboutinterval, obs.t)
    end
    return lpdf_θ + lpdf_d + lpdf_t
end

Base.length(hmm::ZebrafishHMM_G3_Sym_Full) = size(hmm.transition_matrix, 1) # number of hidden states
HiddenMarkovModels.transition_matrix(hmm::ZebrafishHMM_G3_Sym_Full) = hmm.transition_matrix

function HiddenMarkovModels.initial_distribution(hmm::ZebrafishHMM_G3_Sym_Full)
    p0 = hmm.pinit_turn
    return @SVector [1 - p0, p0 / 2, p0 / 2]
end

function normalize_transition_matrix!(hmm::ZebrafishHMM_G3_Sym_Full)
    # normalize
    hmm.transition_matrix .= hmm.transition_matrix ./ sum(hmm.transition_matrix, dims=2)

    # left/right symmetry
    hmm.transition_matrix[2,2] = hmm.transition_matrix[3,3] = middle(hmm.transition_matrix[2,2], hmm.transition_matrix[3,3]) # L -> L, R -> R
    hmm.transition_matrix[2,3] = hmm.transition_matrix[3,2] = middle(hmm.transition_matrix[2,3], hmm.transition_matrix[3,2]) # L -> R, R -> L
    hmm.transition_matrix[1,2] = hmm.transition_matrix[1,3] = middle(hmm.transition_matrix[1,2], hmm.transition_matrix[1,3]) # F -> L, F -> R
    hmm.transition_matrix[2,1] = hmm.transition_matrix[3,1] = middle(hmm.transition_matrix[2,1], hmm.transition_matrix[3,1]) # L -> F, R -> F

    return hmm.transition_matrix
end

function normalize_all!(hmm::ZebrafishHMM_G3_Sym_Full)
    normalize_transition_matrix!(hmm)
    return hmm
end

function StatsAPI.fit!(
    hmm::ZebrafishHMM_G3_Sym_Full, init_count, trans_count,
    obs_seq::AbstractVector{ZebrafishHMM_G3_Sym_Full_Obs}, state_marginals
)
    @assert length(init_count) == 3
    @assert size(trans_count) == (3, 3)

    θs = [obs.θ for obs in obs_seq]
    ds = [obs.d for obs in obs_seq]
    ts = [obs.t for obs in obs_seq]

    #= If hmm.only_train_spacetime is true, we only train the space-time components
    (displacements, interbout times). =#
    if !hmm.only_train_spacetime
        #= Update initial state probabilities =#
        hmm.pinit_turn = (init_count[2] + init_count[3]) / sum(init_count)

        #= Update transition matrix =#
        hmm.transition_matrix .= trans_count / sum(trans_count; dims=2)

        #= Update forward emission probabilities. Forward angles are always centered at μ = 0 =#
        hmm.σforw = fit_mle(Normal, θs, state_marginals[1,:]; mu = 0.0).σ

        #= Update left-right turn emission probabilities. =#
        @assert iszero(state_marginals[2, findall(θs .> 0)])
        @assert iszero(state_marginals[3, findall(θs .< 0)])

        turn_obs = [
            -θs[(θs .< 0) .& (state_marginals[2,:] .> 0)];
            +θs[(θs .> 0) .& (state_marginals[3,:] .> 0)]
        ]
        turn_marginals = [
            state_marginals[2, (θs .< 0) .& (state_marginals[2,:] .> 0)];
            state_marginals[3, (θs .> 0) .& (state_marginals[3,:] .> 0)]
        ]
        @assert all(>(0), turn_obs)
        @assert all(>(0), turn_marginals)

        hmm.turn = fit_mle(typeof(hmm.turn), turn_obs, turn_marginals)

        # enforce minimum alpha
        if hmm.turn.α < hmm.min_alpha
            hmm.turn = Gamma(hmm.min_alpha, hmm.turn.θ)
        end
    end

    # fit displacement emissions
    hmm.forward_displacement = fit_mle(typeof(hmm.forward_displacement), ds, state_marginals[1,:])
    hmm.turn_displacement = fit_mle(typeof(hmm.turn_displacement), ds, state_marginals[2,:] + state_marginals[3,:])

    # fit interbout interval emissions
    hmm.forward_interboutinterval = fit_mle(typeof(hmm.forward_interboutinterval), ts, state_marginals[1,:])
    hmm.turn_interboutinterval = fit_mle(typeof(hmm.turn_interboutinterval), ts, state_marginals[2,:] + state_marginals[3,:])

    # enforce minimum alpha
    if hmm.forward_displacement.α < hmm.min_alpha
        hmm.forward_displacement = Gamma(hmm.min_alpha, hmm.forward_displacement.θ)
    end
    if hmm.turn_displacement.α < hmm.min_alpha
        hmm.turn_displacement = Gamma(hmm.min_alpha, hmm.turn_displacement.θ)
    end
    if hmm.forward_interboutinterval.α < hmm.min_alpha
        hmm.forward_interboutinterval = Gamma(hmm.min_alpha, hmm.forward_interboutinterval.θ)
    end
    if hmm.turn_interboutinterval.α < hmm.min_alpha
        hmm.turn_interboutinterval = Gamma(hmm.min_alpha, hmm.turn_interboutinterval.θ)
    end

    return hmm
end

function load_full_obs(temperature::Int)
    data = load_behaviour_free_swimming_data(temperature)

    angles_trajs = [filter(!isnan, traj) for traj = eachcol(data.dtheta)]
    displacements_trajs = [filter(!isnan, traj) for traj = eachcol(data.displacements)]
    interboutintervals_trajs = [filter(!isnan, traj) for traj = eachcol(data.interboutintervals)]

    @assert map(length, interboutintervals_trajs) == map(length, displacements_trajs) == map(length, angles_trajs)

    return [map(ZebrafishHMM_G3_Sym_Full_Obs, traj...) for traj = zip(angles_trajs, displacements_trajs, interboutintervals_trajs)]
end

# function save_hmm(path::AbstractString, hmm::ZebrafishHMM_G3_Sym_Full)
#     h5open(path, "w") do h5
#         write(h5, "type", "ZebrafishHMM_G3_Sym")
#         write(h5, "pinit_turn", [hmm.pinit_turn])
#         write(h5, "transition_matrix", hmm.transition_matrix)
#         write(h5, "σforw", [hmm.σforw])
#         write(h5, "turn", collect(params(hmm.turn)))
#     end
# end

# function load_hmm(path::AbstractString, ::Type{ZebrafishHMM_G3_Sym_Full})
#     h5open(path, "r") do h5
#         read(h5, "type") == "ZebrafishHMM_G3_Sym" || throw(ArgumentError("HMM type missmatch"))
#         pinit_turn = only(read(h5, "pinit_turn"))
#         transition_matrix = read(h5, "transition_matrix")
#         σforw = only(read(h5, "σforw"))
#         turn_params = read(h5, "turn")
#         return ZebrafishHMM_G3_Sym(pinit_turn, transition_matrix, σforw, Gamma(turn_params...))
#     end
# end
