# Here we try to enforce that the 3rd state has the same mean field in the left and right sides.
# Thus resembling a "forward" state.
struct HMM_ARTR_3_SYM <: HiddenMarkovModels.AbstractHMM
    transition_matrix::Matrix{Float64} # T[i,j] = P(i -> j)
    h::Matrix{Float64} # h[:,i] = fields in hidden state 'i'
    pinit::AbstractVector{Float64} # initial state probabilities
    L::Int # number of left neurons
    pseudocount::Float64 # pseudocount for the inference of 'h'
    function HMM_ARTR_3_SYM(transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, pinit::AbstractVector{<:Real}, L::Int, pseudocount::Float64)
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
        @assert all(≥(0), transition_matrix)
        @assert all(≈(1), sum(transition_matrix; dims=2))
        @assert all(≥(0), pinit)
        @assert sum(pinit) ≈ 1
        @assert pseudocount ≥ 0
        @assert length(pinit) == 3 # only 3 states
        @assert 0 ≤ L ≤ size(h, 1)
        return new(transition_matrix, h, pinit, L, pseudocount)
    end
end

function HMM_ARTR_3_SYM(transition_matrix::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, L::Int, pseudocount::Real=0.0)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == size(h, 2)
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    return HMM_ARTR_3_SYM(transition_matrix, h, pinit, L, pseudocount)
end

Base.length(hmm::HMM_ARTR_3_SYM) = size(hmm.transition_matrix, 1) # number of hidden states
HiddenMarkovModels.transition_matrix(hmm::HMM_ARTR_3_SYM) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_ARTR_3_SYM) = hmm.pinit

function HiddenMarkovModels.obs_distribution(hmm::HMM_ARTR_3_SYM, i::Int)
    return NeuronsBinaryDistribution(view(hmm.h, :, i))
end

function StatsAPI.fit!(hmm::HMM_ARTR_3_SYM, init_count::AbstractVector, trans_count::AbstractMatrix, obs_seq::AbstractVector, state_marginals::AbstractMatrix)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert all(≈(1), sum(state_marginals; dims=1))

    hmm.pinit .= normalize(init_count, 1)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    obs_mat = stack(obs_seq)
    @assert size(obs_mat) == (size(hmm.h, 1), length(obs_seq))

    obs_p = stack(obs_seq) * (state_marginals ./ sum(state_marginals; dims=2))'
    @assert size(obs_p) == size(hmm.h)

    λ = hmm.pseudocount / length(obs_seq)
    q = (1 - λ) * obs_p .+ λ * 0.5
    @assert all(0 .≤ q .≤ 1)

    hmm.h .= log.(q)

    # We impose that the 3rd state has the same mean field in the left and right sides
    mean_1 = mean(hmm.h[begin:hmm.L, 3])
    mean_2 = mean(hmm.h[hmm.L + 1:end, 3])

    hmm.h[begin:hmm.L, 3] .-= mean_2
    hmm.h[hmm.L + 1:end, 3] .-= mean_1

    return hmm
end

# function save_hmm(path::AbstractString, hmm::HMM_ARTR)
#     h5open(path, "w") do h5
#         write(h5, "type", "HMM_ARTR")
#         write(h5, "transition_matrix", hmm.transition_matrix)
#         write(h5, "h", hmm.h)
#         write(h5, "pinit", hmm.pinit)
#         write(h5, "h_abs_max", [h_abs_max])
#     end
# end

# function load_hmm(path::AbstractString, ::Type{HMM_ARTR})
#     h5open(path, "r") do h5
#         read(h5, "type") == "HMM_ARTR" || throw(ArgumentError("HMM type missmatch"))
#         transition_matrix = read(h5, "transition_matrix")
#         h = read(h5, "h")
#         pinit = read(h5, "pinit")
#         h_abs_max = only(read(h5, "h_abs_max"))
#         return HMM_ARTR(transition_matrix, h, pinit, h_abs_max)
#     end
# end

function _find_root_for_artr_sym(q1::AbstractVector, q2::AbstractVector; maxiters::Int=10, )
    L1 = length(q1)
    L2 = length(q2)

    # bounds
    lb, ub = _find_root_for_artr_sym_bounds(q1, q2)

    # initial value
    λ0 = middle(lb, ub)

    for iter = 1:maxiters
        lhs = mean(log, (q1 .+ λ/L1) / (1 .- q1 .- λ/L1))
        rhs = mean(log, (q2 .- λ/L2) / (1 .- q2 .- λ/L2))
        if lhs < rhs

        end
    end
end

function _find_root_for_artr_sym_f(q1::AbstractVector, q2::AbstractVector, λ::Real)
    L1 = length(q1)
    L2 = length(q2)

    lhs = mean(log, (q1 .+ λ/L1) ./ (1 .- q1 .- λ/L1))
    rhs = mean(log, (q2 .- λ/L2) ./ (1 .- q2 .+ λ/L2))

    return lhs - rhs
end

function _find_root_for_artr_sym_bounds(q1::AbstractVector, q2::AbstractVector)
    L1 = length(q1)
    L2 = length(q2)

    lb = max(-minimum(q1) * L1, -(1 - maximum(q2)) * L2)
    ub = min((1 - maximum(q1)) * L1, minimum(q2) * L2)

    return lb, ub
end

function _find_root_for_artr_sym_df(q1::AbstractVector, q2::AbstractVector, λ::Real)
    L1 = length(q1)
    L2 = length(q2)

    lhs = sum(inv, (L1 * (1 .- q1) .- λ) .* (L1 .* q1 .+ λ))
    rhs = sum(inv, (L2 * (1 .- q2) .+ λ) .* (L2 .* q2 .- λ))

    return lhs + rhs
end
