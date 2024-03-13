# uses Gaussian emission dist for mL - mR
struct HMM_ARTR_m_diff <: HiddenMarkovModels.AbstractHMM
    transition_matrix::Matrix{Float64} # T[y,z] = P(y -> z)
    μ::Vector{Float64}
    σ::Vector{Float64}
    pinit::Vector{Float64} # initial state probabilities
    λreg::Vector{Float64} # regularization for μ

    function HMM_ARTR_m_diff(
        transition_matrix::AbstractMatrix{<:Real},
        μ::AbstractVector{<:Real},
        σ::AbstractVector{<:Real},
        pinit::AbstractVector{<:Real},
        λreg::AbstractVector{<:Real}
    )
        @assert length(pinit) == size(transition_matrix, 1) == size(transition_matrix, 2) == length(μ) == length(σ) == length(λreg) # number of states
        @assert all(≥(0), transition_matrix)
        @assert sum(transition_matrix; dims=2) ≈ ones(size(transition_matrix, 1))
        @assert all(≥(0), pinit)
        @assert all(>(0), σ)
        @assert all(≥(0), λreg)
        @assert all(isfinite, μ)
        @assert all(isfinite, σ)
        @assert all(isfinite, λreg)
        @assert isapprox(sum(pinit), 1; rtol=1e-8)
        return new(transition_matrix, μ, σ, pinit, λreg)
    end
end

function HMM_ARTR_m_diff(
    transition_matrix::AbstractMatrix{<:Real},
    μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2) == length(μ) == length(σ)
    pinit = ones(size(transition_matrix, 1)) / size(transition_matrix, 1)
    λreg = zeros(length(μ))
    return HMM_ARTR_m_diff(transition_matrix, μ, σ, pinit, λreg)
end

# number of hidden states
Base.length(hmm::HMM_ARTR_m_diff) = size(hmm.transition_matrix, 1)

HiddenMarkovModels.transition_matrix(hmm::HMM_ARTR_m_diff) = hmm.transition_matrix
HiddenMarkovModels.initial_distribution(hmm::HMM_ARTR_m_diff) = hmm.pinit

function HiddenMarkovModels.obs_distribution(hmm::HMM_ARTR_m_diff, i::Int)
    return Normal(hmm.μ[i], hmm.σ[i])
end

function StatsAPI.fit!(
    hmm::HMM_ARTR_m_diff,
    init_count::AbstractVector,
    trans_count::AbstractMatrix,
    obs_seq::AbstractVector{<:Real},
    state_marginals::AbstractMatrix
)
    @assert length(init_count) == length(hmm)
    @assert size(trans_count) == (length(hmm), length(hmm))
    @assert size(state_marginals) == (length(hmm), length(obs_seq))
    @assert dropdims(sum(state_marginals; dims=1); dims=1) ≈ ones(length(obs_seq))

    hmm.pinit .= normalize(init_count, 1)
    hmm.transition_matrix .= trans_count ./ sum(trans_count; dims=2)

    for i = 1:length(hmm)
        d = fit_mle(Normal, obs_seq, state_marginals[i,:])
        hmm.μ[i] = d.μ
        hmm.σ[i] = d.σ
    end

    return hmm
end

function save_hmm(path::AbstractString, hmm::HMM_ARTR_m_diff)
    h5open(path, "w") do h5
        write(h5, "type", "HMM_ARTR_m_diff")
        write(h5, "transition_matrix", hmm.transition_matrix)
        write(h5, "μ", hmm.μ)
        write(h5, "σ", hmm.σ)
        write(h5, "pinit", hmm.pinit)
        write(h5, "λreg", hmm.λreg)
    end
end

function load_hmm(path::AbstractString, ::Type{HMM_ARTR_m_diff})
    h5open(path, "r") do h5
        read(h5, "type") == "HMM_ARTR_m_diff" || throw(ArgumentError("HMM type missmatch"))
        transition_matrix = read(h5, "transition_matrix")
        μ = read(h5, "μ")
        σ = read(h5, "σ")
        pinit = read(h5, "pinit")
        λreg = read(h5, "λreg")
        return HMM_ARTR_m_diff(transition_matrix, μ, σ, pinit, λreg)
    end
end
