struct SignedGamma
    gamma::Gamma{Float64}
    positive::Bool
end

function SignedGamma(gamma::Gamma; positive::Bool)
    return SignedGamma(gamma, positive)
end

function SignedGamma(α::Real, θ::Real; positive::Bool)
    return SignedGamma(Gamma(α, θ); positive)
end

function Base.rand(r::AbstractRNG, d::SignedGamma)
    if d.positive
        return rand(r, d.gamma)
    else
        return -rand(r, d.gamma)
    end
end

function DensityInterface.logdensityof(d::SignedGamma, x::Real)
    if iszero(x)
        return oftype(logdensityof(d.gamma, x), -Inf)
    elseif d.positive
        return logdensityof(d.gamma, x)
    else
        return logdensityof(d.gamma, -x)
    end
end
