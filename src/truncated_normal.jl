"""
    half_normal_fit(S1, S2)

Fits a truncated normal distribution to data, where S1 = <x> and S2 = <x^2> are empirical
moments. The truncation interval is fixed to [0, ∞).
"""
function half_normal_fit(S1::Real, S2::Real; maxiter=1000, atol=1e-10, rtol=1e-10, γ=1e-10)
    S1 > 0 && S2 > 0 || throw(ArgumentError("S1, S2 must be positive; got $S1, $S2"))
    S1^2 < S2 || throw(ArgumentError("S1^2 < S2 violated; got $S1, $S2"))
    γ > 0 || throw(ArgumentError("γ must be positive; got $γ"))

    s = S2 / S1^2
    r = half_normal_fit_inv(s; maxiter, atol, rtol, γ)
    σ = S1/2 * (sqrt(r^2 + 4s) - r)
    μ = r * σ
    return (μ, σ)
end

function half_normal_fit_inv(s::Real; maxiter, atol, rtol, γ)
    s > 1 || throw(ArgumentError("s must be larger than 1; got $s"))
    γ > 0 || throw(ArgumentError("γ must be positive; got $γ"))

    r_left_left = -1
    while __psiinv(r_left_left; γ) ≤ s
        r_left_left *= 2
    end
    r_left = r_left_left
    while true
        if half_normal_psi(r_left) + γ * r_left ≤ 0
            r_left = r_left_left / 2
        elseif __psiinv(r_left; γ) < s
            r_left_left = r_left
        else
            break
        end
    end
    @assert half_normal_psi(r_left) + γ * r_left > 0
    r_right = 10
    while __psiinv(r_right; γ) > s
        r_right *= 2
    end
    @assert __psiinv(r_left; γ) ≥ s ≥ __psiinv(r_right; γ)
    for iter = 1:maxiter
        r = (r_left + r_right) / 2
        if __psiinv(r; γ) > s
            r_left = r
        else
            r_right = r
        end
        if isapprox(r_left, r_right; atol, rtol)
            return r
        end
    end
end

function __psiinv(r::Real; γ::Real)
    psiinv = 1/(half_normal_psi(r) + γ * r)
    return psiinv * (psiinv + r)
end

function half_normal_loglikelihood(μ::Real, σ::Real, S1::Real, S2::Real)
    a = μ/σ
    b = 1/σ
    return 1/2 * log(2/π) + log(b) - logerfcx(-a / √2) + a * b * S1 - b^2 * S2 / 2
end

half_normal_psi(x) = 2/sqrt(2π) / erfcx(-x/√2) + x
