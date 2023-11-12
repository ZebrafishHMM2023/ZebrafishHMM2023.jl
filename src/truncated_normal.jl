"""
    half_normal_fit(S1, S2)

Fits a truncated normal distribution to data, where S1 = <x> and S2 = <x^2> are empirical
moments. The truncation interval is fixed to [0, ∞).
"""
function half_normal_fit(S1::Real, S2::Real; maxiter=1000, atol=1e-10, rtol=1e-10)
    @assert S1 ≥ 0 && S2 ≥ 0
    s = S2 / S1^2
    r = half_normal_fit_inv(s; maxiter, atol, rtol)
    σ = S1/2 * (sqrt(r^2 + 4s) - r)
    μ = r * σ
    return (μ, σ)
end

function half_normal_fit_inv(s::Real; maxiter, atol, rtol)
    @assert 1 < s < 2
    r_left = -10
    r_right = 10
    while __psiinv(r_left) ≤ s
        r_left *= 5
    end
    while __psiinv(r_right) ≥ s
        r_right *= 5
    end
    for iter = 1:maxiter
        r = (r_left + r_right) / 2
        if __psiinv(r) > s
            r_left = r
        else
            r_right = r
        end
        if isapprox(r_left, r_right; atol, rtol)
            return r
        end
    end
end

function __psiinv(r::Real)
    psiinv = 1/half_normal_psi(r)
    return psiinv * (psiinv + r)
end

function half_normal_loglikelihood(μ::Real, σ::Real, S1::Real, S2::Real)
    a = μ/σ
    b = 1/σ
    return 1/2 * log(2/π) + log(b) - logerfcx(-a / √2) + a * b * S1 - b^2 * S2 / 2
end

half_normal_psi(x) = 2/sqrt(2π) / erfcx(-x/√2) + x
