"""
    half_normal_fit(data::AbstractVector)

Fits a truncated normal distribution to the data. The truncation interval is fixed to [0, ∞).
"""
function half_normal_fit(data::AbstractVector{<:Real})
    S1 = mean(data)
    S2 = mean(abs2, data)
    return half_normal_fit(S1, S2)
end

function half_normal_fit(S1::Real, S2::Real)
    return half_normal_fit_optim(S1, S2)
end

function half_normal_fit_optimization(S1::Real, S2::Real)
    loss(u, p) = -half_normal_loglikelihood(u[1], abs(u[2]), p[1], p[2])
    func = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(
        func, [1.0, 1.0], [S1, S2];
        #lb = [-1e3, 1e-3],
        #ub = [+1e3, 1e4],
    )
    sol = Optimization.solve(prob, OptimizationOptimJL.Newton())
    σ = 1 / abs(sol.u[2])
    μ = σ * sol.u[1]
    return (μ, σ)
end

function half_normal_fit_optim(S1::Real, S2::Real)
    S1 > 0 && S2 > 0 || throw(ArgumentError("S1 and S2 must be positive"))
    loss(u) = -half_normal_loglikelihood(u[1], abs(u[2]), S1, S2)
    # opt = Optim.Options(g_tol = 1e-12,
    #     iterations = 1000,
    #     store_trace = true,
    #     show_trace = true
    # )
    # sol = Optim.optimize(loss, [0.1, 1.0], Optim.LBFGS(), opt; autodiff = :forward)
    sol = Optim.optimize(loss, [0.1, 1.0], Optim.Newton(); autodiff = :forward)
    a, b = Optim.minimizer(sol)
    σ = 1 / abs(b)
    μ = σ * a
    return (μ, σ)
end

function half_normal_loglikelihood(a::Real, b::Real, S1::Real, S2::Real)
    return 1/2 * log(2/π) + log(b) - logerfcx(-a / √2) + a * b * S1 - b^2 * S2 / 2
end

function half_normal_fit_iter(S1::Real, S2::Real; maxiter = 1000, atol = 1e-6, rtol = 1e-6, damping = 1e-3)
    S1 > 0 && S2 > 0 || throw(ArgumentError("S1 and S2 must be positive"))
    μ = 0.01S1
    σ = sqrt(S2)

    for iter = 1:maxiter
        r = μ / σ
        y = r * half_normal_psi(r) # μ/σ * S1/σ

        σ_new = sqrt(S2 / (1 + y))
        μ_new = r * σ_new

        converged = isapprox(σ_new, σ; atol, rtol) && isapprox(μ_new, μ; atol, rtol)

        μ = damp(μ_new, μ; damping)
        σ = damp(σ_new, σ; damping)

        println("μ=$μ, σ=$σ")

        if converged
            return (μ, σ)
        end
    end

    @warn "Maximum number of iterations ($maxiter) reached"
    return nothing
end

half_normal_psi(x) = 2/sqrt(2π) / erfcx(-x/√2) + x
