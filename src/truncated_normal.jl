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
    loss(u, p) = -half_normal_loglikelihood(u[1], u[2], p[1], p[2])
    func = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(
        func, [1.0, 1.0], [S1, S2];
        lb = [-Inf, 0.0],
        ub = [Inf, Inf],
    )
    sol = Optimization.solve(prob, OptimizationOptimJL.LBFGS())
    σ = 1 / sol.u[2]
    μ = σ * sol.u[1]
    return (μ, σ)
end

function half_normal_fit_optim(S1::Real, S2::Real)
    loss(u) = -half_normal_loglikelihood(u[1], abs(u[2]), S1, S2)
    # opt = Optim.Options(g_tol = 1e-12,
    #     iterations = 1000,
    #     store_trace = true,
    #     show_trace = true
    # )
    # sol = Optim.optimize(loss, [0.1, 1.0], Optim.LBFGS(), opt; autodiff = :forward)
    sol = Optim.optimize(loss, [0.1, 1.0], Optim.LBFGS(); autodiff = :forward)
    a, b = Optim.minimizer(sol)
    σ = 1 / abs(b)
    μ = σ * a
    return (μ, σ)
end

function half_normal_loglikelihood(a::Real, b::Real, S1::Real, S2::Real)
    return 1/2 * log(2/π) + log(b) - logerfcx(-a / √2) + a * b * S1 - b^2 * S2 / 2
end
