#=
Currently baum_welch from HMMs with check_loglikelihood_increasing=false will stop running
whenever logL_evolution[end] - logL_evolution[end - 1] < atol, which can be true even if
the log-likelihood difference is quite large (just because it can be negative).

While we wait for a fix of this issue, we can use the following hacky workaround. We define
a custom ATol number type which translates progress < atol to abs(progress) < atol. We can
then use this `atol::ATol` in the baum_welch call.
=#

struct ATol <: Real
    atol::Float64
end

function Base.:<(progress::Real, atol::ATol)
    return abs(progress) < atol.atol
end
