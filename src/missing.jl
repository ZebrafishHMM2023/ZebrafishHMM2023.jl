#= A distribution that treats missing data as Pdist(missing) = 1, at inference time,
with the goal of to marginalizing over the missing data points. =#
struct DistributionMissingWrapper{D}
    dist::D
end

function Base.rand(r::AbstractRNG, d::DistributionMissingWrapper)
    return rand(r, d.dist)
end

function DensityInterface.logdensityof(d::DistributionMissingWrapper, x::Float64)
    return logdensityof(d.dist, x)
end

function DensityInterface.logdensityof(::DistributionMissingWrapper, ::Missing)
    return 0.0
end
