# MSR = Mean Squared Reorientation

function MSR(traj::AbstractVector, q::Int)
    return mean(abs2, sum(traj[i:(i + q)]) for i = 1:(length(traj) - q))
end
