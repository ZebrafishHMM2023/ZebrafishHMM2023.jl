#= Build a vector of trajectories. Every NaN splits a trajectory in two =#
function build_trajectories(dtheta::AbstractMatrix{T}) where {T}
    all_trajs = Vector{Float64}[]
    for n in axes(dtheta, 2)
        column = dtheta[:, n]
        nan_idx = findall(isnan, column)

        if isempty(nan_idx)
            push!(all_trajs, column)
        else
            if first(nan_idx) > 1
                push!(all_trajs, column[1:(first(nan_idx) - 1)])
            end

            for (i, j) = zip(nan_idx[1:end - 1], nan_idx[begin + 1:end])
                if i + 1 ≤ j - 1
                    push!(all_trajs, column[(i + 1):(j - 1)])
                end
            end
        end
    end
    return all_trajs
end
