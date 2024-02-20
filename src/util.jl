function mean_square_reorientation(δθs::AbstractVector, q::Int)
    return mean(abs2, sum(δθs[i:(i + q)]) for i = 1:(length(δθs) - q))
end

# filter observation sequence according to some condition `cond`
function filter_obs(cond, obs_seq::AbstractVector, state_marginals::AbstractVector)
    _idx = findall(cond, obs_seq)
    return map(identity, obs_seq[_idx]), map(identity, state_marginals[_idx])
end

# Split `list` into `n` chunks of approximately same size
function chunks(list, n::Int)
    len = ceil(Int, length(list) / n)
    return [list[i:min(end, i + len - 1)] for i = firstindex(list):len:lastindex(list)]
end

normalize_transition_matrix(T::AbstractMatrix) = T ./ sum(T; dims=2)
