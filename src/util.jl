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

# correct version, from https://discourse.julialang.org/t/split-vector-into-n-potentially-unequal-length-subvectors/73548/4?u=e3c6
function makechunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n-1 ? end : c*k+c)] for k = 0:n-1]
end

normalize_transition_matrix(T::AbstractMatrix) = T ./ sum(T; dims=2)

"""
    split_into_repeated_subsequences(seq)

Splits `seq` into subsequences of repeated elements.
"""
function split_into_repeated_subsequences(seq::AbstractVector{Int})
    subseqs = [Int[]]
    for s = seq
        if isempty(last(subseqs)) || s == last(last(subseqs))
            push!(last(subseqs), s)
        else
            push!(subseqs, [s])
        end
    end
    return subseqs
end

"""
    find_repeats(seq)

Returns index ranges of repeated elements in `seq`.
"""
function find_repeats(seq::AbstractVector{Int})
    repeats = [1:0]
    for (n, s) = enumerate(seq)
        if n == 1 || s == seq[n - 1]
            repeats[end] = first(repeats[end]):n
        else
            push!(repeats, n:n)
        end
    end
    return repeats
end
