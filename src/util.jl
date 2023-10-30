function mean_square_reorientation(δθs::AbstractVector, q::Int)
    return mean(abs2, sum(δθs[i:(i + q)]) for i = 1:(length(δθs) - q))
end
