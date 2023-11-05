function markov_equilibrium(transition_matrix::AbstractMatrix)
    p_eq = real.(eigvecs(transition_matrix')[:, end])
    return p_eq ./ sum(p_eq)
end
