function markov_equilibrium(transition_matrix::AbstractMatrix)
    p_eq = real.(eigvecs(transition_matrix')[:, end])
    return p_eq ./ sum(p_eq)
end

#= Stubborness factor for a 4-state model, with the states:
    FL, FR, L, R (in that order). =#
function _stubborness_factor_4_state(transition_matrix::AbstractMatrix, q::Int)
    @assert size(transition_matrix) == (4, 4)

    p0 = markov_equilibrium(transition_matrix)
    TFq = transition_matrix[1:2,1:2]^q
    sL = p0[3] * transition_matrix[3,1:2]' * TFq * transition_matrix[1:2,3]
    sR = p0[4] * transition_matrix[4,1:2]' * TFq * transition_matrix[1:2,4]
    wL = p0[3] * transition_matrix[3,1:2]' * TFq * transition_matrix[1:2,4]
    wR = p0[4] * transition_matrix[4,1:2]' * TFq * transition_matrix[1:2,3]
    return (sL + sR) / (wL + wR)
end

#= Stubborness factor for a 3-state model, with the states:
    F, L, R (in that order). =#
function _stubborness_factor_3_state(transition_matrix::AbstractMatrix)
    @assert size(transition_matrix) == (3, 3)
    p0 = markov_equilibrium(transition_matrix)
    sL = p0[2] * transition_matrix[2,1] * transition_matrix[1,2]
    sR = p0[3] * transition_matrix[3,1] * transition_matrix[1,3]
    wL = p0[2] * transition_matrix[2,1] * transition_matrix[1,3]
    wR = p0[3] * transition_matrix[3,1] * transition_matrix[1,2]
    return (sL + sR) / (wL + wR)
end
