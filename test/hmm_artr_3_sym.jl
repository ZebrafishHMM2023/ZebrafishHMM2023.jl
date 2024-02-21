using ForwardDiff: derivative
using Statistics: middle
using Test: @testset, @test
using ZebrafishHMM2023: _find_root_for_artr_sym_f, _find_root_for_artr_sym_df, _find_root_for_artr_sym_bounds

@testset "find_root_for_artr_sym" begin
    q1 = rand(10)
    q2 = rand(15)

    lb, ub = _find_root_for_artr_sym_bounds(q1, q2)
    λ = middle(lb, ub)

    @test _find_root_for_artr_sym_df(q1, q2, λ) ≈ derivative(λ -> _find_root_for_artr_sym_f(q1, q2, λ), λ)

    @test _find_root_for_artr_sym_f(q1, q2, lb) == -Inf
    @test _find_root_for_artr_sym_f(q1, q2, ub) == Inf
end
