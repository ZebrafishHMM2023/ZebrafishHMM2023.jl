
prob = NonlinearSolve.IntervalNonlinearProblem((λ, _) -> _find_root_for_artr_sym_f(q1, q2, λ), (lb, ub))
sol = NonlinearSolve.solve(prob, NonlinearSolve.ITP(); abstol = 0.01)


f(u, p) = u * u - 2.0
uspan = (1.0, 2.0) # brackets
prob_int = IntervalNonlinearProblem(f, uspan)
sol = solve(prob_int)
