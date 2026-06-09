@testset "search_linesearch                      " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)
    D_adj = SystemLinearAdjoint(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow((t, x, v, dv) -> D(t, x, dv, v, dv),
             RK4(zeros(2), Flows.DiscreteMode(false)),
             TimeStepFromCache())

    # Adjoint flow for L-BFGS methods
    adj_flow = flow(
        (t, x, w, dw) -> D_adj(x, w, dw),
        RK4(zeros(2), Flows.DiscreteMode(true)),
        TimeStepFromCache())

    for method in (:tr_iterative,
                   :lbfgs_opt,
                   )
        @testset "$method" begin
            # define initial guess, a slightly perturbed orbit
            z = MVector(([2, 0.0], [-2, 0.0]), 2π)
            # z = MVector(([1.5, 0.0], [-1.5, 0.0]), 6)
            # z = MVector(([50, 0.0], [-50, 0.0]), 100)
            # search
            search!(G,
                    L,
                    (dxdt, x)->F(0, x, dxdt),
                    z,
                    Options(maxiter=100,
                            dz_norm_tol=1e-16,
                            gmres_verbose=false,
                            e_norm_tol=1e-16,
                            gmres_maxiter=5,
                            verbose=true,
                            tr_radius_init=0.0001,
                            method=method,
                            ls_maxiter=20,
                            gmres_start=dz->dz,
                            lbfgs_memory=5,
                            lbfgs_adj_system=(adj_flow, adj_flow)));

            # solution is a loop of unit radius and with T = 2π
            @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
            @test abs(z.d[1] - 2π ) < 1e-9

        end
    end
end

