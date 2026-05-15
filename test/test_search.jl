@testset "search_linesearch                      " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(couple(F, D),
             RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()),
             TimeStepConstant(1e-3))

    # FIXME: tr_direct does not pass
    for method in (:tr_direct,
                   #:ls_direct,
                   #:ls_iterative,
                   :tr_iterative,
                   )
        # define initial guess, a slightly perturbed orbit
        z = MVector(([2, 0.0], [-2, 0.0]), 2π)

        # search
        search!(G,
                L,
                (dxdt, x)->F(0, x, dxdt),
                z,
                Options(maxiter=25,
                        dz_norm_tol=1e-18,
                        gmres_verbose=false,
                        e_norm_tol=1e-18,
                        gmres_maxiter=5,
                        verbose=true,
                        tr_radius_init=0.001,
                        method=method,
                        ϵ=1e-7,
                        gmres_start=dz->dz))

        # solution is a loop of unit radius and with T = 2π
        @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
        @test abs(z.d[1] - 2π ) < 1e-9
    end
end
