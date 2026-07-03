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

    for method in (#:ls_direct,      # only works with single thread (julia -t 1)
                   :ls_iterative,   
                   #:tr_direct,      # only works with single thread (julia -t 1)
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
                        dz_norm_tol=1e-8,
                        gmres_verbose=false,
                        e_norm_tol=1e-8,
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

@testset "search_lbfgs                           " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)
    D_adj = SystemLinearAdjoint(μ)

    # define propagators (stage-cache style for StageIterCache)
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(TangentSystem(D),
             RK4(zeros(2), Flows.DiscreteMode(false)),
             TimeStepFromCache())

    # Adjoint flow for L-BFGS methods
    adj_flow = flow(
        AdjointTangentSystem(D_adj),
        RK4(zeros(2), Flows.DiscreteMode(true)),
        TimeStepFromCache())

    for method in (:lbfgs_opt,)
        @testset "$method" begin
            # define initial guess, a slightly perturbed orbit
            z = MVector(([2, 0.0], [-2, 0.0]), 2π)

            # search
            search!(G,
                    L,
                    adj_flow,
                    (dxdt, x)->F(0, x, dxdt),
                    z,
                    Options(maxiter=100,
                            dz_norm_tol=1e-8,
                            gmres_verbose=false,
                            e_norm_tol=1e-8,
                            gmres_maxiter=5,
                            verbose=true,
                            tr_radius_init=0.0001,
                            method=method,
                            ls_maxiter=20,
                            gmres_start=dz->dz,
                            lbfgs_memory=5));

            # solution is a loop of unit radius and with T = 2π
            @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
            @test abs(z.d[1] - 2π ) < 1e-9

        end
    end
end
