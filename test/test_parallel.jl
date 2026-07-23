# ----------------------------------------------------------------- #
# Test: parallel stage-cached forward & adjoint mat-vecs            #
#                                                                   #
# Run with:                                                         #
#   julia --project=. -t 4   test/runtests.jl     # 4 threads       #
#   julia --project=. -t auto test/runtests.jl     # all cores       #
#                                                                   #
# Thread-safety check (data races are non-deterministic):            #
#   julia --project=. -t 1 test/runtests.jl  # serial baseline      #
#   julia --project=. -t 2 test/runtests.jl  # 2 threads            #
#   julia --project=. -t 4 test/runtests.jl  # 4 threads            #
#   julia --project=. -t 8 test/runtests.jl  # 8 threads            #
# ----------------------------------------------------------------- #
using Test
using NKSearch
using LinearAlgebra
using Flows
using Random
using Base.Threads

# ============================================================
# 1.  Build test system (Hopf normal form — shared with runtests.jl)
# ============================================================
μ = 1.0
F_sys     = System(μ)                # nonlinear RHS
D         = SystemLinear(μ)          # forward linearised (J)
D_adj     = SystemLinearAdjoint(μ)   # adjoint (J^T)

phase_lock = (dxdt, x) -> F_sys(0, x, dxdt)

# ============================================================
# 2.  Build flows
# ============================================================
dt = 1e-3
G = flow(F_sys, RK4(zeros(2), Flows.NormalMode()), TimeStepConstant(dt))

# Forward linearised: named struct (not lambda) so deepcopy works under threads
L = flow(TangentSystem(D),
         RK4(zeros(2), Flows.DiscreteMode(false)),
         TimeStepFromCache())

# Adjoint: named struct (not lambda) so deepcopy works under threads
L_adj = flow(AdjointTangentSystem(D_adj),
             RK4(zeros(2), Flows.DiscreteMode(true)),
             TimeStepFromCache())

# ============================================================
# 3.  Helper: build a fresh cache pair for N-segment orbit
# ============================================================
function build_caches(N::Int)
    z0 = MVector(ntuple(i -> randn(2) .+ 2.0, N), 2π)
    fwd = NKSearch.StageIterCache(
        ntuple(i -> deepcopy(G),  N),
        ntuple(i -> deepcopy(L),  N),
        nothing,
        (phase_lock,),
        z0)
    adj = NKSearch.AdjointIterSolCache(
        ntuple(i -> deepcopy(L_adj), N),
        (phase_lock,),
        nothing,                         # S = nothing for NS=1
        fwd.xT,
        fwd.dxTdT,
        fwd.z0,
        fwd.tmp,
        fwd.stage_caches,
        fwd.phase_ref)
    return z0, fwd, adj
end

# ============================================================
# 4.  Test header
# ============================================================
@testset "Parallel stage-cached mat-vecs ($(nthreads()) threads)" begin
    Random.seed!(42)

    # --- test across several segment counts ---
    for N in [2, 4, 8]
        z0, fwd, adj = build_caches(N)
        b = similar(z0)
        NKSearch.update!(fwd, b, z0)

        @testset "N = $N segments" begin

            # ========================================================
            # 4a.  Adjoint identity  ⟨J·v, w⟩ = ⟨v, J^T·w⟩
            #     Tests fwd_mul! + adj_mul! consistency.
            #     Wrong scratch (tmp[1] vs tmp[i]) or missing @sync
            #     will break this identity.
            # ========================================================
            @testset "Adjoint identity" begin

                # --- segment-only (zero scalar components) ---
                # Exercises Sites ① (fwd segments) + ⑥ (adj segments).
                v_seg = MVector(ntuple(i -> randn(2), N), 0.0)
                w_seg = MVector(ntuple(i -> randn(2), N), 0.0)

                Jv  = fwd * v_seg
                JTw = adj * w_seg

                @test dot(Jv, w_seg) ≈ dot(v_seg, JTw) atol=1e-10

                # --- scalar-only (zero segment components) ---
                # Exercises Sites ② (fwd period col) + ⑦ (adj period row
                # reduction) + ③ (shift col).  The reduction race on
                # out_d_1 lives here — if present, this test will fail
                # intermittently.
                v_sca = MVector(ntuple(i -> zeros(2), N), randn())
                w_sca = MVector(ntuple(i -> zeros(2), N), randn())

                Jv  = fwd * v_sca
                JTw = adj * w_sca

                @test dot(Jv, w_sca) ≈ dot(v_sca, JTw) atol=1e-10

                # --- full random vectors ---
                v_full = MVector(ntuple(i -> randn(2), N), randn())
                w_full = MVector(ntuple(i -> randn(2), N), randn())

                Jv  = fwd * v_full
                JTw = adj * w_full

                @test dot(Jv, w_full) ≈ dot(v_full, JTw) atol=1e-10
            end

            # ========================================================
            # 4b.  Repeated random trials to catch non-deterministic
            #      races (e.g. reduction on out_d_1).
            # ========================================================
            @testset "Random trials (×50)" begin
                for _ in 1:50
                    v = MVector(ntuple(i -> randn(2), N), randn())
                    w = MVector(ntuple(i -> randn(2), N), randn())

                    Jv  = fwd * v
                    JTw = adj * w

                    @test dot(Jv, w) ≈ dot(v, JTw) atol=1e-10
                end
            end

            # ========================================================
            # 4c.  Gradient-consistency test
            #      ϕ(z) = ½‖F(z)‖²,  ∇ϕ(z) = J(z)^T · F(z)
            #      ϕ(z + ε·dz) − ϕ(z) ≈ ε · ⟨∇ϕ(z), dz⟩
            #
            #      Tests the full L-BFGS pipeline end-to-end:
            #        update!  (Sites ④+⑤, parallel)
            #        adj_mul! (Sites ⑥+⑦+⑧, parallel)
            # ========================================================
            @testset "Gradient consistency" begin
                Fz   = similar(z0)
                dz   = MVector(ntuple(i -> randn(2), N), randn())
                ε    = 1e-6

                # ϕ(z)
                NKSearch.update!(fwd, Fz, z0)
                ϕ0 = 0.5 * dot(Fz, Fz)

                # ∇ϕ(z) = J^T · F(z)  — uses * operator (Base.:* is extended)
                ∇ϕ = adj * Fz

                # ϕ(z + ε·dz)
                z_pert = similar(z0)
                z_pert .= z0 .+ ε .* dz
                NKSearch.update!(fwd, Fz, z_pert)
                ϕ_pert = 0.5 * dot(Fz, Fz)

                # directional derivative check
                lhs = ϕ_pert - ϕ0
                rhs = ε * dot(∇ϕ, dz)

                # First-order Taylor: relative error ~ O(ε)
                @test abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-3
            end

            # ========================================================
            # 4d.  Residual correctness after parallel update!
            #      F(z) ≠ 0 for an unconverged guess.
            # ========================================================
            @testset "update! residual" begin
                NKSearch.update!(fwd, b, z0)
                @test norm(b) > 0
            end
        end
    end

    println("\nAll parallel tests passed with $(nthreads()) threads.")
end
