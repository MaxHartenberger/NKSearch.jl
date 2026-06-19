# ----------------------------------------------------------------- #
# Test: Adjoint identity  ⟨J·v, w⟩ = ⟨v, J^T·w⟩  for random v, w    #
# ----------------------------------------------------------------- #
using Test
using NKSearch
using LinearAlgebra
using Flows
using Random

# ============================================================
# 1.  Define the test system (Hopf normal form)
# ============================================================
μ = 1.0
F_sys = System(μ)                    # nonlinear RHS
D     = SystemLinear(μ)              # forward linearised
D_adj = SystemLinearAdjoint(μ)       # adjoint (J^T) RHS

# Phase-locking closure
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
# 3.  Two-segment orbit
# ============================================================
z0 = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)
N = nsegments(z0)

fwd_cache = NKSearch.StageIterCache(
    ntuple(i -> deepcopy(G),  N),
    ntuple(i -> deepcopy(L),  N),
    nothing,
    (phase_lock,),
    z0)

adj_cache = NKSearch.AdjointIterSolCache(
    ntuple(i -> deepcopy(L_adj), N),
    (phase_lock,),
    fwd_cache.xT,
    fwd_cache.z0,
    fwd_cache.tmp,
    fwd_cache.stage_caches)
b = similar(z0)
NKSearch.update!(fwd_cache, b, z0)

# ============================================================
# 4.  Verify residual
# ============================================================
@testset "Residual" begin
    # F(z) should be nonzero (we're at a guess, not a root)
    @test norm(b) > 0
    println("  ‖F(z)‖ = $(norm(b))")
end

# ============================================================
# 5.  Adjoint identity  ⟨J·v, w⟩ = ⟨v, J^T·w⟩
# ============================================================
@testset "Adjoint identity" begin

    # --- segment-only (zero scalar components) ---
    v_seg = MVector(ntuple(i -> randn(2), N), 0.0)
    w_seg = MVector(ntuple(i -> randn(2), N), 0.0)

    Jv  = fwd_cache * v_seg        # J * v
    JTw = adj_cache * w_seg        # J^T * w

    lhs = dot(Jv, w_seg)
    rhs = dot(v_seg, JTw)
    @test lhs ≈ rhs atol=1e-10
    println("  segments only:  diff = $(abs(lhs - rhs))")

    # --- scalar-only (zero segment components) ---
    v_sca = MVector(ntuple(i -> zeros(2), N), randn())
    w_sca = MVector(ntuple(i -> zeros(2), N), randn())

    Jv  = fwd_cache * v_sca
    JTw = adj_cache * w_sca

    lhs = dot(Jv, w_sca)
    rhs = dot(v_sca, JTw)
    @test lhs ≈ rhs atol=1e-10
    println("  scalar only:    diff = $(abs(lhs - rhs))")

    # --- full random vectors ---
    v_full = MVector(ntuple(i -> randn(2), N), randn())
    w_full = MVector(ntuple(i -> randn(2), N), randn())

    Jv  = fwd_cache * v_full
    JTw = adj_cache * w_full

    lhs = dot(Jv, w_full)
    rhs = dot(v_full, JTw)
    @test lhs ≈ rhs atol=1e-10
    println("  full vectors:   diff = $(abs(lhs - rhs))")

    # --- multiple random trials ---
    for trial in 1:5
        v = MVector(ntuple(i -> randn(2), N), randn())
        w = MVector(ntuple(i -> randn(2), N), randn())

        Jv  = fwd_cache * v
        JTw = adj_cache * w

        @test dot(Jv, w) ≈ dot(v, JTw) atol=1e-10
    end
    println("  5 random trials passed.")
end

println("\nAll adjoint identity tests passed.")
