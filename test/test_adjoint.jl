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
    nothing,                         # S = nothing for NS=1
    fwd_cache.xT,
    fwd_cache.dxTdT,
    fwd_cache.z0,
    fwd_cache.tmp,
    fwd_cache.stage_caches,
    fwd_cache.phase_ref)
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

# ============================================================
# 6.  Adjoint identity with spatial shift (NS = 2)
# ============================================================
# The Hopf normal form is rotationally symmetric.
# Spatial shift S(x, s) rotates state x by angle s.
# Its derivative dS/ds is the infinitesimal generator (-x[2], x[1]).

struct SpatialShift end
function (::SpatialShift)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    out[1] = -x[2]
    out[2] =  x[1]
    return out
end

@testset "Adjoint identity (NS=2, with spatial shift)" begin
    S_op   = SpatialShift()
    dS_op  = SpatialShiftDerivative()
    D_ns2  = (phase_lock, dS_op)

    # Initial guess with zero spatial shift
    z0_ns2 = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π, 0.0)

    fwd_ns2 = NKSearch.StageIterCache(
        ntuple(i -> deepcopy(G),  N),
        ntuple(i -> deepcopy(L),  N),
        S_op,
        D_ns2,
        z0_ns2)

    adj_ns2 = NKSearch.AdjointIterSolCache(
        ntuple(i -> deepcopy(L_adj), N),
        D_ns2,
        S_op,
        fwd_ns2.xT,
        fwd_ns2.dxTdT,
        fwd_ns2.z0,
        fwd_ns2.tmp,
        fwd_ns2.stage_caches,
        fwd_ns2.phase_ref)

    b_ns2 = similar(z0_ns2)
    NKSearch.update!(fwd_ns2, b_ns2, z0_ns2)

    @test norm(b_ns2) > 0
    println("  ‖F(z)‖ (NS=2) = $(norm(b_ns2))")

    # --- segment-only (zero scalar components) ---
    v_seg = MVector(ntuple(i -> randn(2), N), 0.0, 0.0)
    w_seg = MVector(ntuple(i -> randn(2), N), 0.0, 0.0)

    Jv  = fwd_ns2 * v_seg
    JTw = adj_ns2 * w_seg

    @test dot(Jv, w_seg) ≈ dot(v_seg, JTw) atol=1e-10
    println("  segments only:  diff = $(abs(dot(Jv, w_seg) - dot(v_seg, JTw)))")

    # --- scalar-only (zero segment components) ---
    v_sca = MVector(ntuple(i -> zeros(2), N), randn(), randn())
    w_sca = MVector(ntuple(i -> zeros(2), N), randn(), randn())

    Jv  = fwd_ns2 * v_sca
    JTw = adj_ns2 * w_sca

    @test dot(Jv, w_sca) ≈ dot(v_sca, JTw) atol=1e-10
    println("  scalar only:    diff = $(abs(dot(Jv, w_sca) - dot(v_sca, JTw)))")

    # --- full random vectors ---
    for trial in 1:10
        v = MVector(ntuple(i -> randn(2), N), randn(), randn())
        w = MVector(ntuple(i -> randn(2), N), randn(), randn())

        Jv  = fwd_ns2 * v
        JTw = adj_ns2 * w

        @test dot(Jv, w) ≈ dot(v, JTw) atol=1e-10
    end
    println("  10 random NS=2 trials passed.")
end

println("\nAll adjoint identity tests (NS=1 and NS=2) passed.")
