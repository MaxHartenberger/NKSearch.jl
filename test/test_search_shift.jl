# ----------------------------------------------------------------- #
# Test: RPO convergence with spatial shift (NS=2)                    #
#                                                                   #
# System: Two decoupled Hopf oscillators with incommensurate         #
#         frequencies, sharing a common spatial shift symmetry.      #
#                                                                   #
#   Oscillator 1 (wavenumber k1=1):                                  #
#     x1' = -w1*y1 + mu*x1*(1 - r1)     r1 = sqrt(x1^2 + y1^2)     #
#     y1' =  w1*x1 + mu*y1*(1 - r1)                                  #
#                                                                   #
#   Oscillator 2 (wavenumber k2=2):                                  #
#     x2' = -w2*y2 + mu*x2*(1 - r2)     r2 = sqrt(x2^2 + y2^2)     #
#     y2' =  w2*x2 + mu*y2*(1 - r2)                                  #
#                                                                   #
# Spatial shift S(x, s) rotates oscillator j by angle kj*s:         #
#   S((x1,y1,x2,y2), s) = (R(s)*(x1,y1),  R(2s)*(x2,y2))           #
#                                                                   #
# Generator: dS(out, x) = (-y1, x1, -2*y2, 2*x2)                   #
#                                                                   #
# Analytical RPO (n1=2, n2=3):                                      #
#   T = 2*pi / (2 - sqrt(2))  ~ 10.723805                           #
#   s = 4*pi - T              ~  1.842565                           #
# ----------------------------------------------------------------- #
using Test
using NKSearch
using LinearAlgebra
using Flows

# ============================================================
# 1.  Parameters & analytical reference values
# ============================================================
const mu  = 1.0
const w1  = 1.0
const w2  = sqrt(2)              # incommensurate -> unique (T, s)
const dim = 4                    # (x1, y1, x2, y2)

# Solve:  w1*T + 1*s = 2*pi*n1,   w2*T + 2*s = 2*pi*n2   (n1=2, n2=3)
const T_exact = 2pi / (2 - w2)          # ~ 10.723805
const s_exact = 4pi - T_exact           # ~  1.842565

println("Analytical RPO: T = $T_exact, s = $s_exact")

# ============================================================
# 2.  Nonlinear RHS
# ============================================================
struct TwoMode
    mu::Float64
    w1::Float64
    w2::Float64
end
function (s::TwoMode)(t, u, dudt)
    x1, y1, x2, y2 = u[1], u[2], u[3], u[4]
    r1 = sqrt(x1^2 + y1^2)
    r2 = sqrt(x2^2 + y2^2)
    @inbounds dudt[1] = -s.w1*y1 + s.mu*x1*(1 - r1)
    @inbounds dudt[2] =  s.w1*x1 + s.mu*y1*(1 - r1)
    @inbounds dudt[3] = -s.w2*y2 + s.mu*x2*(1 - r2)
    @inbounds dudt[4] =  s.w2*x2 + s.mu*y2*(1 - r2)
    return dudt
end

# ============================================================
# 3.  Forward linearised RHS (5-arg, for Flows.couple)
#     Fills ONLY dvdt = J*v.  dudt is handled by TwoMode.
# ============================================================
struct TwoModeLin
    mu::Float64
    w1::Float64
    w2::Float64
    J::Matrix{Float64}
    TwoModeLin(mu, w1, w2) = new(mu, w1, w2, zeros(dim, dim))
end
function (s::TwoModeLin)(t, u, dudt, v, dvdt)
    x1, y1, x2, y2 = u[1], u[2], u[3], u[4]
    r1 = sqrt(x1^2 + y1^2)
    r2 = sqrt(x2^2 + y2^2)

    # Block 1 (oscillator 1, 2x2)
    if r1 > 0
        s.J[1,1] = s.mu*(1 - r1 - x1^2/r1)
        s.J[1,2] = -s.w1 - s.mu*x1*y1/r1
        s.J[2,1] =  s.w1 - s.mu*x1*y1/r1
        s.J[2,2] = s.mu*(1 - r1 - y1^2/r1)
    end
    # Block 2 (oscillator 2, 2x2)
    if r2 > 0
        s.J[3,3] = s.mu*(1 - r2 - x2^2/r2)
        s.J[3,4] = -s.w2 - s.mu*x2*y2/r2
        s.J[4,3] =  s.w2 - s.mu*x2*y2/r2
        s.J[4,4] = s.mu*(1 - r2 - y2^2/r2)
    end
    return mul!(dvdt, s.J, v)
end

# ============================================================
# 4.  Adjoint RHS (3-arg, computes J^T * w for L-BFGS gradient)
# ============================================================
struct TwoModeAdj
    mu::Float64
    w1::Float64
    w2::Float64
    J::Matrix{Float64}
    TwoModeAdj(mu, w1, w2) = new(mu, w1, w2, zeros(dim, dim))
end
function (s::TwoModeAdj)(u, w, dw)
    x1, y1, x2, y2 = u[1], u[2], u[3], u[4]
    r1 = sqrt(x1^2 + y1^2)
    r2 = sqrt(x2^2 + y2^2)

    if r1 > 0
        s.J[1,1] = s.mu*(1 - r1 - x1^2/r1)
        s.J[1,2] = -s.w1 - s.mu*x1*y1/r1
        s.J[2,1] =  s.w1 - s.mu*x1*y1/r1
        s.J[2,2] = s.mu*(1 - r1 - y1^2/r1)
    end
    if r2 > 0
        s.J[3,3] = s.mu*(1 - r2 - x2^2/r2)
        s.J[3,4] = -s.w2 - s.mu*x2*y2/r2
        s.J[4,3] =  s.w2 - s.mu*x2*y2/r2
        s.J[4,4] = s.mu*(1 - r2 - y2^2/r2)
    end
    return mul!(dw, s.J', w)
end

# ============================================================
# 5.  Named wrappers (mandatory for thread safety)
# ============================================================
struct TanSys{D}; D::D; end
(s::TanSys)(t, x, v, dv) = s.D(t, x, dv, v, dv)

struct AdjSys{D}; D::D; end
(s::AdjSys)(t, x, w, dw) = s.D(x, w, dw)

# ============================================================
# 6.  Spatial shift operator & generator
# ============================================================
struct SpatialShift end
function (::SpatialShift)(x, s)
    # Mode 1: rotate by s  (wavenumber k1 = 1)
    c1, sn1 = cos(s), sin(s)
    x1, y1 = x[1], x[2]
    x[1] = c1*x1 - sn1*y1
    x[2] = sn1*x1 + c1*y1
    # Mode 2: rotate by 2s (wavenumber k2 = 2)
    c2, sn2 = cos(2s), sin(2s)
    x3, y3 = x[3], x[4]
    x[3] = c2*x3 - sn2*y3
    x[4] = sn2*x3 + c2*y3
    return x
end

struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    # d/ds [R(s)*(x1,y1)]|s=0 = (-y1, x1)
    out[1] = -x[2]
    out[2] =  x[1]
    # d/ds [R(2s)*(x2,y2)]|s=0 = (-2*y2, 2*x2)
    out[3] = -2*x[4]
    out[4] =  2*x[3]
    return out
end

# ============================================================
# 7.  Build flows
# ============================================================
F_sys = TwoMode(mu, w1, w2)
D_lin = TwoModeLin(mu, w1, w2)
D_adj = TwoModeAdj(mu, w1, w2)
S_op  = SpatialShift()
dS_op = SpatialShiftDerivative()

dt = 1e-3

# --- Hookstep flows (NormalMode, no stage cache) ---
G_hook = flow(F_sys,
              RK4(zeros(dim), Flows.NormalMode()),
              TimeStepConstant(dt))

L_hook = flow(couple(F_sys, D_lin),
              RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
              TimeStepConstant(dt))

# --- L-BFGS flows (DiscreteMode + stage cache) ---
G_lbfgs = flow(F_sys,
               RK4(zeros(dim), Flows.NormalMode()),
               TimeStepConstant(dt))

L_lbfgs = flow(TanSys(D_lin),
               RK4(zeros(dim), Flows.DiscreteMode(false)),
               TimeStepFromCache())

L_adj = flow(AdjSys(D_adj),
             RK4(zeros(dim), Flows.DiscreteMode(true)),
             TimeStepFromCache())

# ============================================================
# 8.  Initial guess (perturbed ~5% from the true RPO)
# ============================================================
T_half = T_exact / 2
th1 = w1 * T_half       # phase of oscillator 1 at t = T/2
th2 = w2 * T_half       # phase of oscillator 2 at t = T/2

# Seed 1 (t=0):  true = (1, 0, 1, 0)
# Seed 2 (t=T/2): true = (cos(th1), sin(th1), cos(th2), sin(th2))
z_guess = MVector(
    ([1.05,  0.08,  0.92,  0.12],
     [cos(th1) + 0.05, sin(th1) - 0.03,
      cos(th2) + 0.07, sin(th2) + 0.04]),
    T_exact + 0.3,          # period guess
    s_exact + 0.15)         # shift guess

# Phase-lock wrapper: 3-arg RHS -> 2-arg F(out, x)
F_phase = (out, x) -> F_sys(0, x, out)

# ============================================================
# 9.  Test: Hookstep (trust-region GMRES) -- NS == 2
# ============================================================
@testset "Hookstep RPO convergence (NS=2)   " begin
    z = deepcopy(z_guess)
    status = search!(G_hook, L_hook, S_op, F_phase, dS_op, z,
                     Options(method=:tr_iterative, maxiter=50,
                             e_norm_tol=1e-10, dz_norm_tol=1e-8,
                             gmres_maxiter=15, gmres_rtol=1e-3,
                             tr_radius_init=0.01, verbose=true))

    @test status == :converged
    @test abs(z.d[1] - T_exact) < 1e-7
    @test abs(z.d[2] - s_exact) < 1e-7

    for seg in z.x
        @test abs(sqrt(seg[1]^2 + seg[2]^2) - 1.0) < 1e-7
        @test abs(sqrt(seg[3]^2 + seg[4]^2) - 1.0) < 1e-7
    end

    println("  Hookstep: T = $(z.d[1]), s = $(z.d[2])")
    println("  Error: dT = $(abs(z.d[1] - T_exact)), ds = $(abs(z.d[2] - s_exact))")
end

# ============================================================
# 10. Test: L-BFGS -- NS == 2
# ============================================================
@testset "L-BFGS RPO convergence (NS=2)     " begin
    z = deepcopy(z_guess)
    status = search!(G_lbfgs, L_lbfgs, L_adj, S_op, F_phase, dS_op, z,
                     Options(method=:lbfgs_opt, maxiter=200,
                             e_norm_tol=1e-10, dz_norm_tol=1e-8,
                             lbfgs_memory=320, ls_maxiter=30, verbose=true))

    @test status == :converged
    @test abs(z.d[1] - T_exact) < 1e-7
    @test abs(z.d[2] - s_exact) < 1e-7

    for seg in z.x
        @test abs(sqrt(seg[1]^2 + seg[2]^2) - 1.0) < 1e-7
        @test abs(sqrt(seg[3]^2 + seg[4]^2) - 1.0) < 1e-7
    end

    println("  L-BFGS:  T = $(z.d[1]), s = $(z.d[2])")
    println("  Error: dT = $(abs(z.d[1] - T_exact)), ds = $(abs(z.d[2] - s_exact))")
end

# ============================================================
# 11. End-to-end orbit closure check
# ============================================================
@testset "Orbit closure (NS=2)               " begin
    z = deepcopy(z_guess)
    search!(G_hook, L_hook, S_op, F_phase, dS_op, z,
            Options(method=:tr_iterative, maxiter=50,
                    e_norm_tol=1e-10, dz_norm_tol=1e-8,
                    gmres_maxiter=15, gmres_rtol=1e-3,
                    tr_radius_init=0.01, verbose=false))

    T, s = z.d[1], z.d[2]
    N = nsegments(z)
    dt_seg = T / N

    for i in 1:N
        x0 = deepcopy(z.x[i])
        G = deepcopy(G_hook)
        G(x0, (0.0, dt_seg))
        x0_shifted = deepcopy(x0)
        S_op(x0_shifted, s)
        x_next = z.x[mod1(i, N)]
        @test x0_shifted ≈ x_next atol=1e-6
    end
    println("  All segments close to within 1e-6 after shift and propagation.")
end

println("\nAll RPO spatial shift tests passed.")