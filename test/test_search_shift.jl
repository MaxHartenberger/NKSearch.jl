# ----------------------------------------------------------------- #
# Test: spatial shift (NS=2) — relative periodic orbits              #
# ----------------------------------------------------------------- #
# The Hopf normal form is rotationally symmetric, so the exact
# solution is the unit circle with period 2π and spatial shift 0.
#
# This file verifies that the 7-argument (L-BFGS with adjoint)
# search! signature correctly handles the spatial shift operators
# S, F, dS, and that the residual decreases.

# Spatial shift: rotate state x by angle s (in place)
struct SpatialShift end
function (::SpatialShift)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

# Generator of the spatial shift: d/ds S(x, s) evaluated at s=0
struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    out[1] = -x[2]
    out[2] =  x[1]
    return out
end

# =========================================================================
#  Smoke test: verify the L-BFGS + spatial-shift loop runs and the
#  residual decreases significantly in the first few iterations.
# =========================================================================
@testset "search_lbfgs_shift smoke (NS=2)          " begin
    μ = 1.0
    F_sys = System(μ)
    D     = SystemLinear(μ)
    D_adj = SystemLinearAdjoint(μ)

    S_op  = SpatialShift()
    dS_op = SpatialShiftDerivative()

    F_phase = (out, x) -> F_sys(0, x, out)

    G = flow(F_sys,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(TangentSystem(D),
             RK4(zeros(2), Flows.DiscreteMode(false)),
             TimeStepFromCache())
    adj_flow = flow(AdjointTangentSystem(D_adj),
                    RK4(zeros(2), Flows.DiscreteMode(true)),
                    TimeStepFromCache())

    z = MVector(([1.1, 0.0], [-1.1, 0.0]), 2π, 0.0)

    residuals = Float64[]
    cb = (iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> push!(residuals, e_norm)

    status = search!(G, L, adj_flow, S_op, F_phase, dS_op, z,
                     Options(maxiter=100,
                             dz_norm_tol=0.0,
                             e_norm_tol=0.0,
                             verbose=true,
                             method=:lbfgs_opt,
                             ls_maxiter=20,
                             lbfgs_memory=20,
                             callback=cb))

    @test length(residuals) >= 2                     # at least iter 0 + 1
    @test residuals[end] < residuals[1] * 0.5        # F-norm at least halved
    @test status == :maxiter_reached                 # stopped by maxiter, not crashed
end
