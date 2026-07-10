# ------------------------------------------------------------------ #
# Copyright 2026, Maximilian Hartenberger, University of Southampton #
# ------------------------------------------------------------------ #
import LinearAlgebra: dot
import Flows
import Flows: RAMStageCache
import Base.Threads: @sync, @spawn

export StageIterCache, AdjointIterSolCache

# ~~~ Stage-based Iterative Solver Cache (forward) ~~~
"""
    StageIterCache(Gs, Ls, S, D, z0)

Matrix-free cache for Newton-Raphson periodic orbit search using
pre-computed integration stages (stage caching).

Arguments (user-provided, one per segment):
  - `Gs`:     nonlinear flows  (TimeStepConstant, NormalMode)
  - `Ls`:     forward linearised flows  (TimeStepFromCache, DiscreteMode{false})
  - `S`:      spatial shift operator (or `nothing`)
  - `D`:      phase-locking derivative operators
  - `z0`:     initial guess

Allocated:
  - `xT`:           end-of-segment states
  - `dxTdT`:        time derivative f(φ_i), shifted if NS==2
  - `tmp`:          temporary storage (one per segment)
  - `z0`:           copy of the current orbit
  - `stage_caches`: stage caches — populated by update!, read by mat-vecs
"""
struct StageIterCache{X, N, NS, GST, LST, ST, DT, SCT}
          Gs::GST               # nonlinear flows
          Ls::LST               # forward linearised flows (DiscreteMode{false})
           S::ST                # space shift operator
           D::DT                # phase-locking derivative operators
          xT::NTuple{N, X}      # end-of-segment states (populated by update!)
      dxTdT::NTuple{N, X}      # time derivative f(φ_i), shifted if NS==2
         tmp::NTuple{N, X}      # temporary storage (one per segment)
          z0::MVector{X, N, NS} # current orbit
stage_caches::SCT               # stage caches (one per segment)
end

# Main outer constructor
function StageIterCache(Gs, Ls, S, D, z0::MVector{X, N, NS}) where {X, N, NS}
    nstages = Flows.nstages(Gs[1].meth)
    stage_caches = ntuple(i -> RAMStageCache(nstages, z0[1]), N)
    StageIterCache(Gs, Ls, S, D,
                   similar.(z0.x),
                   similar.(z0.x),
                   ntuple(i -> similar(z0[1]), N),
                   similar(z0),
                   stage_caches)
end

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::StageIterCache{X}, δz::MVector{X}) where {X} = mul!(similar(δz), mm, δz)

# Compute mat-vec product  out = J * δz
#
#   J = ∂F/∂z  where  F_i = z[i+1] - ϕ(z[i], T/N)
#   (J·δz)_i = -Dϕ_i·δz[i] + δz[i+1] - f(xT[i])·δT/N
#
# For relative periodic orbits (NS == 2) there are additional
# spatial-shift contributions on the last segment (see code below).
#
# Uses DiscreteMode{false} (tangent-linear) with cached stages:
# computes Dϕ·v, the algebraic transpose of what DiscreteMode{true}
# (adjoint mode) computes on the same cached stages.
#
# GMRES solves  J·dz = F(z)  →  z_new = z - dz
function mul!(out::MVector{X, N, NS},
               mm::StageIterCache{X, N, NS},
               δz::MVector{X, N, NS}) where {X, N, NS}
    xT    = mm.xT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    z0    = mm.z0
    tmp   = mm.tmp
    sc    = mm.stage_caches

    # compute  -Dϕ_i·δz[i] + δz[i+1]  using cached stages
    @sync for i in 1:N
        @spawn begin
            out[i] .= δz[i]
            Ls[i](out[i], sc[i])          # DiscreteMode{false}
            out[i] .*= -1.0               # -Dϕ_i·δz[i]
            NS == 2 && i == N && S(out[i], z0.d[2])
            out[i] .+= δz[i%N + 1]        # + δz[i+1]
        end
    end

    # period column:  -f(xT[i]) / N · δT  (pre-computed in update!;
    # dxTdT[i] already holds S(f(φ_i), s) for NS==2, matching IterSolCache)
    @sync for i in 1:N
        @spawn begin
            out[i] .-= mm.dxTdT[i] .* (δz.d[1] ./ N)
        end
    end

    NS == 2 && (out[N] .-= D[2](tmp[1], xT[N]) .* δz.d[2])

    # phase-locking constraints
    out.d = ntuple(j -> dot(δz[1], D[j](tmp[1], z0[1])), NS)

    return out
end

# Update the linear operator and rhs arising in the Newton-Raphson iterations
function update!(mm::StageIterCache{X, N, NS},
                  b::MVector{X, N, NS},
                 z0::MVector{X, N, NS}) where {X, N, NS}

    mm.z0 .= z0

    xT = mm.xT
    Gs = mm.Gs
    S  = mm.S
    D  = mm.D
    T  = z0.d[1]
    sc = mm.stage_caches

    @sync for i in 1:N
        @spawn begin
            xT[i] .= z0[i]
            Flows.reset!(sc[i])
            Gs[i](xT[i], (0, T/N), sc[i])   # xT[i] = φ, fills stage caches
            D[1](mm.dxTdT[i], xT[i])         # dxTdT[i] = f(φ)  (exact, before shift)
        end
    end

    NS == 2 && S(   xT[N], z0.d[2])
    NS == 2 && S(mm.dxTdT[N], z0.d[2])       # shift period derivative to match

    # residual  b = F(z)
    @sync for i in 1:N
        @spawn begin
            b[i] .= z0[i%N+1] .- xT[i]
        end
    end
    b.d = zero.(b.d)

    return nothing
end

# =========================================================================== #
# ~~~ Adjoint (transpose) of StageIterCache ~~~
# =========================================================================== #

"""
    AdjointIterSolCache(Ls_adj, D, S, xT, z0, tmp, stage_caches)

Adjoint (transpose) of `StageIterCache`.  Computes `J^T * w` matrix-free.

Shares `xT`, `tmp`, `z0`, `stage_caches` with the forward cache
(`StageIterCache`).  Construct both caches together in the driver
(e.g. `newton.jl`) and pass the adjoint cache explicitly to search
routines.

Fields:
  - `Ls_adj`:       adjoint flows (user-provided, one per segment)
  - `D`:            phase-locking derivative operators (shared)
  - `S`:            spatial shift operator (`nothing` for NS == 1)
  - `xT`:           end-of-segment states (shared, from `update!`)
  - `dxTdT`:        time derivatives f(φ_i) (shared, from `update!`)
  - `z0`:           current orbit (shared)
  - `tmp`:          temporary storage (shared)
  - `stage_caches`: stage caches (one per segment) — bridge forward→adjoint

Supports both ordinary periodic orbits (`NS == 1`) and relative
periodic orbits (`NS == 2`).  For `NS == 2`, the spatial-shift
transpose applies `S(·, -s)` to `w[N]` *before* the adjoint
integration, matching the forward composition `S ∘ Dϕ_N`.
"""
struct AdjointIterSolCache{X, N, NS, LAT, DT, ST, SCT}
    Ls_adj::LAT            # adjoint flows (user-provided, one per segment)
         D::DT             # phase-locking derivative operators
         S::ST             # spatial shift operator (nothing for NS == 1)
        xT::NTuple{N, X}   # end-of-segment states (from fwd update!)
    dxTdT::NTuple{N, X}    # time derivatives f(φ_i) (from fwd update!)
        z0::MVector{X, N, NS}
      tmp::NTuple{N, X}    # shared with fwd cache
stage_caches::SCT          # stage caches (one per segment)
end

# Main interface for AdjointIterSolCache
Base.:*(mm::AdjointIterSolCache{X}, w::MVector{X}) where {X} = mul!(similar(w), mm, w)

# Adjoint mat-vec product: out = J^T * w  (matrix-free)
#
# Integrates the adjoint equations backward through each segment using
# the stage caches populated by the forward `update!`.  The forward
# mat-vec computes:
#   (J·δz)_i = -Dϕ_i·δz[i] + δz[i+1] - f(xT[i])·δT/N
# and the adjoint computes the exact algebraic transpose.
#
# For relative periodic orbits (NS == 2), the forward applies a
# spatial shift S on segment N after the tangent propagation:
#   (J·δz)[N] = S(-Dϕ_N·δz[N], s) + δz[1] - f(xT[N])/N·δT + dS/ds·δs
# The adjoint applies the inverse shift S(·, -s) to w[N] *before*
# the backward adjoint integration (Dϕ_N^T ∘ S^T).
function mul!(out::MVector{X, N, NS},
              mm::AdjointIterSolCache{X, N, NS},
               w::MVector{X, N, NS}) where {X, N, NS}
    Ls_adj       = mm.Ls_adj
    D            = mm.D
    S            = mm.S
    z0           = mm.z0
    stage_caches = mm.stage_caches
    tmp          = mm.tmp

    # Per-segment backward adjoint integrations.
    # For relative periodic orbits (NS == 2), the forward applies
    # S *after* the tangent propagation on segment N (S ∘ Dϕ_N).
    # The adjoint therefore applies S^T = S(·, -s) *before* the
    # adjoint propagation (Dϕ_N^T ∘ S^T).
    @sync for i in 1:N
        @spawn begin
            out[i] .= w[i]
            NS == 2 && i == N && S(out[i], -z0.d[2])
            Ls_adj[i](out[i], stage_caches[i])
            # Subtract w[i-1] — transposed super-diagonal identity block
            i_prev = (i == 1) ? N : i - 1
            out[i] .-= w[i_prev]
        end
    end

    # Period row of J^T:  ∂F/∂T = -f(φ)/N,  transpose is -f^T/N.
    # dxTdT[i] already holds S(f(φ_i), s) (shifted if NS==2),
    # pre-computed by the forward update! — no D[1] call needed here.
    partials = zeros(N)
    @sync for i in 1:N
        @spawn begin
            partials[i] = dot(mm.dxTdT[i], w[i]) / N
        end
    end
    out_d_1 = sum(partials)                   # serial reduction

    # Spatial-shift row of J^T (NS == 2 only).
    # Forward:  out[N] -= dS/ds(xT[N]) · δs  (negated, consistent with rest of J)
    # Adjoint:  out.d[2] = -⟨w[N], dS/ds(xT[N])⟩  (transpose, inherits the negation)
    if NS == 2
        D[2](tmp[N], mm.xT[N])
        out_d_2 = dot(w[N], tmp[N])
    end

    # Negate segments (flip -Dϕ^T+I^T → +Dϕ^T-I^T = J_seg^T)
    # and negate period row (+f^T/N → -f^T/N = J_per^T).
    @sync for i in 1:N
        @spawn begin
            out[i] .*= -1.0
        end
    end

    # Set scalar unknowns
    if NS == 2
        out.d = (-out_d_1, -out_d_2)
    else
        out.d = (-out_d_1,)
    end

    # Phase-locking condition transposed (NOT negated — independent of -Dϕ+I)
    D[1](tmp[1], z0[1])           # f(z0[1]) → tmp[1]
    tmp[1] .*= w.d[1]             # scale in-place
    out[1] .+= tmp[1]             # add to out[1]
    if NS == 2
        D[2](tmp[1], z0[1])       # reuse tmp[1] (value already consumed by out[1] above)
        tmp[1] .*= w.d[2]
        out[1] .+= tmp[1]
    end

    return out
end