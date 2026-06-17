# ------------------------------------------------------------------ #
# Copyright 2026, Maximilian Hartenberger, University of Southampton #
# ------------------------------------------------------------------ #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot, norm
import GMRES: gmres!
import Flows
import Flows: RAMStageCache

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
# Uses DiscreteMode{false} with cached stages — exact transpose
# of the adjoint (DiscreteMode{true}).
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

    # period column:  -f(xT[i]) / N · δT
    for i in 1:N
        D[1](tmp[1], xT[i])
        out[i] .-= tmp[1] .* (δz.d[1] ./ N)
    end

    NS == 2 && (out[N] .+= D[2](tmp[1], xT[N]) .* δz.d[2])

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
    T  = z0.d[1]
    sc = mm.stage_caches

    @sync for i in 1:N
        @spawn begin
            xT[i] .= z0[i]
            Flows.reset!(sc[i])
            Gs[i](xT[i], (0, T/N), sc[i])   # fills stage caches
        end
    end

    NS == 2 && S(xT[N], z0.d[2])

    # residual  b = F(z)
    for i in 1:N
        b[i] .= z0[i%N+1] .- xT[i]
    end
    b.d = zero.(b.d)

    return nothing
end

# solution for iterative method
_solve(x::MV, A::StageIterCache, b::MV, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b; rel_rtol=opts.gmres_rtol,
                     maxiter=opts.gmres_maxiter,
                     verbose=opts.gmres_verbose,
                    callback=opts.gmres_callback)

_solve(x::MV, A::StageIterCache, b::MVector, tr_radius::Real, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b, tr_radius; rel_rtol=opts.gmres_rtol,
                                maxiter=opts.gmres_maxiter,
                                verbose=opts.gmres_verbose,
                               callback=opts.gmres_callback)


# =========================================================================== #
# ~~~ Adjoint (transpose) of StageIterCache ~~~
# =========================================================================== #

"""
    AdjointIterSolCache(Ls_adj, D, xT, z0, tmp, stage_caches)

Adjoint (transpose) of `StageIterCache`.  Computes `J^T * w` matrix-free.

Shares `xT`, `tmp`, `z0`, `stage_caches` with the forward cache
(`StageIterCache`).  Construct both caches together in the driver
(e.g. `newton.jl`) and pass the adjoint cache explicitly to search
routines.

Fields:
  - `Ls_adj`:       adjoint flows (user-provided, one per segment)
  - `D`:            phase-locking derivative operators (shared)
  - `xT`:           end-of-segment states (shared, from `update!`)
  - `z0`:           current orbit (shared)
  - `tmp`:          temporary storage (shared)
  - `stage_caches`: stage caches (one per segment) — bridge forward→adjoint

Note: the spatial shift transpose (`NS == 2`) is not yet implemented.
"""
struct AdjointIterSolCache{X, N, NS, LAT, DT, SCT}
    Ls_adj::LAT            # adjoint flows (user-provided, one per segment)
         D::DT             # phase-locking derivative operators
        xT::NTuple{N, X}   # end-of-segment states (from fwd update!)
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
# Thread safety:  @sync @spawn parallelises the per-segment backward
# integrations.  Each segment integration is independent (different
# `out[i]`, `w[i]`, `stage_caches[i]`), so no data races.  The scalar
# accumulation `out_d_1` is computed sequentially after the @sync barrier.
function mul!(out::MVector{X, N, NS},
              mm::AdjointIterSolCache{X, N, NS},
               w::MVector{X, N, NS}) where {X, N, NS}
    Ls_adj       = mm.Ls_adj
    D            = mm.D
    z0           = mm.z0
    stage_caches = mm.stage_caches
    tmp          = mm.tmp

    # Per-segment backward adjoint integrations (thread-safe: independent segments)
    @sync for i in 1:N
        @spawn begin
            out[i] .= w[i]
            Ls_adj[i](out[i], stage_caches[i])
            # Subtract w[i-1] — transposed super-diagonal identity block
            i_prev = (i == 1) ? N : i - 1
            out[i] .-= w[i_prev]
        end
    end

    # Period row of J^T:  ∂F/∂T = -f(φ(u,T/N)) / N,  transpose is -f^T/N
    out_d_1 = 0.0
    for i in 1:N
        D[1](tmp[1], mm.xT[i])  # f(xT[i]) → tmp[1]
        tmp[1] .*= (1.0 / N)     # scaled in-place to avoid FTField broadcast issues
        out_d_1 += dot(tmp[1], w[i])
    end

    # Negate segments (flip -Dϕ^T+I^T → +Dϕ^T-I^T = J_seg^T)
    # and negate period row (-f^T/N → +f^T/N = J_per^T).
    for i in 1:N
        out[i] .*= -1.0
    end
    out.d = (-out_d_1,)

    # Phase-locking condition transposed (NOT negated — independent of -Dϕ+I)
    D[1](tmp[1], z0[1])           # f(z0[1]) → tmp[1]
    tmp[1] .*= w.d[1]             # scale in-place
    out[1] .+= tmp[1]             # add to out[1]
    if NS == 2
        D[2](tmp[2], z0[1])       # need second temp for second shift component
        tmp[2] .*= w.d[2]
        out[1] .+= tmp[2]
    end

    return out
end
