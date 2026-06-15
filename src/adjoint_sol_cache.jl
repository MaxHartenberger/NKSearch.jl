# ------------------------------------------------------------------ #
# Copyright 2026, Maximilian Hartenberger, University of Southampton #
# ------------------------------------------------------------------ #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot
import Flows

export AdjointIterSolCache

"""
    AdjointIterSolCache

Adjoint (transpose) of `IterSolCache`.  Computes `J^T * w` matrix-free.

Shares `xT`, `tmp`, `z0` with the forward cache.  Owns `stage_caches`
(stage storage) which are populated by `update!` and read by the
adjoint integration in `_mul_adj!`.

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

function Base.adjoint(mm::IterSolCache{X, N, NS}) where {X, N, NS}
    mm.Ls_adj === nothing && error(
        "Ls_adj is nothing. Pass the adjoint flows as the third argument " *
        "to IterSolCache(Gs, Ls, Ls_adj, ...) to use L-BFGS optimisation.")
    return AdjointIterSolCache(mm.Ls_adj, mm.D, mm.xT,
                               mm.z0, mm.tmp, mm.stage_caches)
end

# Adjoint mat-vec product: out = J^T * w  (matrix-free)
function mul!(out::MVector{X, N, NS},
              mm::AdjointIterSolCache{X, N, NS},
               w::MVector{X, N, NS}) where {X, N, NS}
    _mul_adj!(out, mm, w)
    return out
end

# Main interface for AdjointIterSolCache
Base.:*(mm::AdjointIterSolCache{X}, w::MVector{X}) where {X} = mul!(similar(w), mm, w)

function _mul_adj!(out::MVector{X, N, NS},
               mm::AdjointIterSolCache{X, N, NS},
                w::MVector{X, N, NS}) where {X, N, NS}
    Ls_adj       = mm.Ls_adj
    D            = mm.D
    z0           = mm.z0
    stage_caches = mm.stage_caches
    tmp          = mm.tmp

    @sync for i in 1:N
        @spawn begin
            # Integrate adjoint backward through segment i using cached stages
            out[i] .= w[i]
            Ls_adj[i](out[i], stage_caches[i])
            # Subtract w[i-1] — transposed super-diagonal identity block
            i_prev = (i == 1) ? N : i - 1
            out[i] .-= w[i_prev]
        end
    end

    # Period row of J^T:  ∂F/∂T = -f(φ(u,T/N)) / N,  transpose is -f^T/N
    # Transpose of the forward mat-vec period column (-f/N).
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
