# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot
import GMRES: gmres!
import Flows
import Flows: RAMStageCache

"""
    IterSolCache(Gs, Ls, Ls_adj, S, D, z0)

Matrix-free cache for Newton-Raphson periodic orbit search.

Arguments (user-provided, one per segment):
  - `Gs`:     nonlinear flows  (TimeStepConstant, NormalMode)
  - `Ls`:     forward linearised flows  (TimeStepFromCache, DiscreteMode{false})
  - `Ls_adj`: adjoint flows  (TimeStepFromCache, DiscreteMode{true}), or `nothing`
  - `S`:      spatial shift operator (or `nothing`)
  - `D`:      phase-locking derivative operators
  - `z0`:     initial guess

Allocated:
  - `xT`:           end-of-segment states
  - `tmp`:          temporary storage (one per segment)
  - `z0`:           copy of the current orbit
  - `stage_caches`: stage caches — populated by update!, read by mat-vecs
"""
struct IterSolCache{X, N, NS, GST, LST, LAT, ST, DT, SCT}
       Gs::GST               # nonlinear flows
       Ls::LST               # forward linearised flows (DiscreteMode{false})
    Ls_adj::LAT              # adjoint flows (or nothing)
        S::ST                # space shift operator
        D::DT                # phase-locking derivative operators
       xT::NTuple{N, X}      # end-of-segment states (populated by update!)
      tmp::NTuple{N, X}      # temporary storage (one per segment)
       z0::MVector{X, N, NS} # current orbit
stage_caches::SCT            # stage caches (one per segment)
end

# Main outer constructor
function IterSolCache(Gs, Ls, Ls_adj, S, D, z0::MVector{X, N, NS}) where {X, N, NS}
    nstages = Flows.nstages(Gs[1].meth)
    stage_caches = ntuple(i -> RAMStageCache(nstages, z0[1]), N)
    IterSolCache(Gs, Ls, Ls_adj, S, D,
                 similar.(z0.x),
                 ntuple(i -> similar(z0[1]), N),
                 similar(z0),
                 stage_caches)
end

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::IterSolCache{X}, δz::MVector{X}) where {X} = mul!(similar(δz), mm, δz)

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
               mm::IterSolCache{X, N, NS},
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
function update!(mm::IterSolCache{X, N, NS},
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

# Compute the residual norm according to opts.e_norm_type.
# :euclidean   → ‖b‖ (standard Euclidean over all segments + scalar)
# :max_segment → max_i ‖b.x[i]‖ (worst segment, ignores scalar part)
function _residual_norm(b::MVector, opts::Options)
    if opts.e_norm_type == :max_segment
        return maximum(norm, b.x)
    else
        return norm(b)
    end
end

# solution for iterative method
_solve(x::MV, A::IterSolCache, b::MV, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b; rel_rtol=opts.gmres_rtol,
                     maxiter=opts.gmres_maxiter,
                     verbose=opts.gmres_verbose,
                    callback=opts.gmres_callback)

_solve(x::MV, A::IterSolCache, b::MVector, tr_radius::Real, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b, tr_radius; rel_rtol=opts.gmres_rtol,
                                maxiter=opts.gmres_maxiter,
                                verbose=opts.gmres_verbose,
                               callback=opts.gmres_callback)