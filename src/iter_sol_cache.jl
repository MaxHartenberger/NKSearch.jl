# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot
import GMRES: gmres!
import Flows

# ~~~ Matrix Type ~~~
struct IterSolCache{X, N, NS, M, GST, LST, ST, DT, MT}
       Gs::GST               # flow operator(s)
       Ls::LST               # linearised flow operator (s)
        S::ST                # space shift operator
        D::DT                # time (and space) derivative operators
       xT::NTuple{N, X}      # time shifted conditions
    dxTdT::NTuple{N, X}      # time derivative of flow operator
      tmp::NTuple{M, X}      # temporary storage
       z0::MVector{X, N, NS} # current orbit
     mons::MT                # monitor
     opts::Options           # options
 phase_ref::X                # frozen reference state u₁⁽⁰⁾ for phase conditions
end

# Main outer constructor
function IterSolCache(Gs, Ls, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    mon_type = opts.fd_order == 1 ? Flows.StoreNFromLast{0} : Flows.StoreNFromLast{2}
    ntmps = opts.fd_order == 1 ? nsegments(z0) : 2*nsegments(z0)
    # Freeze u₁ as the phase-condition reference (constant throughout optimization)
    phase_ref = deepcopy(z0[1])
    IterSolCache(Gs, Ls, S, D,
                 similar.(z0.x),
                 similar.(z0.x),
                 ntuple(i->similar(z0[1]), ntmps),
                 similar(z0),
                 ntuple(i->mon_type(z0[1]), nsegments(z0)),
                 opts,
                 phase_ref)
end

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::IterSolCache{X}, δz::MVector{X}) where {X} = mul!(similar(δz), mm, δz)

# Compute mat-vec product
function mul!(out::MVector{X, N, NS},
               mm::IterSolCache{X, N, NS},
               δz::MVector{X, N, NS}) where {X, N, NS}
    # aliases
    xT    = mm.xT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    z0    = mm.z0
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    T     = mm.z0.d[1]

    # comput L{x0[i]}-δz[i] - δz[i+1]
    @sync for i in 1:N
        @spawn begin
            # set perturbation initial condition
            out[i] .= δz[i]

            # set nonlinear initial condition
            tmp[i] .= z0[i]

            # propagate by T/N
            Ls[i](Flows.couple(tmp[i], out[i]), (0, T/N))

            # apply shift on last segment (if we have one)
            NS == 2 && i == N && S(out[i], z0.d[2])

            # this is the identity operators on the upper diagonal
            out[i] .-= δz[i%N + 1]
        end
    end

    # period derivative
    for i = 1:N
        out[i] .+= dxTdT[i].*(δz.d[1]./N)
    end

    # shift derivative (if present) goes only on last element
    NS == 2 && (out[N] .+= D[2](tmp[1], xT[N]).*δz.d[2])

    # add phase locking constraints — use frozen reference u₁⁽⁰⁾
    D[1](tmp[1], mm.phase_ref)          # f(u_ref) → tmp[1]
    out_d1 = dot(δz[1], tmp[1])
    if NS == 2
        D[2](tmp[1], mm.phase_ref)      # ∂_s S(u_ref, 0) → tmp[1]
        out_d2 = dot(δz[1], tmp[1])
        out.d = (out_d1, out_d2)
    else
        out.d = (out_d1,)
    end

    return out
end

# Update the linear operator and rhs arising in the Newton-Raphson iterations
function update!(mm::IterSolCache{X, N, NS},
                  b::MVector{X, N, NS},
                 z0::MVector{X, N, NS},
               opts::Options) where {X, N, NS}

    # store this vector for the products
    mm.z0 .= z0

    # aliases
    xT    = mm.xT
    Gs    = mm.Gs
    S     = mm.S
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    ϵ     = opts.ϵ
    T     = z0.d[1]
    mons  = mm.mons

    @sync for i in 1:N
        @spawn begin
            # set and propagate
            xT[i] .= z0[i]
            Gs[i](xT[i], (0, T/N), mons[i])

            # finite difference derivative of flow operator
            # see https://epubs.siam.org/doi/10.1137/070705623 page 27
            tmp[i] .= mons[i].x;
            opts.fd_order == 2 && (tmp[N+i] .= mons[i].x)
            Gs[i](tmp[i], (mons[i].t, T/N + ϵ))
            opts.fd_order == 2 && Gs[i](tmp[N+i], (mons[i].t, T/N - ϵ))
            if opts.fd_order == 2
                dxTdT[i] .= (tmp[i] .- tmp[N+i])./(2*ϵ)
            else
                dxTdT[i] .= (tmp[i] .- mons[i].x)./ϵ
            end
        end
    end

    # last one (may) get shifted
    NS == 2 && S(   xT[N], z0.d[2])
    NS == 2 && S(dxTdT[N], z0.d[2])

    # ~~ RIGHT HAND SIDE ~~
    # calculate negative error
    for i = 1:N
        b[i] .= z0[i%N+1] .- xT[i]
    end

    # reset shifts — Newton methods constrain gauge via the phase *rows* of J,
    # not via the phase residual values.  Keeping F.d = 0 ensures the phase rows
    # constrain the step direction without fighting the continuity equations.
    b.d = zero.(b.d)

    return nothing
end

# solution for iterative method. Return only (solution, residual norm) so the
# contract matches the direct solver used by the line-search driver.
function _solve(x::MV, A::IterSolCache, b::MV, opts::Options) where {MV<:MVector}
    x, res_err_norm, _ = gmres!(x, A, b; rel_rtol=opts.gmres_rtol,
                                          maxiter=opts.gmres_maxiter,
                                          verbose=opts.gmres_verbose,
                                         callback=opts.gmres_callback)
    return x, res_err_norm
end

_solve(x::MV, A::IterSolCache, b::MVector, tr_radius::Real, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b, tr_radius; rel_rtol=opts.gmres_rtol,
                                maxiter=opts.gmres_maxiter,
                                verbose=opts.gmres_verbose,
                               callback=opts.gmres_callback)
