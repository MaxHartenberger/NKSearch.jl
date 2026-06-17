# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: nthreads

export search!

# NOTE: multithreading only works reliably (no race conditions) if the number of threads
# NOTE: equals the number of segments used for the multiple-shooting.
# NOTE: See https://julialang.org/blog/2023/07/PSA-dont-use-threadid/ for details
# NOTE: on the faulty pattern that is being used that causes the problems encountered.

# Arguments
# ---------
# G    : nonlinear propagator  - obeys `G(x, (0, T))` where `x` is modified in place
# L    : linearised propagator - obeys `L(Flows.couple(x, y), (0, T))` where `x`
#        and `y` are modified in place
# S    : spatial shift operator - obeys `S(x, s)` where `x` is shifted by `s`
# F    : the right hand side of the governing equations. Obeys `F(out, x)`, where
#        `out` gets overwritten
# dS   : derivative of `S` wrt to `s` - obeys `dS(out, x)` where `out` gets
#        overwritten
# z0   : initial guess vector, gets overwritten
# opts : search options (see src/options.jl)

"""
    search!(G, L, S, F, dS, z0::MVector{X,N,2}, opts=Options()) -> status
    search!(G, L,       F,     z0::MVector{X,N,1}, opts=Options()) -> status

Refine the candidate orbit `z0` in place with a Newton–Krylov / multiple-
shooting iteration until convergence or `opts.maxiter` is reached.

Use the 6-argument form to search for a **relative periodic orbit** (an orbit
closing up to a spatial shift, `z0` has a shift unknown, `NS == 2`), and the
4-argument form for an ordinary **periodic orbit** (`NS == 1`).

`z0` is overwritten with the refined orbit. The return value is a status
symbol such as `:converged`, `:maxiter_reached`, `:min_step_reached`, or
`:callback_satisfied` (line-search method returns `nothing`).

# Arguments
- `G`: nonlinear flow operator. `G(x, (0, T))` advances state `x` in place
  over time span `(0, T)`.
- `L`: linearised flow operator. `L(Flows.couple(x, y), (0, T))` advances the
  base state `x` and the perturbation `y` in place. To avoid writing a
  hand-coded linearisation, pass a [`JFOp`](@ref) built from `G`.
- `S`: spatial shift operator (relative periodic orbits only). `S(x, s)`
  shifts state `x` by `s` in place.
- `F`: right-hand side of the governing ODE. `F(out, x)` overwrites `out`
  with the time derivative at `x`; it sets the phase-locking constraint that
  removes the time-translation degeneracy.
- `dS`: generator of the spatial shift (relative periodic orbits only).
  `dS(out, x)` overwrites `out`, fixing the spatial phase.
- `z0::MVector`: initial guess; overwritten with the result. See
  [`MVector`](@ref).
- `opts::Options`: solver settings; see [`Options`](@ref).

`G` and `L` are deep-copied once per shooting segment, so the same operator
instance can be passed for all segments.

!!! note "Threading"
    The iterative methods (`:ls_iterative`, `:tr_iterative`) parallelise the
    shooting segments across tasks; run Julia with as many threads as there
    are segments. The direct methods (`:ls_direct`, `:tr_direct`) require a
    single thread.

# Example
```julia
using NKSearch, Flows, LinearAlgebra

F = ...   # ODE right-hand side, callable as F(t, x, dxdt)
G = flow(F, RK4(zeros(2)), TimeStepConstant(1e-3))                      # nonlinear flow
L = flow(couple(F, Fjac), RK4(couple(zeros(2), zeros(2))), TimeStepConstant(1e-3))

z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)   # 2-segment guess, period 2π
search!(G, L, (dxdt, x) -> F(0, x, dxdt), z,
        Options(method=:tr_iterative, maxiter=25))
```
See the manual for a complete, runnable tutorial.
"""
search!(G, L, S, F, dS, z0::MVector{X, N, 2}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)),
             ntuple(i->deepcopy(L), nsegments(z0)), S, (F, dS), z0, opts)

# when we do not have shifts
search!(G, L, F, z0::MVector{X, N, 1}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)), 
             ntuple(i->deepcopy(L), nsegments(z0)), nothing, (F, ), z0, opts)

# dispatch to correct method
function _search!(Gs, Ls, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    return (  opts.method == :ls_direct
            ? _search_linesearch!(Gs, Ls, S, D, z0, DirectSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :ls_iterative
            ? _search_linesearch!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :tr_direct
            ? _search_trustregion!(Gs, Ls, S, D, z0, DirectSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :tr_iterative
            ? _search_hookstep!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :lbfgs_opt
            ? _search_lbfgs_opt!(Gs, Ls, S, D, z0, StageIterCache(Gs, Ls, opts.lbfgs_adj_system, S, D, z0), opts)
            : opts.method == :lbfgs_newton_dogleg
            ? _search_lbfgs_dogleg!(Gs, Ls, S, D, z0, StageIterCache(Gs, Ls, opts.lbfgs_adj_system, S, D, z0), opts)
            : throw(ArgumentError("unknown method: $(opts.method)")))
end
