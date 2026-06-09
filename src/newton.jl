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
            ? _search_linesearch!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, nothing, S, D, z0), opts)
            : opts.method == :ls_iterative
            ? _search_linesearch!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, nothing, S, D, z0), opts)
            : opts.method == :tr_direct
            ? _search_trustregion!(Gs, Ls, S, D, z0, DirectSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :tr_iterative
            ? _search_hookstep!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, nothing, S, D, z0), opts)
            : opts.method == :lbfgs_opt
            ? _search_lbfgs_opt!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, opts.lbfgs_adj_system, S, D, z0), opts)
            : opts.method == :tr_lbfgs
            ? _search_lbfgs_dogleg!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, opts.lbfgs_adj_system, S, D, z0), opts)
            : throw(ArgumentError("unknown method: $(opts.method)")))
end

