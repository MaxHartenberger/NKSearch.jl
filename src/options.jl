# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~
"""
    Options(; kwargs...)

Configuration for [`search!`](@ref). All fields are keyword arguments with
defaults; only override what you need.

# Key options
- `method::Symbol = :ls_direct`: globalization + linear-solve strategy, one
  of `:ls_direct`, `:ls_iterative`, `:tr_direct`, `:tr_iterative`. The
  `ls_` variants use a line search, the `tr_` variants a trust region
  (dogleg for `:tr_direct`, hookstep for `:tr_iterative`). The `_direct`
  variants assemble and LU-factorise the Jacobian; the `_iterative`
  variants solve it matrix-free with GMRES. See the manual for guidance.
- `maxiter::Int = 10`: maximum number of Newton iterations.
- `e_norm_tol::Float64 = 1e-10`: convergence tolerance on the residual norm.
- `dz_norm_tol::Float64 = 1e-10`: convergence tolerance on the Newton step norm.
- `ϵ::Float64 = 1e-6`: step used for the finite-difference approximation of
  the time derivative of the flow operator.
- `fd_order::Int = 2`: order (1 or 2) of that finite-difference scheme.
- `verbose::Bool = true`, `io = stdout`, `skipiter::Int = 1`: control status
  printing (`io` receives a table, every `skipiter` iterations).
- `callback = (iter, z) -> false`: called after each iteration; returning
  `true` terminates the search.

# Line-search options
- `ls_maxiter::Int = 10`, `ls_rho::Float64 = 0.5`: maximum backtracking
  iterations and step-reduction factor.

# GMRES options (iterative methods)
- `gmres_maxiter::Int = 10`, `gmres_rtol::Float64 = 1e-3`,
  `gmres_verbose::Bool = true`, `gmres_callback = nothing`,
  `gmres_start = dz -> (dz .*= 0; dz)`: GMRES iteration count, relative
  tolerance, verbosity, callback, and warm-start initialiser.

# Trust-region options (`tr_` methods)
- `tr_radius_init::Float64 = 1`, `tr_radius_max::Float64 = 1e8`: initial and
  maximum trust-region radius.
- `min_step::Float64 = 1e-4`: minimum accepted step before stopping.
- `NR_lim::Float64 = 1e-8`: residual level below which a full Newton step is
  taken regardless of the trust-region test.
- `α::Float64 = 1`, `eta::Float64 = 0.0`: over-relaxation factor and the
  minimum reduction ratio for accepting a step.

# Example
```julia
opts = Options(method=:tr_iterative, maxiter=25,
               e_norm_tol=1e-12, gmres_maxiter=5, verbose=false)
```
"""
@with_kw struct Options{GT, W, CB}
    # generic parameters
    method::Symbol          = :ls_direct           # search method
    maxiter::Int            = 10                   # maximum newton iteration number
    io                      = stdout               # where to print stuff
    skipiter::Int           = 1                    # skip iteration between displays
    verbose::Bool           = true                 # print iteration status
    dz_norm_tol::Float64    = 1e-10                # tolerance on correction
    e_norm_tol::Float64     = 1e-10                # tolerance on residual
    e_norm_type::Symbol     = :euclidean           # :euclidean or :max_segment
    fd_order::Int           = 2                    # use forward or central difference scheme
                                                   # to approximate the derivative of the flow
                                                   # operator
    ϵ::Float64              = 1e-6                 # dt for finite difference approximation
                                                   # of the derivative of the flow operator
    callback::CB            = (iter, z, Fz, f_norm, ∇ϕ_norm, λ, T) -> false

    # line search parameters
    ls_method::Symbol       = :backtracking        # line search method
    ls_maxiter::Int         = 10                   # maximum number of line search iterations
    ls_rho::Float64         = 0.5                  # line search step reduction factor

    # GMRES parameters
    gmres_maxiter::Int      = 10                   # maximum number of GMRES iterations
    gmres_verbose::Bool     = true                 # print GMRES iteration status
    gmres_rtol::Float64     = 1e-3                 # GMRES relative stopping tolerance
    gmres_callback::GT      = nothing              # GMRES callback function
    gmres_start::W          = dz->(dz .*= 0.0; dz) # GMRES warm start based on previous Newton step

    # trust_region algorithm parameters
    min_step::Float64       = 1e-4                 # minimum step tolerance
    α::Float64              = 1                    # over-relaxation factor for trust region update
    NR_lim::Float64         = 1e-8                 # maximum limit for newton region update
    tr_radius_init::Float64 = 1                    # initial trust region radius
    tr_radius_max::Float64  = 10^8                 # maximum trust region radius
    eta::Float64            = 0.00                 # maximum trust region radius

    # L-BFGS parameters
    lbfgs_memory::Int       = 10                   # number of history vectors for L-BFGS

    @assert method in (:tr_direct, :ls_direct, :ls_iterative, :tr_iterative, :lbfgs_opt)
    @assert skipiter > 0
    @assert fd_order in (1, 2)
    @assert ls_method in (:backtracking,)
end
