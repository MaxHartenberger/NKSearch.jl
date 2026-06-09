# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~
@with_kw struct Options{GT, W, CB, LA}
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
    callback::CB            = (iter, z, Fz, f_norm, ∇ϕ_norm, λ) -> false

    # line search parameters
    ls_method::Symbol      = :backtracking        # :armijo, :strong_wolfe, :weak_wolfe, :nonmonotone, :goldstein, :interp, :filter, :safeguarded, :backtracking
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

    lbfgs_memory::Int       = 10                   # number of history vectors to store for optimization L-BFGS
    lbfgs_adj_system::LA    = nothing              # adjoint linear system(s) for L-BFGS gradient (J^T action)

    @assert method in (:tr_direct, :ls_direct, :ls_iterative, :tr_iterative, :lbfgs_opt, :lbfgs_newton_dogleg)
    @assert skipiter > 0
    @assert fd_order in (1, 2)
    @assert ls_method in (:armijo, :strong_wolfe, :weak_wolfe, :nonmonotone, :goldstein, :interp, :filter, :safeguarded, :backtracking)
end

