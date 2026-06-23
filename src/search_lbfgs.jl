# ------------------------------------------------------------------ #
# Copyright 2026, Maximilian Hartenberger, University of Southampton #
# ------------------------------------------------------------------ #
import LinearAlgebra: norm, dot
import Flows

export OptLBFGSCache

# ~~~ L-BFGS Optimization Cache ~~~
# Minimises the scalar objective ϕ(z) = 0.5 * ||F(z)||^2
mutable struct OptLBFGSCache{X, N, NS}
    s_history::Vector{MVector{X, N, NS}}  # step vectors s_k = z_{k+1} - z_k
    y_history::Vector{MVector{X, N, NS}}  # gradient change y_k = ∇ϕ_{k+1} - ∇ϕ_k
    
    s_scratch::MVector{X, N, NS}          # temporary s_k
    y_scratch::MVector{X, N, NS}          # temporary y_k
    
    Fz_curr::MVector{X, N, NS}            # current residual F(z)
    ∇ϕ_curr::MVector{X, N, NS}            # current gradient J(z)^T F(z)
    ∇ϕ_prev::MVector{X, N, NS}            # previous gradient
    z_prev::MVector{X, N, NS}             # previous z
    
    n_hist::Int                           # current history length
    max_hist::Int                         # maximum history length
    k::Int                                # iteration
    last_idx::Int                         # buffer index of most recent entry (1-based, 0 when empty)
    
    α::Vector{Float64}                    # two-loop recursion temporaries
    direction::MVector{X, N, NS}          # search direction
end

function OptLBFGSCache(z0::MVector{X, N, NS}, opts) where {X, N, NS}
    m = opts.lbfgs_memory
    return OptLBFGSCache(
        [similar(z0) for _ in 1:m], # s_history
        [similar(z0) for _ in 1:m], # y_history
        similar(z0),                # s_scratch
        similar(z0),                # y_scratch
        similar(z0),                # Fz_curr
        similar(z0),                # ∇ϕ_curr
        similar(z0),                # ∇ϕ_prev
        similar(z0),                # z_prev
        0,                          # n_hist
        m,                          # max_hist
        0,                          # k
        0,                          # last_idx
        zeros(m),                   # α
        similar(z0)                 # direction
    )
end

"""
    compute_gradient!(∇ϕ, Fz, z, fwd_cache, adj_cache, opts)

Compute the residual `Fz = F(z)` and the gradient `∇ϕ = J(z)^T F(z)`.

Populates the forward cache's stage caches via `update!`, then applies
the adjoint operator to `Fz` to obtain the gradient of `ϕ(z) = ½‖F(z)‖²`.
"""
function compute_gradient!(∇ϕ::MVector{X, N, NS},
                           Fz::MVector{X, N, NS},
                           z::MVector{X, N, NS},
                           fwd_cache::StageIterCache,
                           adj_cache::AdjointIterSolCache,
                           opts::Options) where {X, N, NS}
    # 1. Compute residual F(z). This also populates the base states and stage caches.
    update!(fwd_cache, Fz, z)
    
    # Check if objective is small enough, avoid gradient mapping if unnecessary
    if _residual_norm(Fz, opts.e_norm_type) < opts.e_norm_tol
        return ∇ϕ, Fz
    end

    # 2. Compute ∇ϕ = J^T * F(z) using exact algebraic transpose.
    mul!(∇ϕ, adj_cache, Fz)
    
    return ∇ϕ, Fz
end

# ~~~ L-BFGS two-loop recursion ~~~
# Computes dz = -H_k * ∇ϕ ≈ -(J^T J)^{-1} ∇ϕ
# Iterates the circular buffer in *chronological* order (via last_idx),
# not in linear index order, to handle buffer wrapping correctly.
function lbfgs_two_loop_recursion!(dz::MVector{X, N, NS},
                                   ∇ϕ::MVector{X, N, NS},
                                   cache::OptLBFGSCache) where {X, N, NS}
    n_hist = cache.n_hist
    m = cache.max_hist
    last_idx = cache.last_idx
    α = cache.α
    
    # q starts as -∇ϕ
    dz .= ∇ϕ
    dz .*= -1.0

    # First loop (backward): most recent → oldest, chronologically
    for offset in 0:n_hist-1
        i = mod(last_idx - 1 - offset, m) + 1   # chronological index
        s_i = cache.s_history[i]
        y_i = cache.y_history[i]
        ρ_i = 1.0 / dot(y_i, s_i)
        α[offset + 1] = ρ_i * dot(s_i, dz)
        dz .-= α[offset + 1] .* y_i
    end

    # Initial scaling H0 = γ * I, using the most recent (s,y) pair
    if n_hist > 0
        s_last = cache.s_history[last_idx]
        y_last = cache.y_history[last_idx]
        γ = dot(s_last, y_last) / dot(y_last, y_last)
    else
        γ = 1.0
    end
    dz .*= γ

    # Second loop (forward): oldest → most recent, chronologically
    for offset in n_hist-1:-1:0
        i = mod(last_idx - 1 - offset, m) + 1   # chronological index
        s_i = cache.s_history[i]
        y_i = cache.y_history[i]
        ρ_i = 1.0 / dot(y_i, s_i)
        β = ρ_i * dot(y_i, dz)
        dz .+= (α[offset + 1] - β) .* s_i
    end

    return dz
end

function update_lbfgs_opt_history!(cache::OptLBFGSCache,
                                   z::MVector{X, N, NS},
                                   z_prev::MVector{X, N, NS},
                                   ∇ϕ::MVector{X, N, NS},
                                   ∇ϕ_prev::MVector{X, N, NS}) where {X, N, NS}
    m = cache.max_hist
    n_hist = cache.n_hist

    # s_k = z_{k+1} - z_k
    cache.s_scratch .= z
    cache.s_scratch .-= z_prev

    # y_k = ∇ϕ_{k+1} - ∇ϕ_k
    cache.y_scratch .= ∇ϕ
    cache.y_scratch .-= ∇ϕ_prev

    if norm(cache.s_scratch) < 1e-14 || norm(cache.y_scratch) < 1e-14
        return
    end

    sy = dot(cache.s_scratch, cache.y_scratch)
    if sy <= 0
        return
    end

    idx = (cache.k - 1) % m + 1
    cache.s_history[idx] .= cache.s_scratch
    cache.y_history[idx] .= cache.y_scratch
    cache.last_idx = idx                   # track most recent entry for chronological two-loop

    if n_hist < m
        cache.n_hist = n_hist + 1
    end
    return
end

function _e_norm_opt(fwd_cache, z::MVector{X, N, NS}, dz::MVector{X, N, NS}, λ::Real, Fz::MVector{X, N, NS}, opts) where {X, N, NS}
    # Evaluate F(z + λ*dz) without updating the base state representations
    z .+= λ .* dz
    update!(fwd_cache, Fz, z)
    val = norm(Fz)^2
    z .-= λ .* dz
    return val
end

function safe_e_norm_opt(fwd_cache, z, dz, λ, Fz, opts)
    try
        return true, _e_norm_opt(fwd_cache, z, dz, λ, Fz, opts)
    catch err
        if isa(err, Flows.InvalidSpanError) ||
           isa(err, Base.TaskFailedException) ||
           isa(err, Base.CompositeException) ||
           isa(err, InexactError) ||
           isa(err, ArgumentError)
            return false, Inf
        end
        rethrow(err)
    end
end

"""
    linesearch_opt_lbfgs(fwd_cache, z, dz, Fz, opts) -> (λ, val)

Backtracking line search on the objective ϕ(z) = ½‖F(z)‖².

Starting from λ = 1, evaluates ϕ(z + λ·dz) and accepts the first step
that reduces ϕ.  Reduces λ by `opts.ls_rho` on each rejection.
Returns `(λ, ϕ(z + λ·dz))`.
"""
function linesearch_opt_lbfgs(fwd_cache, z, dz, Fz, opts)
    ok_0, val_0 = safe_e_norm_opt(fwd_cache, z, dz, 0.0, Fz, opts)
    ok_0 || (val_0 = Inf)
    λ = 1.0
    val_λ = λ * val_0

    for iter = 1:opts.ls_maxiter
        ok_λ, val_λ = safe_e_norm_opt(fwd_cache, z, dz, λ, Fz, opts)
        ok_λ || (λ *= opts.ls_rho; continue)

        val_λ < val_0 && return λ, val_λ
        λ *= opts.ls_rho
    end

    return 1.0, val_0
end

"""
    _search_lbfgs_opt!(Gs, Ls, S, D, z0, fwd_cache, adj_cache, opts)

Minimise ϕ(z) = ½‖F(z)‖² using L-BFGS.

The adjoint cache `adj_cache` must be constructed externally (in
`newton.jl`) from the forward cache's shared arrays and the user-
provided adjoint flows.  See `AdjointIterSolCache`.
"""
function _search_lbfgs_opt!(Gs, Ls, S, D, z0, fwd_cache, adj_cache, opts)
    if opts.verbose
        display_header_lbfgs(opts.io, z0)
    end

    opt_cache = OptLBFGSCache(z0, opts)

    # Initial gradient and residual
    compute_gradient!(opt_cache.∇ϕ_curr, opt_cache.Fz_curr, z0, fwd_cache, adj_cache, opts)
    e_norm = _residual_norm(opt_cache.Fz_curr, opts.e_norm_type)   # ‖F(z)‖
    ∇ϕ_norm = norm(opt_cache.∇ϕ_curr)

    # Callback at iteration 0 (λ = 0.0 since no step has been taken)
    opts.callback(0, z0, opt_cache.Fz_curr, e_norm, ∇ϕ_norm, 0.0)

    opts.verbose && display_status_lbfgs(opts.io, 0, "lbfgs", ∇ϕ_norm, e_norm, 0.0)

    # Exit early if initial guess is already converged
    e_norm < opts.e_norm_tol && return nothing

    for iter = 1:opts.maxiter
        # dz = -H_k * ∇ϕ
        lbfgs_two_loop_recursion!(opt_cache.direction, opt_cache.∇ϕ_curr, opt_cache)

        # Store prev state and gradient
        opt_cache.z_prev .= z0
        opt_cache.∇ϕ_prev .= opt_cache.∇ϕ_curr

        # Line search on ϕ(z) = 0.5 * ||F(z)||^2
        λ, _ = linesearch_opt_lbfgs(fwd_cache, z0, opt_cache.direction, opt_cache.Fz_curr, opts)

        # Apply step
        z0 .+= λ .* opt_cache.direction
        dz_norm = norm(opt_cache.direction) * λ

        # Re-evaluate Fz and ∇ϕ
        compute_gradient!(opt_cache.∇ϕ_curr, opt_cache.Fz_curr, z0, fwd_cache, adj_cache, opts)
        e_norm = _residual_norm(opt_cache.Fz_curr, opts.e_norm_type)   # ‖F(z)‖
        ∇ϕ_norm = norm(opt_cache.∇ϕ_curr)

        # Callback after iteration
        opts.callback(iter, z0, opt_cache.Fz_curr, e_norm, ∇ϕ_norm, λ)

        # Convergence check on the freshly computed residual
        e_norm < opts.e_norm_tol && break

        # Update quasi-Newton history
        opt_cache.k += 1
        update_lbfgs_opt_history!(opt_cache, z0, opt_cache.z_prev, opt_cache.∇ϕ_curr, opt_cache.∇ϕ_prev)

        if opts.verbose && iter % opts.skipiter == 0
            display_status_lbfgs(opts.io, iter, "lbfgs", ∇ϕ_norm, e_norm, λ)
        end

        dz_norm < opts.dz_norm_tol && break
    end

    return nothing
end
