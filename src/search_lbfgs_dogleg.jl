# ----------------------------------------------------------------- #
# Copyright 2026, Maximilian Hartenberger, University of Southampton #
# ----------------------------------------------------------------- #
# Trust-region dogleg method with L-BFGS point (instead of Cauchy).
# Combines the global convergence of trust-region methods with the
# curvature-aware quasi-Newton direction from L-BFGS.
#
# Subproblem:  min  m(dz) = ½‖F‖² + (J^T F)^T dz + ½ dz^T J^T J dz
#              s.t. ‖dz‖ ≤ Δ
#
# Solved via dogleg between d_lbfgs and d_newton.
# Fully matrix-free — no Jacobian assembly.
# ----------------------------------------------------------------- #
using Printf
import LinearAlgebra: norm, dot

# ~~~ Main search function ~~~
function _search_lbfgs_dogleg!(Gs, Ls, S, D, z0, fwd_cache, opts)
    # display header (reuse trust-region format)
    opts.verbose && display_header_tr(opts.io, z0)

    # set up adjoint and L-BFGS caches
    adj_cache = Base.adjoint(fwd_cache)
    opt_cache = OptLBFGSCache(z0, opts)

    # aliases into the caches
    Fz  = opt_cache.Fz_curr
    ∇φ  = opt_cache.∇ϕ_curr   # will hold J^T·F

    # allocate working vectors
    b         = similar(z0)        # right-hand side (-F) for GMRES
    dz        = similar(z0)        # chosen step
    d_lbfgs   = similar(z0)        # L-BFGS direction
    d_newton  = similar(z0)        # Newton direction (from GMRES)

    # --- initial residual and gradient ---
    compute_gradient!(∇φ, Fz, z0, fwd_cache, adj_cache, opts)
    f_norm = norm(Fz)                        # ‖F(z)‖

    # Callback at iteration 0 (λ = 0.0 since no step has been taken)
    opts.callback(0, z0, Fz, f_norm, 0.0, 0.0)

    # initial display
    Δ = opts.tr_radius_init
    opts.verbose && display_status_tr(opts.io, 0, :start, 0.0, f_norm, 0.0, Δ)

    # --- main iteration loop ---
    for iter = 1:opts.maxiter
        _residual_norm(Fz, opts) < opts.e_norm_tol && break

        # ----- 1.  Compute L-BFGS direction (matrix-free two-loop) -----
        lbfgs_two_loop_recursion!(d_lbfgs, ∇φ, opt_cache)

        # ----- 2.  Compute Newton direction via GMRES (matrix-free) -----
        update!(fwd_cache, b, z0)
        d_newton .*= 0.0
        d_newton, gmres_res, gmres_it = _solve(d_newton, fwd_cache, b, opts)

        # ----- 3.  Dogleg: blend L-BFGS and Newton within trust region -----
        flag = _dogleg_lbfgs!(dz, d_lbfgs, d_newton, Δ)

        # ----- 4.  Predicted reduction  m(0) - m(dz) -----
        Jdz = fwd_cache * dz
        pred = norm(Jdz)^2
        pred < 1e-16 && (pred = 1e-16)

        # ----- 5.  Actual reduction -----
        f_norm_curr = f_norm

        # save pre-step state for L-BFGS history update
        opt_cache.z_prev .= z0
        opt_cache.∇ϕ_prev .= ∇φ

        z0 .-= dz
        compute_gradient!(∇φ, Fz, z0, fwd_cache, adj_cache, opts)
        f_norm_next = norm(Fz)
        actual = f_norm_curr^2 - f_norm_next^2
        ρ = actual / pred

        # ----- 6.  Trust-region update -----
        dz_norm = norm(dz)
        hits_boundary = (flag != :newton)

        if ρ < 0.25
            Δ *= 0.25
            z0 .+= dz  # undo the tentative z0 .-= dz
            update!(fwd_cache, b, z0)
            compute_gradient!(∇φ, Fz, z0, fwd_cache, adj_cache, opts)
            f_norm = norm(Fz)
        else
            f_norm = f_norm_next
            opt_cache.k += 1
            update_lbfgs_opt_history!(opt_cache, z0, opt_cache.z_prev, ∇φ, opt_cache.∇ϕ_prev)

            if ρ > 0.75 && hits_boundary
                Δ = min(2 * Δ, opts.tr_radius_max)
            end
        end

        # Callback after iteration
        opts.callback(iter, z0, Fz, f_norm, norm(∇φ), 1.0)

        # ----- 7.  Display -----
        if opts.verbose && iter % opts.skipiter == 0
            display_status_tr(opts.io, iter, flag, dz_norm, f_norm, ρ, Δ)
        end

        dz_norm < opts.dz_norm_tol && break
    end

    return nothing
end


# ~~~ Dogleg with L-BFGS point ~~~
# Path:  0 → d_lbfgs → d_newton
# Returns the point on this path that lies on the trust-region boundary,
# or the Newton point if it lies inside the region.
function _dogleg_lbfgs!(dz, d_lbfgs, d_newton, Δ)
    nN = norm(d_newton)

    # Case 1: Newton step is inside trust region → take full Newton
    if nN < Δ
        dz .= d_newton
        return :newton
    end

    nL = norm(d_lbfgs)

    # Case 2: L-BFGS point is outside → scale it down to boundary
    if nL > Δ
        dz .= (Δ / nL) .* d_lbfgs
        return :lbfgs
    end

    # Case 3: Dogleg — find intersection of segment  d_lbfgs → d_newton
    #         with the trust-region boundary  ‖d_lbfgs + τ·p‖² = Δ²
    p = similar(d_newton)
    p .= d_newton .- d_lbfgs
    a = norm(p)^2
    b = 2 * dot(d_lbfgs, p)
    c = nL^2 - Δ^2

    # τ ∈ [0, 1] — take the larger (positive) root
    τ = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
    dz .= d_lbfgs .+ τ .* p
    return :dogleg
end
