# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
using Printf

# trust region method implementation
function _search_trustregion!(Gs, Ls, S, D, z, cache, opts)
    # display nice header
    opts.verbose && display_header_tr(opts.io, z)

    # allocate memory
    b    = similar(z)                             # right hand side
    dz   = similar(z)                             # temporary
    tmps = ntuple(i->similar(z[1]), nsegments(z)) # one temporary for each segment

    # calculate initial error
    e_norm = e_norm_λ(Gs, S, z, z, 0.0, tmps)

    # init
    tr_radius = opts.tr_radius_init

    # display status if verbose
    opts.verbose && display_status_tr(opts.io,
                                      0,
                                      :start,
                                      0,
                                      e_norm,
                                      0,
                                      tr_radius)

    status = :maxiter_reached

    # newton iterations loop
    for iter = 1:opts.maxiter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # UPDATE CACHE
        update!(cache, b, z, opts)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM
        hits_boundary, which, step = solve_tr_subproblem!(dz, b, z, cache, tr_radius, opts)

        # calc actual reductions
        e_norm_curr = e_norm_λ(Gs, S, z, dz, 0.0, tmps)
        e_norm_next = e_norm_λ(Gs, S, z, dz, 1.0, tmps)
        actual = e_norm_curr^2 - e_norm_next^2

        # calc predicted reduction
        predicted = norm(cache * dz)^2

        # calc ratio
        rho = actual/predicted

        if e_norm_curr > opts.NR_lim
            # trust region update
            if rho < 1/4
                tr_radius *= 1/4
            elseif rho > 3/4 && hits_boundary
                tr_radius = min(2*tr_radius, opts.tr_radius_max)
            end

            # solution update if reduction is large enough
            if rho > opts.eta
                z .= z .+ dz
                e_norm = e_norm_next
            else
                e_norm = e_norm_curr
            end
        else
            z .= z .+ opts.α.*dz
            e_norm = e_norm_next
        end

        dz_norm = norm(dz)

        # display status if verbose
        if opts.verbose && iter % opts.skipiter == 0
            display_status_tr(opts.io,
                              iter,
                              which,
                              dz_norm,
                              e_norm,
                              rho,
                              tr_radius)
        end

        # tolerances reached
        if e_norm <  opts.e_norm_tol
            status = :converged
            break
        end
        if dz_norm < opts.dz_norm_tol
            status = :converged
            break
        end
        if step < opts.min_step
            status = :min_step_reached
            break
        end
    end

    # return input
    return status
end