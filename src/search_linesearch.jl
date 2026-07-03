# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: @sync, @spawn, atomic_add!, Atomic
import LinearAlgebra: norm
import Flows

# line search method implementation
function _search_linesearch!(G, L, S, D, z0, A, opts)
    # display nice header
    opts.verbose && display_header_ls(opts.io, z0)

    # allocate memory
    b   = similar(z0)                   # right hand side
    dz  = similar(z0); dz .*= 0.0       # temporary step
    tmps = ntuple(i->similar(z0[1]), nsegments(z0)) # one temporary for each segment

    # calculate initial error
    e_norm = e_norm_λ(G, S, z0, z0, 0.0, tmps)

    # display status if verbose
    opts.verbose && display_status_ls(opts.io,
                                      0,
                                      0,
                                      z0.d,
                                      e_norm,
                                      0.0,
                                      0.0)

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton update matrix operator and right hand side
        update!(A, b, z0, opts)

        # solve system by overwriting b in place
        dz, res_err_norm = _solve(dz, A, b, opts)

        # perform line search
        λ, e_norm = linesearch(G, S, z0, dz, opts, tmps)

        # actually apply correction
        z0 .+= λ.*dz

        # correction norm
        dz_norm = norm(dz)

        # display status if verbose
        if opts.verbose && iter % opts.skipiter == 0
            display_status_ls(opts.io,
                              iter,
                              dz_norm,
                              z0.d,
                              e_norm,
                              λ,
                              res_err_norm)
        end

        # tolerances reached
         e_norm  < opts.e_norm_tol && break # norm of error
        dz_norm < opts.dz_norm_tol && break # norm of orbit correction
    end

    # return input
    return nothing
end

function e_norm_λ(Gs::NTuple{N},
                    S,
                   z0::MVector{X, N, NS},
                   δz::MVector{X, N, NS},
                    λ::Real,
                 tmps::NTuple{N, X}) where {X, N, NS}
    # error output
    out = Atomic{Float64}(0.0)

    # loop over segments summing the error
    @sync for i in 1:N
        @spawn begin
            # set initial condition
            tmps[i] .= z0[i] .+ λ.*δz[i]

            # actual propagation
            Gs[i](tmps[i], (0, (z0.d[1] + λ*δz.d[1])/N))

            # last segment is shifted (if we have a shift)
            NS == 2 && i == N && S(tmps[i], z0.d[2] + λ*δz.d[2])

            # calc difference
            tmps[i] .-= z0[i%N + 1] .+ λ.*δz[i%N + 1]

            # add to error
            atomic_add!(out, norm(tmps[i])^2)
        end
    end

    return sqrt(out[])
end

function linesearch(G, S, z0::MVector{X, N}, δz::MVector{X, N}, opts::Options, tmp::NTuple{N, X}) where {X, N}
    # current error
    val_0 = e_norm_λ(G, S, z0, δz, 0.0, tmp)

    # start with full Newton step
    λ = 1.0

    # initialize this variable
    val_λ = λ*val_0

    for iter = 1:opts.ls_maxiter
        # calculate error
        try
            val_λ = e_norm_λ(G, S, z0, δz, λ, tmp)
        catch err
            # We might end up in a situation where the
            # new time span has negative length. In
            # such a case, we might just continue
            if !isa(err, Flows.InvalidSpanError)
                rethrow(err)
            end
        end

        # accept any reduction of error
        val_λ < val_0 && return λ, val_λ

        # ~ otherwise attempt with shorter step ~
        λ *= opts.ls_rho
    end

    error("maximum number of line search iterations reached")
end