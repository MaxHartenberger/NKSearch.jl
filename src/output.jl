# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
using Printf

# ~~~ HEADERS ~~~

# line search
const _header_1_ls = "+------+----------+-----------+----------+-----------+----------+\n"*
                     "| iter |   |dz|   |     T     |   ||e||  |     λ     |    res   |\n"*
                     "+------+----------+-----------+----------+-----------+----------+\n"

const _header_2_ls = "+------+----------+-----------+-----------+----------+----------+----------+\n"*
                     "| iter |  ||dz||  |    T      |     s     |   ||e||  |     λ    |    res   |\n"*
                     "+------+----------+-----------+-----------+----------+----------+----------+\n"

const _headers_ls = [_header_1_ls, _header_2_ls]

display_header_ls(io::IO, ::MVector{X, N, NS}) where {X, N, NS} = 
    (print(io, _headers_ls[NS]); flush(io))

# trust region
const _header_tr = "+------+--------+-----------+-----------+------------+-----------+\n"*
                   "| iter | which  |  ||dz||   |   ||e||   |    rho     | tr_radius |\n"*
                   "+------+--------+-----------+-----------+------------+-----------+\n"

display_header_tr(io::IO, ::MVector{X, N, NS}) where {X, N, NS} = 
    (print(io, _header_tr); flush(io))

# hookstep
const _header_hks = "+------+--------+-----------+-------------+------------+-----------+-----------+----------+\n"*
                    "| iter | which  |  ||dz||   |    ||e||    |    rho     | tr_radius | GMRES res | GMRES it |\n"*
                    "+------+--------+-----------+-------------+------------+-----------+-----------+----------+\n"

display_header_hks(io::IO, ::MVector{X, N, NS}) where {X, N, NS} = 
    (print(io, _header_hks); flush(io))

# ~~~ DISPLAY FUNCTIONS ~~~

# line search

# print output when we have a shift. `e_norm` is the actual norm ||F||.
function display_status_ls(io::IO, iter, dz_norm, d::TUP2, e_norm, λ, res_err_norm) where {TUP2<:Tuple{Any, Any}}
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e |" iter dz_norm d[1] d[2] e_norm λ res_err_norm
    println(io, str)
    flush(io)
    return nothing
end

# print output when we don't
function display_status_ls(io::IO, iter, dz_norm, d::Tuple{Any}, e_norm, λ, res_err_norm)
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %5.2e | %5.2e | %5.2e |" iter dz_norm d[1] e_norm λ res_err_norm
    println(io, str)
    flush(io)
    return nothing
end

# trust region
function display_status_tr(io::IO, iter, which, dz_norm, e_norm, rho, tr_radius)
    str = @sprintf "|%4d  | %s | %5.3e | %5.3e | %+5.3e | %5.3e |" iter lpad(which, 6) dz_norm e_norm rho tr_radius
    println(io, str)
    flush(io)
end

# trust region
function display_status_hks(io::IO, iter, which, dz_norm, e_norm, rho, tr_radius, GMRES_res, GMRES_it)
    str = @sprintf "|%4d  | %s | %5.3e | %7.5e | %+5.3e | %5.3e | %5.3e | %8d |" iter lpad(which, 6) dz_norm e_norm rho tr_radius GMRES_res GMRES_it
    println(io, str)
    flush(io)
end

# LBFGS
const _header_lbfgs = "+------+--------+---------------+---------------+------------+\n"*
                       "| iter | which  |    ||∇ϕ||     |     ||F||     |     λ     |\n"*
                       "+------+--------+---------------+---------------+------------+\n"

display_header_lbfgs(io::IO, ::MVector{X, N, NS}) where {X, N, NS} =
    (print(io, _header_lbfgs); flush(io))

function display_status_lbfgs(io::IO, iter, which, ∇ϕ_norm, f_norm, λ)
    str = @sprintf "|%4d  | %s | %5.3e | %5.3e | %+5.3e |" iter lpad(which, 6) ∇ϕ_norm f_norm λ
    println(io, str)
    flush(io)
end
