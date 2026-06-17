# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import LinearAlgebra
import HDF5: write, h5open, h5readattr, attributes, read

export MVector, 
       tovector,
       fromvector!,
       nsegments,
       save_seeds,
       load_seeds!,
       find_number_of_segments

# ~~~ INTERFACE FOR MVector and MMatrix ~~~
# The type parameter `X` must support
# 1) dot(::X, ::X)
# 2) similar(::X)
# 3) full broadcast functionality, with variables of type `X` and scalars
# 4) zero(::X)

# ~~~ Vector Type ~~~
"""
    MVector(x::NTuple{N,X}, T::Real)            -> MVector{X,N,1}
    MVector(x::NTuple{N,X}, T::Real, s::Real)   -> MVector{X,N,2}

A multiple-shooting unknown: the state of a candidate (relative) periodic
orbit, packaged as a single vector for the Newton solver.

It bundles `N` *seed* states `x = (x₁, …, x_N)` sampled along the orbit
with the scalar unknowns `d`: the period `T` and, for a relative periodic
orbit, an additional spatial shift `s`.

# Arguments
- `x::NTuple{N,X}`: the `N` seeds, one per shooting segment. Each seed is a
  state of type `X` (see the element-type requirements below).
- `T::Real`: the orbit period (the first scalar unknown).
- `s::Real`: optional spatial shift, present only for relative periodic
  orbits with a continuous spatial symmetry.

# Element type `X`
`X` must support `LinearAlgebra.dot`, `Base.similar`, `Base.zero`, and full
broadcasting against other `X` values and scalars. Plain `Vector{Float64}`
satisfies this, as do the field types in `Flows`.

# Fields
- `x::NTuple{N,X}`: the seeds.
- `d::NTuple{NS,Float64}`: the scalar unknowns, `(T,)` or `(T, s)`.

`MVector` is itself an `AbstractVector{Float64}` of length `N*length(x₁) + NS`,
so it can be passed to linear-algebra routines; use [`tovector`](@ref) /
[`fromvector!`](@ref) to convert to and from a flat `Vector{Float64}`.

# Examples
```jldoctest
julia> z = MVector(([1.0, 2.0], [3.0, 4.0]), 6.28);

julia> nsegments(z)
2

julia> z.d
(6.28,)

julia> tovector(z)
5-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
 6.28
```
"""
mutable struct MVector{X, N, NS} <: AbstractVector{Float64}
    x::NTuple{N, X}        # the seed along the orbit
    d::NTuple{NS, Float64} # period and optional NS-1 shifts
    MVector(x::NTuple{N, X}, T::Real, s::Real) where {N, X} =
        new{X, N, 2}(x, Float64.((T, s)))
    MVector(x::NTuple{N, X}, T::Real) where {N, X} =
        new{X, N, 1}(x, Float64.((T, )))
end

# getindex to have z[i] mean z.x[i]
Base.getindex(z::MVector, i::Int) = z.x[i]

"""
    nsegments(z::MVector) -> Int

Return the number of multiple-shooting segments (seeds) `N` stored in `z`.
"""
nsegments(::MVector{X, N}) where {X, N} = N

# interface for GMRES solver
Base.similar(z::MVector) = MVector(similar.(z.x), z.d...)
Base.copy(z::MVector) = MVector(copy.(z.x), z.d...)
Base.zero(z::MVector) = MVector(zero.(z.x), (zero(d) for d in z.d)...)
LinearAlgebra.norm(z::MVector) = sqrt(LinearAlgebra.dot(z, z))
LinearAlgebra.dot(a::MVector{X, N}, b::MVector{X, N}) where {X, N} =
    sum(a.d.*b.d) + sum(LinearAlgebra.dot.(a.x, b.x))

# define stuff necessary to use . notation with MVector (only works
# for short! expressions) otherwise code is very slow!
const MVectorStyle = Broadcast.ArrayStyle{MVector}
Base.BroadcastStyle(::Type{<:MVector}) = Broadcast.ArrayStyle{MVector}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{MVector},
                    ::Broadcast.DefaultArrayStyle{1}) = Broadcast.DefaultArrayStyle{1}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{1},
                    ::Broadcast.ArrayStyle{MVector}) = Broadcast.DefaultArrayStyle{1}()

"""
    tovector(z::MVector) -> Vector{Float64}

Flatten `z` into a freshly allocated `Vector{Float64}` of length
`N*length(z[1]) + NS`: the `N` seeds concatenated first, followed by the
scalar unknowns `z.d` (period and optional shift). Inverse of
[`fromvector!`](@ref).
"""
function tovector(z::MVector{X, N, NS}) where {X, N, NS}
    n = length(z[1])
    out = zeros(N*n + NS)
    for i = 1:N
        out[_blockrng(i, n)] .= z[i]
    end
    out[end - NS + 1 : end] .= z.d
    return out
end

"""
    fromvector!(out::MVector, v::Vector{<:Real}) -> out

Copy the flat representation `v` (as produced by [`tovector`](@ref)) back
into the seeds and scalar unknowns of `out`, in place, and return `out`.
`v` must have length `nsegments(out)*length(out[1]) + NS`.
"""
function fromvector!(out::MVector{X, N, NS}, v::Vector{<:Real}) where {X, N, NS}
    n = length(out[1])
    for i = 1:N
        out[i] .= view(v, _blockrng(i, n))
    end
    out.d = ntuple(j->v[end-NS+j], NS)
    return out
end

# a hack really!
Base.size(z::MVector{X, N, NS}) where {X, N, NS} = (NS + N*length(z.x[1]), )

# getters
_get_seed(z::MVector, i) = z.x[i]
_get_seed(z, i) = z
_get_d(z::MVector) = z.d
_get_d(z) = z

@inline function Broadcast.copyto!(dest::MVector{X, N},
                                     bc::Broadcast.Broadcasted{MVectorStyle}) where {X, N}
    bcf = Broadcast.flatten(bc)
    for i = 1:N
        Broadcast.broadcast!(bcf.f, dest.x[i], map(arg->_get_seed(arg, i), bcf.args)...)
    end
    dest.d = Broadcast.broadcast(bcf.f, map(_get_d, bcf.args)...)
    return dest
end

function save(z::MVector{X, N, NS}, path::String) where {X, N, NS}
    # save trajectory to a large matrix first
    data = zeros(Float64, length(z[1]), N)
    for (i, zi) in enumerate(z.x)
        data[:, i] .= zi
    end
    h5open(path, "w") do file
        write(file, "seed", data)
        for i = 1:NS
            attributes(file)["d$i"] = z.d[i]
        end
    end
end

"""
    find_number_of_segments(M, T, Tmin, Tmax) -> Int

Pick a number of shooting segments `N` that divides `M` evenly while keeping
each segment's duration `T/N` within `[Tmin, Tmax]`. Concretely, return the
first integer `N` with `M % N == 0` in the range `ceil(T/Tmax) ≤ N ≤
floor(T/Tmin)`, or `0` if none exists.

This is useful when the seeds come from a stored trajectory of `M` samples
that must be split into equal segments.

# Arguments
- `M::Int`: number of available samples / time steps along the trajectory.
- `T::Real`: total period.
- `Tmin::Real`, `Tmax::Real`: minimum and maximum allowed duration per
  segment (`Tmin < Tmax`).

# Examples
```jldoctest
julia> find_number_of_segments(12, 10.0, 1.0, 3.0)
4
```
"""
function find_number_of_segments(M::Int, T::Real, Tmin::Real, Tmax::Real)
    Nmin = Int(floor(T/Tmin))
    Nmax = Int(ceil( T/Tmax))    
    for N = Nmax:Nmin
        if M % N == 0 
            return N
        end
    end
    return 0
end

# a hack
_is_complex_eltype(z::MVector) = eltype(parent(z[1])) <: Complex

"""
    save_seeds(z::MVector, path::String, other=Dict{String,Any}())

Write the orbit `z` to the HDF5 file `path`. Each seed is stored as a
dataset (`seed_i`, or `seed_i_real`/`seed_i_imag` for complex states), the
scalar unknowns `z.d` as attributes `d1`, `d2`, …, and every key/value in
`other` as an attribute prefixed with `other_`.

The seeds are written via `parent(z.x[i])`, so this supports states that
wrap an array (e.g. `Flows` field types). Use [`load_seeds!`](@ref) to read
the file back.
"""
function save_seeds(z::MVector{X, N, NS},
                    path::String,
                    other::Dict{String, <:Any} = Dict{String, Any}()) where {X, N, NS}
    # test whether X is a complex type
    h5open(path, "w") do file
        if _is_complex_eltype(z)
            for i = 1:N
                write(file, "seed_$(i)_real", real(parent(z.x[i])))
                write(file, "seed_$(i)_imag", imag(parent(z.x[i])))
            end
        else
            for i = 1:N
                write(file, "seed_$i", parent(z.x[i]))
            end
        end
        for i = 1:NS
            attributes(file)["d$i"] = z.d[i]
        end
        for (k, v) in other
            attributes(file)["other_$k"] = v
        end
    end
end

"""
    load_seeds!(fun, path::String) -> (z::MVector, other::Dict{String,Any})

Read an orbit written by [`save_seeds`](@ref) from the HDF5 file `path`.

`fun` is applied to each raw seed array as it is read, so it can wrap the
plain array back into the state type you use (pass `identity` to keep plain
arrays). Returns the reconstructed [`MVector`](@ref) together with a `Dict`
of any extra attributes that were stored under the `other_` prefix.
"""
function load_seeds!(fun, path::String)
    h5open(path, "r") do file
        
        # load attributes handle
        attrs = attributes(file)

        # load bit here
        dict = Dict{String, Any}()

        # store seeds here
        xs = []

        # determine if we saved complex data
        is_complex_data = "seed_1_real" in keys(file)
            
        # number of seeds
        N = is_complex_data ? div(length(keys(file)), 2) : length(keys(file))

        # read real and imaginary part if we need so
        if is_complex_data
            for i = 1:N
                push!(xs, fun(read(file, "seed_$(i)_real") .+ im.*read(file, "seed_$(i)_imag")))
            end
        else
            for i = 1:N
                push!(xs, read(file, "seed_$i"))
            end
        end

        # and period and shifts (all those that start with d)
        d = [read(attrs[el]) for el in keys(attrs) if startswith(el, "d")]

        # also load other bits that might have been saved
        for k in keys(attrs)
            if startswith(k, "other_")
                dict[k[7:end]] = read(attrs[k])
            end
        end

        return MVector(tuple(xs...), d...), dict
    end
end

"""
    _residual_norm(b::MVector, e_norm_type::Symbol)

Compute the residual norm according to `e_norm_type`.
- `:euclidean`   → ‖b‖ (standard Euclidean over all segments + scalar part)
- `:max_segment` → max_i ‖b.x[i]‖ (worst segment, ignores scalar part)
"""
function _residual_norm(b::MVector, e_norm_type::Symbol)
    if e_norm_type == :max_segment
        return maximum(norm, b.x)
    else
        return norm(b)
    end
end