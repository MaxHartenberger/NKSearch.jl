using Test
using NKSearch
using LinearAlgebra
using Flows

# Problem from section 4 in Viswanath 2001
# ẋ = -y + μx(1 - √(x² + y²))
# ẏ =  x + μy(1 - √(x² + y²))
# The solution is the limit cycle
# x(t) = cos(t)
# y(t) = sin(t)
# with period T=2π and angular frequency ω=1
# The velocities are:
# ẋ(t) = -sin(t)
# ẏ(t) =  cos(t)

struct System
    μ::Float64
end

# right hand side
function (s::System)(t, x, dxdt)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds dxdt[1] = - y_ + μ*x_*(1 - r)
    @inbounds dxdt[2] =   x_ + μ*y_*(1 - r)
    return dxdt
end

# linearised operator
struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
    SystemLinear(μ::Real) = new(μ, zeros(2, 2))
end

function (s::SystemLinear)(t, x, dxdt, v, dvdt)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds s.J[1, 1] = μ*(1 - r - x_^2/r)
    @inbounds s.J[1, 2] = -1 - μ*x_*y_/r
    @inbounds s.J[2, 1] =  1 - μ*x_*y_/r
    @inbounds s.J[2, 2] = μ*(1 - r - y_^2/r)
    return mul!(dvdt, s.J, v)
end

# adjoint linearised operator (computes J^T * w for L-BFGS gradient)
struct SystemLinearAdjoint
    μ::Float64
    J::Matrix{Float64}
    SystemLinearAdjoint(μ::Real) = new(μ, zeros(2, 2))
end

function (s::SystemLinearAdjoint)(x, w, dw)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds s.J[1, 1] = μ*(1 - r - x_^2/r)
    @inbounds s.J[1, 2] = -1 - μ*x_*y_/r
    @inbounds s.J[2, 1] =  1 - μ*x_*y_/r
    @inbounds s.J[2, 2] = μ*(1 - r - y_^2/r)
    return mul!(dw, s.J', w)
end

# Named callable structs for use with Flows.jl — avoids the lambda closure
# deepcopy issue (Julia closures are immutable; deepcopy does not reliably
# deep-copy mutable captured fields, causing data races under threads).
struct TangentSystem{DType}
    D::DType
end
(s::TangentSystem)(t, x, v, dv) = s.D(t, x, dv, v, dv)

struct AdjointTangentSystem{DType}
    D::DType
end
(s::AdjointTangentSystem)(t, x, w, dw) = s.D(x, w, dw)  # 4-arg → 3-arg, discards t

#include("test_multivector.jl")
include("test_search.jl")
#include("test_search_shift.jl")
#include("test_jfop.jl")
#include("test_adjoint.jl")
#include("test_parallel.jl")