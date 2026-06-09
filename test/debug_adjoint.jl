using NKSearch, LinearAlgebra, Flows

struct Sys; μ::Float64; end
function (s::Sys)(t, x, dxdt)
    x_, y_ = x[1], x[2]
    r = sqrt(x_^2 + y_^2)
    dxdt[1] = -y_ + s.μ*x_*(1 - r)
    dxdt[2] =  x_ + s.μ*y_*(1 - r)
    return dxdt
end

struct SysLinAdj; μ::Float64; J::Matrix{Float64}; SysLinAdj(μ) = new(μ, zeros(2,2)); end
function (s::SysLinAdj)(x, w, dw)
    x_, y_ = x[1], x[2]
    r = sqrt(x_^2 + y_^2)
    s.J[1,1] = s.μ*(1 - r - x_^2/r)
    s.J[1,2] = -1 - s.μ*x_*y_/r
    s.J[2,1] =  1 - s.μ*x_*y_/r
    s.J[2,2] = s.μ*(1 - r - y_^2/r)
    return mul!(dw, s.J', w)
end

μ = 1.0; F = Sys(μ); D_adj_raw = SysLinAdj(μ)

# Forward nonlinear flow (NormalMode, TimeStepConstant)
G = flow(F, RK4(zeros(2), Flows.NormalMode()), TimeStepConstant(1e-3))

# Adjoint flow (DiscreteMode{true}, TimeStepFromCache)
adj_flow = flow(
    (t, x, w, dw) -> D_adj_raw(x, w, dw),
    RK4(zeros(2), Flows.DiscreteMode(true)),
    TimeStepFromCache())

z = MVector(([2, 0.0], [-2, 0.0]), 2π)
D_phase = ((dxdt, x) -> F(0, x, dxdt),)

# Build caches
cache = NKSearch.IterSolCache((G, G), (nothing, nothing), (adj_flow, adj_flow), nothing, D_phase, z)
adj_cache = Base.adjoint(cache)
b = similar(z)
NKSearch.update!(cache, b, z)
println("F(z)   = ", b.x)
println("‖F‖²   = ", norm(b)^2)

# Compute J^T * F via adjoint
w = similar(z); w .= b
JTw = similar(z)
NKSearch.mul!(JTw, adj_cache, w)
println("J^T·F  = ", JTw.x, "  d = ", JTw.d)

# ============================================================
# FD verification: ∇φ = ∇(½‖F‖²) by central finite differences
# ============================================================
ϵ_fd = 1e-5
JTw_fd = similar(z)
ϕ0 = norm(b)^2 / 2

for seg in 1:2
    for j in 1:2
        zp = deepcopy(z); zp.x[seg][j] += ϵ_fd
        cp = NKSearch.IterSolCache((G, G), (nothing, nothing), nothing, nothing, D_phase, zp)
        bp = similar(zp); NKSearch.update!(cp, bp, zp)
        ϕp = norm(bp)^2 / 2

        zm = deepcopy(z); zm.x[seg][j] -= ϵ_fd
        cm = NKSearch.IterSolCache((G, G), (nothing, nothing), nothing, nothing, D_phase, zm)
        bm = similar(zm); NKSearch.update!(cm, bm, zm)
        ϕm = norm(bm)^2 / 2

        JTw_fd.x[seg][j] = (ϕp - ϕm) / (2ϵ_fd)
    end
end

# Period
zp = deepcopy(z); zp.d = (zp.d[1] + ϵ_fd,)
cp = NKSearch.IterSolCache((G, G), (nothing, nothing), nothing, nothing, D_phase, zp)
bp = similar(zp); NKSearch.update!(cp, bp, zp)
ϕp = norm(bp)^2 / 2

zm = deepcopy(z); zm.d = (zm.d[1] - ϵ_fd,)
cm = NKSearch.IterSolCache((G, G), (nothing, nothing), nothing, nothing, D_phase, zm)
bm = similar(zm); NKSearch.update!(cm, bm, zm)
ϕm = norm(bm)^2 / 2

JTw_fd.d = ((ϕp - ϕm) / (2ϵ_fd),)

println("FD ∇φ  = ", JTw_fd.x, "  d = ", JTw_fd.d)

# Adjoint now gives +J^T·F, FD gives +∇φ = +J^T·F.  They should match.
println("\nCheck:  J^T·F ≈ FD ∇φ ?")
println("  seg 1 diff: ", JTw.x[1] .- JTw_fd.x[1])
println("  seg 2 diff: ", JTw.x[2] .- JTw_fd.x[2])
println("  period diff: ", JTw.d[1] - JTw_fd.d[1])
