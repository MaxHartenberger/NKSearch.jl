# How to Correctly Use L-BFGS in NKSearch.jl

## Quick Summary

L-BFGS minimises $\phi(z) = \frac{1}{2}\|F(z)\|^2$ instead of solving $F(z)=0$ directly. It requires **three flow operators** (nonlinear `G`, forward linearised `L`, adjoint `L_adj`) and uses **stage caching** (`TimeStepFromCache`) to avoid recomputing integration stages.

**TL;DR call signature:**
```julia
search!(G, L, L_adj, (dxdt, x) -> F(0, x, dxdt), z,
        Options(method=:lbfgs_opt, lbfgs_memory=10, maxiter=100))
```

---

## 1. Three Required Flow Operators

| Flow | Mode | Purpose | Called when |
|---|---|---|---|
| `G` | `NormalMode` | Nonlinear propagate | `update!` — fills stage caches, computes residual |
| `L` | `DiscreteMode(false)` | Forward tangent-linear | Not directly used in L-BFGS, but needed by `StageIterCache` |
| `L_adj` | `DiscreteMode(true)` | Adjoint (backward) | `mul!` on `AdjointIterSolCache` — computes $J^T \cdot F(z)$ (the gradient) |

**All three must use `TimeStepFromCache()`** (not `TimeStepConstant`) so the stage caches are populated by `G`'s propagate and then reused by `L_adj`'s backward integration.

---

## 2. Flow Construction Pattern

```julia
using NKSearch, Flows, LinearAlgebra

# --- Nonlinear flow ---
G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))          # Note: TimeStepConstant is fine for G

# --- Forward linearised flow (tangent-linear) ---
# MUST use a named callable struct, NOT a lambda (see §5 below)
L = flow(TangentSystem(D_lin),
         RK4(zeros(dim), Flows.DiscreteMode(false)),
         TimeStepFromCache())

# --- Adjoint flow ---
# MUST use a named callable struct, NOT a lambda
L_adj = flow(AdjointTangentSystem(D_adj),
             RK4(zeros(dim), Flows.DiscreteMode(true)),
             TimeStepFromCache())
```

---

## 3. The Three Operator Types You Must Define

### 3a. Nonlinear right-hand side `F(t, x, dxdt)`
Standard 3-argument in-place ODE RHS:
```julia
struct System
    μ::Float64
end
function (s::System)(t, u, dudt)
    # ... fill dudt ...
    return dudt
end
```

### 3b. Forward linearised RHS `D(t, x, dxdt, v, dvdt)` → 5 arguments
`Flows.DiscreteMode(false)` needs the 5-argument form:
```julia
struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
end
function (s::SystemLinear)(t, u, dudt, v, dvdt)
    # fill Jacobian s.J, then dvdt = J * v
    return mul!(dvdt, s.J, v)
end
```

Must be wrapped in a named struct for `Flows` compatibility:
```julia
struct TangentSystem{DType}
    D::DType
end
(s::TangentSystem)(t, x, v, dv) = s.D(t, x, dv, v, dv)  # reorder args for Flows
```

### 3c. Adjoint RHS `D_adj(x, w, dw)` → 3 arguments
`Flows.DiscreteMode(true)` needs the **3-argument** form `(x, w, dw)` computing $J^T \cdot w$:
```julia
struct SystemLinearAdjoint
    μ::Float64
    J::Matrix{Float64}
end
function (s::SystemLinearAdjoint)(x, w, dw)
    # fill Jacobian s.J evaluated at x, then dw = J' * w
    return mul!(dw, s.J', w)
end
```
**Important:** `DiscreteMode(true)` discards `t` — the adjoint RHS receives `(x, w, dw)` only, NOT `(t, x, w, dw)`.

Must be wrapped in a named struct:
```julia
struct AdjointTangentSystem{DType}
    D::DType
end
(s::AdjointTangentSystem)(t, x, w, dw) = s.D(x, w, dw)  # drop t
```

---

## 4. Phase-Locking Closure

The 6th argument to `search!` is the phase-locking condition. Wrap your RHS:
```julia
phase_lock = (dxdt, x) -> F_sys(0, x, dxdt)
```
This is passed as the `D` operator in the internal dispatch.
- For ordinary periodic orbits (`NS == 1`): only `D[1]` (the RHS) is used.
- For relative periodic orbits (`NS == 2`): internally `D = (F, dS)` where `F`
  is the RHS and `dS` is the derivative of the spatial shift operator w.r.t.
  its parameter.  At the `search!` call site, `F` and `dS` are passed as
  separate arguments (see §6).

---

## 5. ⚠️ CRITICAL: No Lambdas for Flow Operators

**Do NOT use lambdas for `L` or `L_adj`.** Julia closures are immutable; `deepcopy` does not reliably deep-copy captured mutable fields, causing **data races under multi-threading**.

❌ **Wrong:**
```julia
L = flow((t, x, v, dv) -> D(t, x, dv, v, dv), ...)   # lambda — WILL cause races
```

✅ **Correct:**
```julia
struct TangentSystem{DType}
    D::DType
end
(s::TangentSystem)(t, x, v, dv) = s.D(t, x, dv, v, dv)
L = flow(TangentSystem(D), ...)   # named struct — safe under threads
```

The same pattern applies to the adjoint flow.  The **spatial shift operator
`S` and its derivative `dS`** must also be named callable structs for thread
safety.  See `test/runtests.jl` lines 68–78 for the canonical definitions
and `test/test_adjoint.jl` for the `SpatialShift` / `SpatialShiftDerivative`
patterns.

---

## 6. Calling `search!`

### Ordinary periodic orbit (NS == 1, no spatial shift)
```julia
search!(G, L, L_adj, (dxdt, x) -> F_sys(0, x, dxdt), z,
        Options(method=:lbfgs_opt, ...))
```
5 arguments: `G, L, L_adj, phase_lock, z`.

### Relative periodic orbit (NS == 2, with spatial shift)
```julia
search!(G, L, L_adj, S, F, dS, z,
        Options(method=:lbfgs_opt, ...))
```
7 arguments: `G, L, L_adj, S, F, dS, z`.  `F` and `dS` are passed
separately (internally combined as `(F, dS)`).  Requires a spatial
shift operator `S` (e.g. `S(x, s)` shifts state `x` by `s` in place)
and its derivative `dS` (`dS(out, x)` computes ∂S/∂s evaluated at `x`).
The adjoint transpose applies `S(·, -s)` to segment N before the
backward integration, matching the forward composition `S ∘ Dϕ_N`.

---

## 7. Recommended Options

```julia
Options(
    method       = :lbfgs_opt,    # required
    maxiter      = 100,           # L-BFGS typically needs more iterations than Newton
    e_norm_tol   = 1e-16,         # tolerance on ‖F(z)‖
    dz_norm_tol  = 1e-16,         # tolerance on step norm
    lbfgs_memory = 10,            # number of (s,y) history pairs (typical: 5–20)
    ls_maxiter   = 20,            # more line-search iterations than Newton (default 10)
    verbose      = true,
    skipiter     = 1,
)
```

**Key differences from Newton methods:**
- `lbfgs_memory` is specific to L-BFGS (ignored by other methods).
- `maxiter` should be larger — L-BFGS converges more slowly per iteration but each iteration is cheaper (no GMRES solve).
- `gmres_*` options are **ignored** (no Krylov solver is used).
- `tr_radius_*` options are **ignored** (no trust region).

---

## 8. Threading

The L-BFGS method parallelises across shooting segments using `@spawn` tasks. **Run Julia with `N` threads** where `N` is the number of shooting segments:

```bash
julia --project=. -t 4 test/runtests.jl
```

A `StageIterCache`/`AdjointIterSolCache` pair shares arrays (`xT`, `tmp`, `stage_caches`). The forward `update!` populates them; the adjoint `mul!` reads them. The per-segment `@spawn` blocks are synchronised with `@sync`.

---

## 9. Adjoint Identity Validation

To verify that the spatial-shift adjoint is the exact algebraic transpose
of the forward Jacobian, run the adjoint identity test:

```bash
julia --project=. -t 2 test/runtests.jl
```

This checks $\langle Jv, w\rangle = \langle v, J^T w\rangle$ for both
`NS == 1` (ordinary periodic orbits) and `NS == 2` (relative periodic
orbits with spatial shift).  The identity must hold to $\sim 10^{-10}$
for all three test modes: segment-only, scalar-only, and full random vectors.

---

## 10. Current Limitations

1. **No JFOp** — The Jacobian-free operator (`JFOp`) only provides forward finite-difference action, not adjoint. L-BFGS requires an explicit adjoint linearisation.
2. **`Direction` memory** — The L-BFGS direction vector is allocated once in `OptLBFGSCache` and reused; the step `dz = direction * λ` is not a separate allocation.
3. **No trust region** — L-BFGS uses backtracking line search only; the step may be large in early iterations.

---

## 11. Complete Minimal Working Example (NS == 1)

```julia
using NKSearch, Flows, LinearAlgebra

# --- System definition ---
struct Hopf
    μ::Float64
end
(f::Hopf)(t, u, dudt) = begin
    x, y = u[1], u[2]
    r = sqrt(x^2 + y^2)
    dudt[1] = -y + f.μ*x*(1 - r)
    dudt[2] =  x + f.μ*y*(1 - r)
    return dudt
end

# --- Forward linearised ---
struct HopfLin
    μ::Float64; J::Matrix{Float64}
    HopfLin(μ) = new(μ, zeros(2,2))
end
function (h::HopfLin)(t, u, dudt, v, dvdt)
    x, y = u[1], u[2]; r = sqrt(x^2 + y^2)
    h.J[1,1] = h.μ*(1 - r - x^2/r); h.J[1,2] = -1 - h.μ*x*y/r
    h.J[2,1] =  1 - h.μ*x*y/r;      h.J[2,2] = h.μ*(1 - r - y^2/r)
    return mul!(dvdt, h.J, v)
end

# --- Adjoint ---
struct HopfAdj
    μ::Float64; J::Matrix{Float64}
    HopfAdj(μ) = new(μ, zeros(2,2))
end
function (h::HopfAdj)(x, w, dw)
    ux, uy = x[1], x[2]; r = sqrt(ux^2 + uy^2)
    h.J[1,1] = h.μ*(1 - r - ux^2/r); h.J[1,2] = -1 - h.μ*ux*uy/r
    h.J[2,1] =  1 - h.μ*ux*uy/r;      h.J[2,2] = h.μ*(1 - r - uy^2/r)
    return mul!(dw, h.J', w)
end

# --- Named wrappers (mandatory for thread safety) ---
struct TanSys{D}; D::D; end
(s::TanSys)(t, x, v, dv) = s.D(t, x, dv, v, dv)

struct AdjSys{D}; D::D; end
(s::AdjSys)(t, x, w, dw) = s.D(x, w, dw)

# --- Build flows ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)

G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

L = flow(TanSys(HopfLin(μ)),
         RK4(zeros(dim), Flows.DiscreteMode(false)),
         TimeStepFromCache())

L_adj = flow(AdjSys(HopfAdj(μ)),
             RK4(zeros(dim), Flows.DiscreteMode(true)),
             TimeStepFromCache())

# --- Initial guess ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)

# --- Search ---
search!(G, L, L_adj, (dxdt, x) -> F_sys(0, x, dxdt), z,
        Options(method=:lbfgs_opt, maxiter=100, e_norm_tol=1e-14,
                lbfgs_memory=10, ls_maxiter=20, verbose=true))

# --- Verify ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
```

## 12. Complete Minimal Working Example (NS == 2, with spatial shift)

The Hopf normal form is rotationally symmetric, so a relative periodic
orbit closes up to a rotation by angle $s$.

```julia
using NKSearch, Flows, LinearAlgebra

# --- System, linearised, and adjoint definitions same as NS == 1 ---
# (Hopf, HopfLin, HopfAdj, TanSys, AdjSys as above)

# --- Spatial shift: rotation by angle s ---
struct SpatialShift end
function (::SpatialShift)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

# --- Derivative of spatial shift: infinitesimal generator ---
struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    out[1] = -x[2]     # ∂/∂s of rotation at x
    out[2] =  x[1]
    return out
end

# --- Build flows (same as NS == 1) ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)
G = flow(F_sys, RK4(zeros(dim), Flows.NormalMode()), TimeStepConstant(dt))
L = flow(TanSys(HopfLin(μ)), RK4(zeros(dim), Flows.DiscreteMode(false)), TimeStepFromCache())
L_adj = flow(AdjSys(HopfAdj(μ)), RK4(zeros(dim), Flows.DiscreteMode(true)), TimeStepFromCache())

S_op  = SpatialShift()
dS_op = SpatialShiftDerivative()

# --- Initial guess with zero initial shift ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π, 0.0)

# --- Search (7-argument form: F and dS are separate args) ---
search!(G, L, L_adj, S_op, F_sys, dS_op, z,
        Options(method=:lbfgs_opt, maxiter=100, e_norm_tol=1e-14,
                lbfgs_memory=10, ls_maxiter=20, verbose=true))

# --- Verify: orbit lies on unit circle, shift is zero for this symmetric case ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
@assert abs(z.d[2]) < 1e-9
```

---

## 13. Files Involved

| File | Role |
|---|---|
| `src/newton.jl` | `search!` entry points, dispatch to `_search_lbfgs_opt!` |
| `src/search_lbfgs.jl` | `_search_lbfgs_opt!`, `OptLBFGSCache`, two-loop recursion, line search |
| `src/lbfgs_sol_cache.jl` | `StageIterCache` (forward, fills stage caches), `AdjointIterSolCache` (adjoint mat-vec $J^T \cdot w$) |
| `src/multivector.jl` | `MVector` type (seeds + scalar unknowns) |
| `src/options.jl` | `Options` struct (fields: `lbfgs_memory`, `ls_maxiter`, etc.) |
| `src/output.jl` | L-BFGS verbose status table |
| `test/runtests.jl` | System definitions (`System`, `SystemLinear`, `SystemLinearAdjoint`, wrappers) |
| `test/test_search.jl` | L-BFGS convergence test |
| `test/test_adjoint.jl` | Adjoint identity test $\langle Jv, w\rangle = \langle v, J^T w\rangle$ |
| `test/test_parallel.jl` | Thread-safety stress test |
