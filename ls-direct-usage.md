# How to Correctly Use the Direct Line-Search Method in NKSearch.jl

## Quick Summary

The `:ls_direct` method solves $F(z) = 0$ via **Newton's method** with a
**direct (assembled) Jacobian** and **backtracking line-search**
globalization.  At each Newton iteration, the full sparse Jacobian matrix
is assembled column-by-column using finite-difference action of the
linearised flow on identity columns, then the linear system $J\,\delta z
= -F(z)$ is solved via **LU factorisation**.

It requires **two flows** (nonlinear `G`, forward linearised `L`).

**TL;DR call signature:**
```julia
search!(G, L, (dxdt, x) -> F(0, x, dxdt), z,
        Options(method=:ls_direct, maxiter=25))
```

Because `:ls_direct` is the **default** method, `Options()` without an
explicit `method` also selects it.

---

## 1. Two Required Flows

| Flow | Mode | Purpose | Called when |
|---|---|---|---|
| `G` | `NormalMode` | Nonlinear propagate | `update!` — fills end-of-segment states, computes residual |
| `L` | `NormalMode` (standard coupled) | Forward tangent-linear | `update!` — column-by-column Jacobian assembly |

**Both use `TimeStepConstant`.**  Unlike L-BFGS, the direct line-search
method does **not** need stage caching (`TimeStepFromCache`) because
there is no adjoint backward integration.  There is **no adjoint flow**
and **no GMRES** in the direct method.

---

## 2. Flow Construction Pattern

```julia
using NKSearch, Flows, LinearAlgebra

# --- Nonlinear flow ---
G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

# --- Forward linearised flow (standard Flows.couple convention) ---
L = flow(couple(F_sys, D_lin),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))
```

Here `couple(F, D)` from `Flows.jl` pairs the base and perturbation RHS so
that `L(Flows.couple(x, v), span)` advances both `x` and `v` simultaneously.
See §3b for the `D_lin` signature.

---

## 3. The Two Operator Types You Must Define

### 3a. Nonlinear right-hand side $F(t, x, dxdt)$
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

### 3b. Forward linearised RHS $D(t, x, dxdt, v, dvdt)$ → 5 arguments

`Flows.couple` passes the linearised RHS **5 arguments**: $(t, x, dxdt, v, dvdt)$.
It must fill `dvdt = J(x)\cdot v` while also filling `dxdt = F(x)`:
```julia
struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
end
function (s::SystemLinear)(t, u, dudt, v, dvdt)
    # fill dudt = F(u)  AND  dvdt = J(u) * v
    s.J[1,1] = ...  # populate Jacobian
    # ...
    dudt[1] = ...   # fill nonlinear RHS
    # ...
    return mul!(dvdt, s.J, v)
end
```

`Flows.couple(F, D)` automatically applies `F` first, then `D`, so both
`dudt` and `dvdt` are updated.

**Note:** The `couple`-based linearised flow uses the standard
`Flows.NormalMode()`, **not** `Flows.DiscreteMode`.  This is the same
pattern as the hookstep (`:tr_iterative`) and iterative line-search
(`:ls_iterative`) methods.

---

## 4. Phase-Locking Closure

The 4th argument to `search!` is the phase-locking condition.  Wrap your RHS:
```julia
phase_lock = (dxdt, x) -> F_sys(0, x, dxdt)
```
This is passed as the `D` operator in the internal dispatch.
- For ordinary periodic orbits (`NS == 1`): only `D[1]` (the RHS) is used.
- For relative periodic orbits (`NS == 2`): internally `D = (F, dS)` where `F`
  is the RHS and `dS` is the derivative of the spatial shift operator w.r.t.
  its parameter.  At the `search!` call site, `F` and `dS` are passed as
  separate arguments (see §5).

---

## 5. Calling `search!`

### Ordinary periodic orbit (NS == 1, no spatial shift)
```julia
search!(G, L, (dxdt, x) -> F_sys(0, x, dxdt), z,
        Options(method=:ls_direct, ...))
```
4 arguments: `G, L, phase_lock, z`.

### Relative periodic orbit (NS == 2, with spatial shift)
```julia
search!(G, L, S, F, dS, z,
        Options(method=:ls_direct, ...))
```
6 arguments: `G, L, S, F, dS, z`.  `F` and `dS` are passed separately
(internally combined as `(F, dS)`).  Requires a spatial shift operator
`S` (e.g. `S(x, s)` shifts state `x` by `s` in place) and its derivative
`dS` (`dS(out, x)` computes $\partial S/\partial s$ evaluated at `x`).

The spatial shift operators must be **named callable structs** for thread
safety (same pattern as described in `lbfgs-usage.md` §5).  Even though
`:ls_direct` does not currently support multithreading, using named
structs is good practice and future-proof.

---

## 6. Callback

A user-supplied callback can monitor progress and optionally halt the
search early.  Set it via `Options(callback = ...)`.  The callback is
called once per Newton iteration with a fixed 7-argument signature:

```julia
callback(iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> Bool
```

| Arg | Type | Meaning in `:ls_direct` |
|---|---|---|
| `iter` | `Int` | Newton iteration number (1-based) |
| `z` | `MVector` | Current orbit — seeds and scalar unknowns (period, shift). **Do not mutate.** |
| `Fz` | `Vector{Float64}` | Copy of the right-hand side $b = -F(z)$ (the flat residual). Useful for saving residuals. |
| `e_norm` | `Float64` | Residual norm $\|F(z)\|$ |
| `∇ϕ_norm` | `Float64` | Always `0.0` in `:ls_direct` (placeholder shared with the L-BFGS callback signature). |
| `λ` | `Float64` | Step length accepted by the line search. `1.0` means the full Newton step was taken. |
| `T` | `Float64` | Period $T =$ `z.d[1]` |

Return `true` to terminate the search immediately.  The returned status
symbol will be `:callback_satisfied`.

### Example: saving the residual history

```julia
residuals = Float64[]
cb = (iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> begin
    push!(residuals, e_norm)
    return false   # never stop early
end

search!(G, L, phase_lock, z,
        Options(method=:ls_direct, callback=cb, ...))
```

---

## 7. Recommended Options

```julia
Options(
    method          = :ls_direct,      # selects the direct line-search method (also the default)
    maxiter         = 25,              # Newton iterations
    e_norm_tol      = 1e-10,           # tolerance on ‖F(z)‖
    dz_norm_tol     = 1e-10,           # tolerance on step norm
    verbose         = true,
    skipiter        = 1,

    # --- Line-search options ---
    ls_maxiter      = 10,              # maximum backtracking iterations
    ls_rho          = 0.5,             # step-reduction factor per backtrack

    # --- Finite-difference options ---
    ϵ               = 1e-6,            # step for finite-difference time-derivative
    fd_order        = 2,               # 1 = forward diff, 2 = central diff (more accurate)

    # --- Not used (ignored) ---
    # gmres_*         → ignored (no GMRES)
    # tr_radius_*     → ignored (no trust region)
    # lbfgs_memory    → ignored (not L-BFGS)
)
```

**Key settings for `:ls_direct`:**
- `ls_maxiter` and `ls_rho` control the backtracking line search.  If the
  full Newton step does not reduce the residual, the step is repeatedly
  shrunk by `ls_rho` up to `ls_maxiter` times.
- `ϵ` and `fd_order` control the accuracy of the finite-difference
  approximation used to compute the time-derivative of the flow (right-hand
  side of the Jacobian system).  Central differences (`fd_order=2`) give
  $\mathcal{O}(\epsilon^2)$ accuracy but cost twice as many flow evaluations.

---

## 8. How the Direct Line-Search Algorithm Works

The `:ls_direct` method is a **Newton solver with direct Jacobian
assembly and backtracking line search**.  Each outer iteration performs:

1. **Update** (`update!`): assemble the full sparse Jacobian matrix $J$
   and right-hand side $b = -F(z)$.
   - **Jacobian assembly**: for each shooting segment $i$, the linearised
     flow $L$ is applied to each column of the identity matrix.  This
     builds the $i$-th block diagonal of $J$ column by column.  For an
     $n$-dimensional state and $N$ segments, this requires $N \cdot n$
     linearised flow integrations per Newton iteration.
   - **Right-hand side**: propagate each seed with the nonlinear flow $G$ to
     its segment endpoint, compute the time-derivative via finite
     differences, and assemble the mismatch vector.
   - The full sparse matrix is factorised with `lu`.

2. **Solve**: the linear system $J \cdot \delta z = b$ is solved directly
   via LU back-substitution (`ldiv!`).

3. **Line search**: starting from $\lambda = 1$ (full Newton step), evaluate
   the residual at $z + \lambda\,\delta z$.  If the residual is lower than
   at the current point, accept the step.  Otherwise, shrink $\lambda$ by
   `ls_rho` and retry, up to `ls_maxiter` times.

4. **Update**: $z \leftarrow z + \lambda\,\delta z$.

The loop terminates when $\|F(z)\| <$ `e_norm_tol`, $\|\delta z\| <$
`dz_norm_tol`, a callback returns `true`, `maxiter` is reached, or the
line search exhausts all backtracking iterations.

### Verbose output columns

**NS == 1 (no spatial shift):**
```
 iter  |   |dz|   |     T     |   ||e||  |     λ     |    res
-------+----------+-----------+----------+-----------+----------
```

**NS == 2 (with spatial shift):**
```
 iter  |  ||dz||  |    T      |     s     |   ||e||  |     λ    |    res
-------+----------+-----------+-----------+----------+----------+----------
```

- **iter**: Newton iteration number.
- **||dz||**: norm of the Newton step $\|\delta z\|$ (before line-search scaling).
- **T**: period.
- **s**: spatial shift (NS == 2 only).
- **||e||**: residual norm $\|F(z)\|$.
- **λ**: step length accepted by the line search ($\leq 1$).
- **res**: linear residual $\|J \cdot \delta z - b\|$ after the LU solve
  (always effectively `0.0` for the direct method, since LU is exact up to
  roundoff).

---

## 9. Threading

**The `:ls_direct` method does NOT support multithreading.**  It throws an
`ArgumentError` if `Threads.nthreads() > 1`.  Run Julia with a single
thread:

```bash
julia --project=. -t 1 script.jl
```

If you need parallelism across shooting segments, use `:ls_iterative` or
`:tr_iterative` instead, which parallelise the GMRES mat-vec products
across threads.

### Why single-threaded?

`DirectSolCache` assembles the Jacobian by applying the linearised flow to
each identity column.  The sparse matrix `A` is mutated in place across
`@spawn` blocks, and the current implementation writes to disjoint regions
of `A` within each spawned task.  However, the `SparseMatrixCSC` data
structure is not designed for concurrent mutation — the internal column
pointers can become inconsistent.  The `DirectSolCache` constructor
explicitly guards against this by checking `Threads.nthreads()`.

---

## 10. Comparison with Other Methods

| Property | `:ls_direct` | `:ls_iterative` | `:tr_iterative` (hookstep) | `:lbfgs_opt` |
|---|---|---|---|---|
| Solves | $F(z) = 0$ directly | $F(z) = 0$ directly | $F(z) = 0$ directly | $\min \frac{1}{2}\|F(z)\|^2$ |
| Globalization | Backtracking line search | Backtracking line search | Trust region | Backtracking line search |
| Linear solve | Direct LU | GMRES (Krylov) | GMRES (Krylov) | None (quasi-Newton) |
| Jacobian | Fully assembled (sparse) | Matrix-free | Matrix-free | Implicit (L-BFGS) |
| Flows needed | 2 (`G`, `L`) | 2 (`G`, `L`) | 2 (`G`, `L`) | 3 (`G`, `L`, `L_adj`) |
| Adjoint required? | No | No | No | Yes |
| Cost per iteration | $\mathcal{O}(N n)$ flow evals per column | $\mathcal{O}(k)$ mat-vecs (GMRES) | $\mathcal{O}(k)$ mat-vecs (GMRES) | 1 forward + 1 adjoint |
| Thread-safe? | No (single-thread only) | Yes | Yes | Yes |
| Best for | Small systems ($n$ small, $N$ small) | Large systems, many segments | Robust convergence from poor guesses | Expensive linearised flows |

**Key trade-off for `:ls_direct`:** the Jacobian assembly cost scales as
$N \cdot n$ (segments $\times$ state dimension), so it is practical only
for **small state spaces and few segments**.  For larger systems, prefer
`:ls_iterative` (matrix-free GMRES) or `:tr_iterative` (trust-region
GMRES).

The **advantage** of `:ls_direct` is that the LU factorisation gives an
**exact** (to roundoff) solution of the Newton linear system, so each
Newton iteration is as accurate as possible.  This can lead to **very fast
convergence** (often 3–5 iterations) when the initial guess is good and
the system is small enough that Jacobian assembly is affordable.

---

## 11. Files Involved

| File | Role |
|---|---|
| `src/newton.jl` | `search!` entry points, dispatch to `_search_linesearch!` |
| `src/search_linesearch.jl` | `_search_linesearch!`, `linesearch`, `e_norm_λ` |
| `src/direct_sol_cache.jl` | `DirectSolCache` — Jacobian assembly (`update!`), LU solve (`_solve`), `op_apply_eye!` |
| `src/multivector.jl` | `MVector` type (seeds + scalar unknowns) |
| `src/options.jl` | `Options` struct (fields: `ls_maxiter`, `ls_rho`, `ϵ`, `fd_order`, etc.) |
| `src/output.jl` | Line-search verbose status table (`display_header_ls`, `display_status_ls`) |
| `test/runtests.jl` | System definitions (`System`, `SystemLinear`, etc.) |
| `test/test_search.jl` | Line-search convergence test (Hopf normal form) |

---

## 12. Complete Minimal Working Example (NS == 1)

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

# --- Forward linearised RHS (5-argument form for Flows.couple) ---
struct HopfLin
    μ::Float64; J::Matrix{Float64}
    HopfLin(μ) = new(μ, zeros(2,2))
end
function (h::HopfLin)(t, u, dudt, v, dvdt)
    x, y = u[1], u[2]; r = sqrt(x^2 + y^2)
    # Fill dudt = F(u)  (nonlinear RHS — required by Flows.couple)
    dudt[1] = -y + h.μ*x*(1 - r)
    dudt[2] =  x + h.μ*y*(1 - r)
    # Fill Jacobian
    h.J[1,1] = h.μ*(1 - r - x^2/r); h.J[1,2] = -1 - h.μ*x*y/r
    h.J[2,1] =  1 - h.μ*x*y/r;      h.J[2,2] = h.μ*(1 - r - y^2/r)
    return mul!(dvdt, h.J, v)
end

# --- Build flows ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)

G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

L = flow(couple(F_sys, HopfLin(μ)),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))

# --- Initial guess (2 segments, slightly perturbed unit circle) ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)

# --- Search (run with julia -t 1) ---
status = search!(G, L, (dxdt, x) -> F_sys(0, x, dxdt), z,
                 Options(method=:ls_direct, maxiter=25,
                         e_norm_tol=1e-12, dz_norm_tol=1e-8,
                         ϵ=1e-7, fd_order=2, verbose=true))

# --- Verify ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
@assert status == :converged
```

---

## 13. Complete Minimal Working Example (NS == 2, with spatial shift)

```julia
using NKSearch, Flows, LinearAlgebra

# --- System, linearised definitions same as NS == 1 ---
# (Hopf, HopfLin as above)

# --- Spatial shift: rotation by angle s (in place) ---
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
    out[1] = -x[2]
    out[2] =  x[1]
    return out
end

# --- Build flows (same as NS == 1) ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)

G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

L = flow(couple(F_sys, HopfLin(μ)),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))

S_op  = SpatialShift()
dS_op = SpatialShiftDerivative()

# --- Initial guess with zero initial shift ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π, 0.0)

# --- Search (6-argument form for NS == 2; run with julia -t 1) ---
status = search!(G, L, S_op, F_sys, dS_op, z,
                 Options(method=:ls_direct, maxiter=25,
                         e_norm_tol=1e-12, dz_norm_tol=1e-8,
                         ϵ=1e-7, fd_order=2, verbose=true))

# --- Verify: orbit lies on unit circle, period is 2π, shift is zero ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
@assert abs(z.d[2]) < 1e-9
@assert status == :converged
```
