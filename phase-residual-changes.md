# Phase-Residual Fix — Summary of Changes

## What changed and why

L-BFGS computes the gradient as $\nabla\phi = J^\top F(z)$. In the old code,
the phase-condition residuals $F_{N+1}$ and $F_{N+2}$ were **forcibly zeroed**
at every evaluation (`b.d = zero.(b.d)`). This meant the phase columns of
$J^\top$ (columns $N+1$ and $N+2$) multiplied zero and contributed nothing
to the gradient — the gauge-restoring force that Newton methods get from
the phase rows was invisible to L-BFGS.

**Fix:** `StageIterCache.update!` now computes actual phase-condition
residuals:

$$F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle$$
$$F_{N+2}(z) = \langle u_1 - u_{\text{ref}},\; \partial_s S(u_{\text{ref}}, 0) \rangle$$

where $u_{\text{ref}} = u_1^{(0)}$ (the initial $u_1$, frozen at cache
construction time). This gives $\nabla\phi$ a first-order component in the
gauge directions sourced from the phase columns of $J^\top$.

**Newton methods are unchanged** — `IterSolCache` and `DirectSolCache` still
use `b.d = zero.(b.d)` because the phase rows of $J$ constrain the Newton
step directly through the linear solve.

---

## Files changed in NKSearch.jl

All changes are in **`src/`** of this repo:

| File | Change |
|---|---|
| `lbfgs_sol_cache.jl` | Added `phase_ref` field to `StageIterCache` and `AdjointIterSolCache`. `StageIterCache.update!` computes actual phase residuals. Forward `mul!` and adjoint `mul!` use `phase_ref` for the phase rows/columns. |
| `iter_sol_cache.jl` | Added `phase_ref` field to `IterSolCache` (stored but not used — `update!` still zeros phase residuals). `mul!` uses `phase_ref` for Jacobian consistency. |
| `direct_sol_cache.jl` | Same as `iter_sol_cache.jl`. |
| `newton.jl` | Passes `fwd_cache.phase_ref` to `AdjointIterSolCache` constructor. |

### Test files changed

| File | Change |
|---|---|
| `test/test_adjoint.jl` | Added `fwd_cache.phase_ref` argument to both `AdjointIterSolCache` constructions. |
| `test/test_parallel.jl` | Fixed `AdjointIterSolCache` construction (was missing `S` and `dxTdT` arguments). |

---

## What you need to sync to your HPC runner

### 1. Push NKSearch.jl

```bash
cd NKSearch.jl
git add -A && git commit -m "phase-ref: compute actual phase-condition residuals for L-BFGS"
git push
```

Make sure the HPC pulls the updated NKSearch.jl (via `Pkg.develop`, a
manifest pin, or however your workflow manages it).

### 2. Fix `test_lbfgs_RPO.jl` in the Iridis repo

In the file that runs the gradient FD check (likely
`Open-Kolmogorov-Flow-Iridis/scripts/test_lbfgs_RPO.jl` or similar), find

```julia
adj_chk = NKSearch.AdjointIterSolCache(
    Ls_adj_chk, D_chk, S,
    fwd_chk.xT, fwd_chk.dxTdT, fwd_chk.z0, fwd_chk.tmp,
    fwd_chk.stage_caches)
```

and add the missing last argument:

```julia
adj_chk = NKSearch.AdjointIterSolCache(
    Ls_adj_chk, D_chk, S,
    fwd_chk.xT, fwd_chk.dxTdT, fwd_chk.z0, fwd_chk.tmp,
    fwd_chk.stage_caches,
    fwd_chk.phase_ref)                          # ← ADD THIS
```

### 3. Re-run

After syncing both repos, the gradient FD check should pass and L-BFGS
will use the full (non-zeroed) phase-condition residuals in the gradient.
