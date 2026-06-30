# L-BFGS RPO (NS=2) Support: Missing in NKSearch

## Summary

L-BFGS optimisation (`method=:lbfgs_opt`) in NKSearch does not support
**relative periodic orbits** (RPOs, NS=2).  Only ordinary periodic orbits
(NS=1, no spatial shift) can be used.  Attempting to search for an RPO
with L-BFGS crashes immediately with:

    ErrorException("AdjointIterSolCache: spatial-shift transpose (NS == 2)
    is not yet implemented. Only ordinary periodic orbits (NS == 1) are
    supported.")

This blocks using L-BFGS on the OKF perturbed-IC dataset, since all
orbits in that dataset are RPOs (they have a spatial shift component).

---

## Where the error is thrown

**File:** `NKSearch.jl/src/lbfgs_sol_cache.jl`, line ~228,
in `AdjointIterSolCache.mul!`.

The error was added **intentionally** in commit `0e6579c` (2026-06-19)
with the message:

    Guard NS==2 path with clear error (spatial-shift transpose not yet
    implemented)

Before that commit, the NS==2 code path existed but was untested.  The
commit replaced the NS==2 logic with an explicit `error()` call so that
users get a clear message instead of a cryptic type-mismatch crash.

---

## What needs to happen to enable NS==2

In `lbfgs_sol_cache.jl`, the `mul!` function for `AdjointIterSolCache`
has all the per-segment adjoint integration logic already working for
NS==2 (the `@spawn` blocks with `S(out[i], -z0.d[2])`, the period-row
reduction, etc.).  The only thing missing is the **final assembly** of
the scalar unknowns `out.d` for the NS==2 case.

Specifically, lines ~232–244 currently read:

```julia
    # Spatial-shift row of J^T (NS == 2 only).
    # Forward:  out[N] += dS/ds(xT[N]) · δs
    # Adjoint:  out.d[2] = ⟨w[N], dS/ds(xT[N])⟩  (no negation, no 1/N factor)
    if NS == 2
        D[2](tmp[N], mm.xT[N])
        out_d_2 = dot(w[N], tmp[N])
    end

    # ... (negation loop) ...

    # Spatial-shift transpose not yet implemented; bail out with a clear
    # error instead of a cryptic tuple-type mismatch.
    if NS == 2
        error("AdjointIterSolCache: spatial-shift transpose (NS == 2) ...")
    end
    out.d = (-out_d_1,)
```

The `out_d_2` is computed correctly above (line ~233).  The fix is to
replace the error-guard with the original tuple assembly:

```julia
    if NS == 2
        out.d = (-out_d_1, out_d_2)
    else
        out.d = (-out_d_1,)
    end
```

and then add the phase-locking transpose for the second scalar unknown
(which also already exists in the code at lines ~254–259):

```julia
    if NS == 2
        D[2](tmp[2], z0[1])
        tmp[2] .*= w.d[2]
        out[1] .+= tmp[2]
    end
```

**In summary:** the mechanics are all there — the only change needed is
removing the `error()` call and restoring the NS==2 tuple assignment.
Then the path needs thorough testing (adjoint identity test, gradient
consistency test, convergence test on a known RPO).

---

## Additional L-BFGS issues (lower priority)

1. **No convergence status returned.**  `_search_lbfgs_opt!` always
   returns `nothing` (unlike Newton methods which return `:converged`,
   `:maxiter_reached`, etc.).  Callers must re-compute the residual
   after `search!` returns and compare against `e_norm_tol` manually.

2. **Default `maxiter=10` is too low for L-BFGS.**  The `Options`
   default of 10 iterations is reasonable for Newton but L-BFGS
   typically needs 100–1000.  Users must remember to override it.

---

## Workaround for now

The Newton-Krylov hookstep method (`method=:tr_iterative`) fully
supports RPOs (NS=2) and is already working — see `test_hookstep.jl`.

To test the L-BFGS pipeline *without* spatial shifts, one could run on
a simple ordinary periodic orbit (NS=1), e.g. the Hopf limit cycle from
the NKSearch test suite (`NKSearch.jl/test/runtests.jl`).

---

Generated: 2026-06-30
