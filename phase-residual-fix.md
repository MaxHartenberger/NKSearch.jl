# Fix: Enable Phase-Condition Residuals for L-BFGS Gradient

## Problem

The current implementation zeros out the scalar (phase-condition) components of
the residual at every evaluation:

```
b.d = zero.(b.d)    # in StageIterCache.update! and IterSolCache.update!
```

where `b.d` holds $F_{N+1}$ (and $F_{N+2}$ for RPOs).  This means

$$F_{N+1}(z) \equiv 0, \qquad F_{N+2}(z) \equiv 0 \qquad \forall z.$$

The augmented Jacobian, however, contains the phase-condition gradient rows

$$J_{N+1,*} = (\,f(u_{\text{ref}})^\top,\; 0, \dots, 0\,), \qquad
  J_{N+2,*} = (\,\partial_s S(u_{\text{ref}}, 0)^\top,\; 0, \dots, 0\,),$$

making $J$ internally inconsistent — the Jacobian row is not the derivative of
the corresponding residual component.

For **Newton methods** this inconsistency is harmless.  The phase rows of $J$
appear in the linear system $J\,\delta z = F$ and constrain the Newton step
$\delta z$ directly, regardless of what value sits in $F_{N+1}$.  The gauge is
pinned by the matrix structure.

For **L-BFGS**, the gradient is

$$\nabla\phi(z) = J(z)^\top F(z).$$

The phase **columns** of $J^\top$ (columns $N+1$ and $N+2$) multiply
$F_{N+1}$ and $F_{N+2}$.  Since these are identically zero, the phase
columns contribute **nothing** to the gradient.  The gradient signal in the
gauge directions comes only from the continuity columns of $J^\top$, which
provide a weaker $O(\|F\|\cdot\delta)$ component (see `gauge-drift-issue.md`
§4.2–4.3).

---

## Fix

Compute the actual phase-condition residuals instead of zeroing them.

### Current (broken for L-BFGS)

```julia
# In update!(cache, b, z):
# ... compute continuity residuals F_1 through F_N ...
b.d = zero.(b.d)   # F_{N+1}, F_{N+2} := 0
```

### Proposed (correct for both Newton and L-BFGS)

```julia
# In update!(cache, b, z):
# ... compute continuity residuals F_1 through F_N ...

# Evaluate phase-condition residuals as proper functions of z
u1  = z.x[1]                            # first shooting point
uref = cache.phase_ref                   # stored once at construction
b.d[1] = dot(u1 - uref, f(uref))        # time-phase residual
b.d[2] = dot(u1 - uref, ∂ₛS(uref, 0))   # space-phase residual  (if applicable)
```

### What must be stored

At cache construction time, store the fixed reference quantities (these are
constant throughout the optimization):

- `u_ref` — the reference state (typically the recurrence state $u_{\text{rec}}$)
- `f_ref = f(u_ref)` — the vector field at the reference
- `∂ₛS_ref = ∂_s S(u_ref, 0)` — the spatial-shift derivative at the reference (RPO only)

These are computed once when the cache is built and reused at every residual
evaluation.

---

## Mathematical effect of the fix

**Before fix.**  $\nabla\phi$ in the gauge direction $w_{\text{time}}$:

$$\langle \nabla\phi, w_{\text{time}} \rangle
   = \sum_{i=1}^{N} \langle F_i,\; \Phi_i f(u_i) - f(u_{i+1}) \rangle
   \;+\; \underbrace{0 \cdot \langle f(u_{\text{ref}}), f(u_1) \rangle}_{=0}.$$

Each continuity term is $O(\|F_i\| \cdot \delta)$ where
$\delta = \|\Phi_i f(u_i) - f(u_{i+1})\|$, which vanishes at a converged orbit.
The gauge-direction gradient is **second-order** near convergence.

**After fix.**  There is an additional term:

$$\langle \nabla\phi, w_{\text{time}} \rangle
   = \sum_{i=1}^{N} \langle F_i,\; \Phi_i f(u_i) - f(u_{i+1}) \rangle
   \;+\; F_{N+1}(z)\,\langle f(u_{\text{ref}}), f(u_1) \rangle,$$

where $F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle$.

Crucially, $F_{N+1}(z)$ is **first-order** in the distance from the solution:
$F_{N+1}(z) = O(\|z - z^*\|)$.  The factor $\langle f(u_{\text{ref}}), f(u_1) \rangle$
does **not** vanish at the solution (it approaches $\|f(u^*_1)\|^2 \neq 0$).
Hence the new contribution is $O(\|F\|)$ — **first-order**, matching the
scaling of non-gauge gradient components.

---

## Why this is safe for Newton methods

Newton methods solve $J\,\delta z = F$.  The phase rows of $J$ constrain
$\delta z$ through the linear solve.  Whether $F_{N+1}$ is $0$ or some small
nonzero value does not affect the *direction* in which the phase row
constrains the step — it only affects the right-hand side seen by that row.

For a Newton method, $F_{N+1} \neq 0$ means the phase condition is treated as
an equation to be driven to zero (like the continuity equations), which is the
mathematically correct formulation.  It does not harm convergence — if
anything, it provides the Newton solver with additional residual information.

For trust-region (hookstep) methods the trust-region radius already limits
step size, so a nonzero $F_{N+1}$ will not cause the solver to take
excessively large steps in the gauge direction.

---

## Implementation notes

### Where to modify

The change is localized to the `update!` methods of the forward caches
(`StageIterCache`, `IterSolCache`, or equivalent) in `NKSearch.jl`.
Specifically, replace

```julia
b.d = zero.(b.d)
```

with a call to evaluate the phase-condition functions using the stored
reference quantities.

### Backward compatibility

The fix is backward-compatible: for Newton and hookstep methods the behavior
is essentially unchanged (the phase row constrains the step, and the RHS
value of that row has negligible effect).  For L-BFGS the gradient becomes
mathematically complete — every row of $J$ contributes to $\nabla\phi$ through
the corresponding residual component that it actually differentiates.

### Verification

After the fix, the following should hold to numerical precision for any
$z$ where the residual is evaluated:

$$F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle,$$
$$\frac{\partial F_{N+1}}{\partial u_1}(z) = f(u_{\text{ref}})^\top.$$

That is, the Jacobian row equals the gradient of the residual component it
corresponds to — restoring internal consistency to the augmented system.
